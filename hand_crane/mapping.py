"""手の情報 (Windows 側から UDP で届く) をロボットの目標姿勢に変換する. ROS に依存しない.

届くデータ (1フレームごと):
    {"hand": true, "x": 0.5, "y": 0.5, "size": 0.18, "pinch": 0.6, "gesture": "paa"}
    x, y   : 手のひら中心の位置 (画面の左上が 0, 右下が 1. 鏡のように左右反転済み)
    size   : 手の大きさ (手首〜中指の付け根の長さ, 画面の高さに対する比). カメラに近いほど大きい
    gesture: "paa" (パー) / "guu" (グー) / "peace" (ピース) / "ok" (OKサイン) / "other"

操作:
    パー      : 手を追いかける (左右→旋回, 上下→高さ, 前後(手の大きさ)→腕の伸び)
    OKサイン  : 追いかけながらグリッパを閉じる (つかむ)
    グー      : その場で止まる (手を動かしても追わない. グリッパはそのまま)
    ピース    : 直立姿勢に戻る
    手が見えない: その場で止まる

2つの操縦方式 (control):
    relative (既定) : 手を「動かした分だけ」ロボットが動く (マウスと同じ).
                      グーで止めて手を画面の真ん中に戻し, パーで続きから動かせる.
    absolute        : 画面上の手の位置がそのままロボットの位置になる.
                      グーで止めて手を動かしてからパーにすると, その位置へ移動する.
"""

import math
from hand_crane import arm_model as am


def interp(v, in_lo, in_hi, out_lo, out_hi):
    t = (v - in_lo) / (in_hi - in_lo)
    t = min(max(t, 0.0), 1.0)
    return out_lo + t * (out_hi - out_lo)


class HandMapper:

    # 画面上の操作範囲 -> ロボットの作業範囲 (absolute の対応, relative の感度もここから決める)
    X_RANGE = (0.15, 0.85)
    Y_RANGE = (0.20, 0.80)
    SIZE_RANGE = (0.12, 0.30)      # 手が小さい(遠い) -> 腕を縮める, 大きい(近い) -> 伸ばす
    TURN_MAX = 90.0                # deg (absolute の左右端)
    Z_RANGE = (0.26, 0.06)         # 画面の上 -> 高く, 下 -> 低く [m]
    R_RANGE = (0.06, 0.22)         # 前後 [m]

    # relative で目標が動いてよい範囲 (届かない分は逆運動学側で近づける)
    TURN_LIMIT = 120.0
    Z_LIMIT = (0.04, 0.30)
    R_LIMIT = (0.04, 0.24)
    START = {'turn': 0.0, 'z': 0.16, 'r': 0.14}   # relative で最初にパーを出したときの位置
    BLEND_DIST = 0.06              # relative: 最初の姿勢から START の姿勢へ移り終えるまでに手を動かす距離 [m]
    BLEND_DEADBAND = 0.01          # relative: これより小さい動きは手のぶれとみなして無視する [m]

    def __init__(self, alpha=0.35, debounce=4, timeout=0.5, invert_turn=False,
                 control='relative', release_debounce=12, sensitivity=1.0):
        if control not in ('relative', 'absolute'):
            raise ValueError("control は 'relative' か 'absolute'")
        self.alpha = alpha          # 平滑化の強さ (小さいほどなめらかで遅れる)
        self.debounce = debounce    # 同じジェスチャーが何フレーム続いたら切り替えるか
        self.release_debounce = release_debounce   # つかんでいる最中に「離す」切り替えに必要なフレーム数
        self.timeout = timeout      # この秒数データが来なければ「手が見えない」扱い
        self.invert_turn = invert_turn
        self.control = control
        self.smooth = None          # 平滑化した (x, y, size)
        self.task = None            # relative: 手先の目標 {'turn', 'z', 'r'}
        self._blend_from = None     # relative: 追従を始めたときの姿勢 (ここから少しずつ移る)
        self._blend = 1.0           # relative: 0 = 始めの姿勢のまま, 1 = 手先の目標どおり
        self.mode = 'hold'
        self._cand, self._count = None, 0
        self.last_time = None
        self.gripper = 'open'
        self.last_pose = dict(am.NEUTRAL)
        self.note = ''
        # 感度: 画面の操作範囲を端から端まで動かすと, absolute と同じだけロボットが動く
        self.k_turn = -2 * self.TURN_MAX / (self.X_RANGE[1] - self.X_RANGE[0])
        self.k_z = (self.Z_RANGE[1] - self.Z_RANGE[0]) / (self.Y_RANGE[1] - self.Y_RANGE[0])
        self.k_r = (self.R_RANGE[1] - self.R_RANGE[0]) / (self.SIZE_RANGE[1] - self.SIZE_RANGE[0])
        # 感度: 小さくすると, 同じ手の動きでロボットが小さく動く (relative のみ)
        self.k_turn *= sensitivity
        self.k_z *= sensitivity
        self.k_r *= sensitivity
        if invert_turn:
            self.k_turn = -self.k_turn

    # ---- 1フレーム分のデータを受け取る ----
    def update(self, pkt, now):
        self.last_time = now
        if not pkt.get('hand'):
            self._switch('lost', force=True)
            self.smooth = None
            return
        g = pkt.get('gesture', 'other')
        want = {'paa': 'follow', 'ok': 'grab', 'guu': 'hold', 'peace': 'home'}.get(g, None)
        if want is not None:
            self._switch(want)
        elif self.mode == 'lost':
            self._switch('hold', force=True)

        if self.mode == 'home':
            self.task = None        # 次にパーを出したら START から始める

        if self.mode in ('follow', 'grab'):
            cur = (float(pkt['x']), float(pkt['y']), float(pkt['size']))
            if self.smooth is None:
                # 追従の開始: ここを基準にする (relative では, この瞬間はロボットを動かさない)
                self.smooth = cur
                if self.task is None:
                    self.task = dict(self.START)
                    self._task0 = dict(self.START)
                    self._blend_from = dict(self.last_pose)
                    self._blend = 0.0
                return
            a = self.alpha
            new = tuple(a * c + (1 - a) * s for c, s in zip(cur, self.smooth))
            if self.control == 'relative':
                dx, dy, ds = (n - s for n, s in zip(new, self.smooth))
                t = self.task
                t['turn'] = min(max(t['turn'] + self.k_turn * dx, -self.TURN_LIMIT), self.TURN_LIMIT)
                t['z'] = min(max(t['z'] + self.k_z * dy, self.Z_LIMIT[0]), self.Z_LIMIT[1])
                t['r'] = min(max(t['r'] + self.k_r * ds, self.R_LIMIT[0]), self.R_LIMIT[1])
                if self._blend < 1.0:
                    t = self.task
                t['turn'] = min(max(t['turn'] + self.k_turn * dx, -self.TURN_LIMIT), self.TURN_LIMIT)
                t['z'] = min(max(t['z'] + self.k_z * dy, self.Z_LIMIT[0]), self.Z_LIMIT[1])
                t['r'] = min(max(t['r'] + self.k_r * ds, self.R_LIMIT[0]), self.R_LIMIT[1])
                if self._blend < 1.0:
                    t0 = self._task0
                    away = math.hypot(t['r'] - t0['r'], t['z'] - t0['z']) \
                        + t['r'] * math.radians(abs(t['turn'] - t0['turn']))
                    b = (away - self.BLEND_DEADBAND) / self.BLEND_DIST
                    self._blend = min(1.0, max(self._blend, b))
            self.smooth = new
        else:
            self.smooth = None      # グー等のあいだの手の動きは無視する

    def _needed(self, want):
        """want へ切り替えるのに必要な連続フレーム数."""
        opens_gripper = want in ('follow', 'home')
        if self.gripper == 'close' and opens_gripper:
            return max(self.debounce, self.release_debounce)   # 離すときだけ慎重に
        return self.debounce

    def _switch(self, want, force=False):
        need = self._needed(want)
        if force or need <= 1:
            self.mode, self._cand, self._count = want, None, 0
            return
        if want == self.mode:
            self._cand, self._count = None, 0
            return
        if want == self._cand:
            self._count += 1
        else:
            self._cand, self._count = want, 1
        if self._count >= need:
            self.mode, self._cand, self._count = want, None, 0

    # ---- 今の目標姿勢を返す ----
    def _goal_absolute(self):
        x, y, size = self.smooth
        turn = interp(x, *self.X_RANGE, self.TURN_MAX, -self.TURN_MAX)   # 画面の右 -> ロボットの右(-)
        if self.invert_turn:
            turn = -turn
        return turn, interp(y, *self.Y_RANGE, *self.Z_RANGE), interp(size, *self.SIZE_RANGE, *self.R_RANGE)

    def _solve(self, turn, r, z, pitch, commit=True):
        pose, shrunk = am.reach_pose(r, z, turn, pitch)
        if pose is None:
            self.note = '届かない位置なので止まっています'
            return None
        if commit:
            self.last_pose = pose
        if shrunk:
            self.note = '届く範囲の端です'
            if self.control == 'relative':
                tr, tz = am.arm_points(pose)[-1]
                self.task['r'], self.task['z'] = tr, tz
        return pose
    
    def target(self, now):
        """戻り値: (pose, gripper('open'/'close'), mode)"""
        if self.last_time is None or now - self.last_time > self.timeout:
            self.mode = 'lost'
            self.smooth = None
        self.note = ''
        if self.mode == 'home':
            self.gripper = 'open'
            self.last_pose = dict(am.NEUTRAL)
        elif self.mode in ('follow', 'grab') and (self.smooth is not None or self.task is not None):
            if self.control == 'absolute':
                if self.smooth is None:
                    return dict(self.last_pose), self.gripper, self.mode
                turn, z, r = self._goal_absolute()
                pitch = interp(z, self.Z_RANGE[1], self.Z_RANGE[0], 150.0, 70.0)
                self._solve(turn, r, z, pitch)
            elif self._blend <= 0.0:
                pass                # 最初にパーを出しただけで, 手はまだ動いていない -> 今の姿勢のまま
            else:
                t = self.task
                pitch = interp(t['z'], self.Z_RANGE[1], self.Z_RANGE[0], 150.0, 70.0)
                start = self._blend_from
                pose = self._solve(t['turn'], t['r'], t['z'], pitch, commit=start is None)
                if pose is not None and start is not None:
                    b = self._blend
                    mixed = {k: (1 - b) * start[k] + b * pose[k] for k in am.KEYS}
                    if am.is_safe(mixed):
                        self.last_pose = mixed
                    if b >= 1.0:
                        self._blend_from = None
            self.gripper = 'close' if self.mode == 'grab' else 'open'
        return dict(self.last_pose), self.gripper, self.mode


def step_toward(cur, target, max_step_deg):
    """関節ごとに max_step_deg 以上は動かさない (速度制限)."""
    out = {}
    for k in am.KEYS:
        d = target[k] - cur[k]
        d = max(-max_step_deg, min(max_step_deg, d))
        out[k] = cur[k] + d
    return out



def step_smooth(cur, vel, target, vmax, amax, dt):
    """速度だけでなく加速度も制限して target へ近づける (なめらかに動き出し, なめらかに止まる)."""
    new, nvel = {}, {}
    for k in am.KEYS:
        d = target[k] - cur[k]
        v0 = vel.get(k, 0.0)
        # 残り距離 |d| をちょうど止まり切れる速度 (20Hz などの刻みでも行き過ぎない)
        ad = amax * dt
        v_stop = ad * (-0.5 + math.sqrt(0.25 + 2.0 * abs(d) / (ad * dt))) if ad > 0 else vmax
        v_des = math.copysign(min(vmax, v_stop), d)
        v = v0 + max(-amax * dt, min(amax * dt, v_des - v0))
        step = v * dt
        if d != 0 and step * d > 0 and abs(step) >= abs(d):   # 目標に届いたらぴったり止める
            step, v = d, d / dt
        new[k] = cur[k] + step
        nvel[k] = v
    return new, nvel