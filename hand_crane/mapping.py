"""手の情報 (Windows 側から UDP で届く) をロボットの目標姿勢に変換する. ROS に依存しない.

届くデータ (1フレームごと):
    {"hand": true, "x": 0.5, "y": 0.5, "size": 0.18, "pinch": 0.6, "gesture": "paa"}
    x, y   : 手のひら中心の位置 (画面の左上が 0, 右下が 1. 鏡のように左右反転済み)
    size   : 手の大きさ (手首〜中指の付け根の長さ, 画面の高さに対する比). カメラに近いほど大きい
    gesture: "paa" (パー) / "guu" (グー) / "peace" (ピース) / "ok" (OKサイン) / "other"

操作:
    パー      : 手を追いかける (左右→旋回, 上下→高さ, 前後(手の大きさ)→腕の伸び)
    OKサイン  : 追いかけながらグリッパを閉じる (つかむ)
    グー      : その場で止まる (手を動かしても追わない. 手の位置を戻したいときに使う)
    ピース    : 直立姿勢に戻る
    手が見えない: その場で止まる
"""

from hand_crane import arm_model as am


def interp(v, in_lo, in_hi, out_lo, out_hi):
    t = (v - in_lo) / (in_hi - in_lo)
    t = min(max(t, 0.0), 1.0)
    return out_lo + t * (out_hi - out_lo)


class HandMapper:

    # 画面上の操作範囲 -> ロボットの作業範囲
    X_RANGE = (0.15, 0.85)
    Y_RANGE = (0.20, 0.80)
    SIZE_RANGE = (0.12, 0.30)      # 手が小さい(遠い) -> 腕を縮める, 大きい(近い) -> 伸ばす
    TURN_MAX = 90.0                # deg
    Z_RANGE = (0.26, 0.06)         # 画面の上 -> 高く, 下 -> 低く [m]
    R_RANGE = (0.06, 0.22)         # 前後 [m]

    def __init__(self, alpha=0.35, debounce=4, timeout=0.5, invert_turn=False):
        self.alpha = alpha          # 平滑化の強さ (小さいほどなめらかで遅れる)
        self.debounce = debounce    # 同じジェスチャーが何フレーム続いたら切り替えるか
        self.timeout = timeout      # この秒数データが来なければ「手が見えない」扱い
        self.invert_turn = invert_turn
        self.smooth = None          # 平滑化した (x, y, size)
        self.mode = 'hold'
        self._cand, self._count = None, 0
        self.last_time = None
        self.gripper = 'open'
        self.last_pose = dict(am.NEUTRAL)
        self.note = ''

    # ---- 1フレーム分のデータを受け取る ----
    def update(self, pkt, now):
        self.last_time = now
        if not pkt.get('hand'):
            self._switch('lost', force=True)
            return
        g = pkt.get('gesture', 'other')
        want = {'paa': 'follow', 'ok': 'grab', 'guu': 'hold', 'peace': 'home'}.get(g, None)
        if want is not None:
            self._switch(want)
        elif self.mode == 'lost':
            self._switch('hold', force=True)

        if self.mode in ('follow', 'grab'):
            cur = (float(pkt['x']), float(pkt['y']), float(pkt['size']))
            if self.smooth is None:
                self.smooth = cur
            else:
                a = self.alpha
                self.smooth = tuple(a * c + (1 - a) * s for c, s in zip(cur, self.smooth))
        else:
            self.smooth = None      # 追従を再開したとき, 古い位置から急に動かないように

    def _switch(self, want, force=False):
        if force or self.debounce <= 1:
            self.mode, self._cand, self._count = want, None, 0
            return
        if want == self.mode:
            self._cand, self._count = None, 0
            return
        if want == self._cand:
            self._count += 1
        else:
            self._cand, self._count = want, 1
        if self._count >= self.debounce:
            self.mode, self._cand, self._count = want, None, 0

    # ---- 今の目標姿勢を返す ----
    def target(self, now):
        """戻り値: (pose, gripper('open'/'close'), mode)"""
        if self.last_time is None or now - self.last_time > self.timeout:
            self.mode = 'lost'
        self.note = ''
        if self.mode == 'home':
            self.gripper = 'open'
            self.last_pose = dict(am.NEUTRAL)
        elif self.mode in ('follow', 'grab') and self.smooth is not None:
            x, y, size = self.smooth
            turn = interp(x, *self.X_RANGE, self.TURN_MAX, -self.TURN_MAX)   # 画面の右 -> ロボットの右(-)
            if self.invert_turn:
                turn = -turn
            z = interp(y, *self.Y_RANGE, *self.Z_RANGE)
            r = interp(size, *self.SIZE_RANGE, *self.R_RANGE)
            pitch = interp(z, self.Z_RANGE[1], self.Z_RANGE[0], 150.0, 70.0)  # 低いほど手先を下に向ける
            pose, shrunk = am.reach_pose(r, z, turn, pitch)
            if pose is None:
                self.note = '届かない位置なので止まっています'
            else:
                self.last_pose = pose
                if shrunk:
                    self.note = '届く範囲まで目標を近づけました'
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
