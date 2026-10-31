"""単体テスト (ROS もカメラも不要).  python3 -m pytest test -q"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'windows'))

from gesture import analyze, synthetic_hand          # noqa: E402
from hand_crane import arm_model as am               # noqa: E402
from hand_crane.mapping import HandMapper, step_toward  # noqa: E402


# ---------------- gesture.py ----------------
def test_gestures_from_synthetic_hands():
    assert analyze(synthetic_hand(0.5, 0.5, 0.2))['gesture'] == 'paa'
    assert analyze(synthetic_hand(0.5, 0.5, 0.2, fingers=()))['gesture'] == 'guu'
    assert analyze(synthetic_hand(0.5, 0.5, 0.2, fingers=('index', 'middle')))['gesture'] == 'peace'
    ok = synthetic_hand(0.5, 0.5, 0.2, fingers=('middle', 'ring', 'pinky'), pinch=True)
    assert analyze(ok)['gesture'] == 'ok'


def test_gesture_is_rotation_invariant():
    pts = synthetic_hand(0.5, 0.5, 0.2, fingers=())
    c, s = math.cos(1.0), math.sin(1.0)
    rot = [(0.5 + c * (x - 0.5) - s * (y - 0.5), 0.5 + s * (x - 0.5) + c * (y - 0.5)) for x, y in pts]
    assert analyze(rot, aspect=1.0)['gesture'] == 'guu'


def test_size_and_center():
    info = analyze(synthetic_hand(0.3, 0.6, 0.25), aspect=1.0)
    assert abs(info['size'] - 0.25 * math.hypot(0.1, 1.0)) < 0.02
    assert abs(info['x'] - 0.3) < 0.05 and abs(info['y'] - 0.6) < 0.1


# ---------------- arm_model.py ----------------
def test_ik_reaches_target():
    for r, z in [(0.15, 0.10), (0.10, 0.20), (0.20, 0.06), (0.06, 0.25)]:
        pose, _ = am.reach_pose(r, z, 0.0, 120.0)
        assert pose is not None and am.is_safe(pose)
        tr, tz = am.arm_points(pose)[-1]
        if not _:
            assert math.hypot(tr - r, tz - z) < 1e-6


def test_unreachable_target_is_shrunk():
    pose, shrunk = am.reach_pose(0.5, 0.5, 0.0, 90.0)
    assert pose is not None and shrunk and am.is_safe(pose)


def test_ik_never_returns_unsafe_pose():
    for r in [0.0, 0.05, 0.1, 0.15, 0.2, 0.25]:
        for z in [0.0, 0.03, 0.08, 0.15, 0.25, 0.35]:
            pose, _ = am.reach_pose(r, z, 30.0, 150.0)
            assert pose is None or am.is_safe(pose)


# ---------------- mapping.py ----------------
def _pkt(g='paa', x=0.5, y=0.5, size=0.2):
    return {'hand': True, 'x': x, 'y': y, 'size': size, 'pinch': 1.0, 'gesture': g}


def test_absolute_left_right_up_down():
    m = HandMapper(debounce=1, alpha=1.0, control='absolute')
    m.update(_pkt(x=0.85), 0.0)
    right, _, mode = m.target(0.0)
    assert mode == 'follow' and right['turn'] < -80
    m.update(_pkt(x=0.15), 0.1)
    left, _, _ = m.target(0.1)
    assert left['turn'] > 80
    m.update(_pkt(y=0.2), 0.2)
    high = am.arm_points(m.target(0.2)[0])[-1][1]
    m.update(_pkt(y=0.8), 0.3)
    low = am.arm_points(m.target(0.3)[0])[-1][1]
    assert high > low + 0.1


def test_invert_turn():
    m = HandMapper(debounce=1, alpha=1.0, invert_turn=True, control='absolute')
    m.update(_pkt(x=0.85), 0.0)
    assert m.target(0.0)[0]['turn'] > 80


def test_fist_holds_and_peace_goes_home():
    m = HandMapper(debounce=1, alpha=1.0)
    m.update(_pkt(x=0.5), 0.0)
    m.update(_pkt(x=0.8), 0.0)
    held = m.target(0.0)[0]
    m.update(_pkt('guu', x=0.2), 0.1)
    pose, _, mode = m.target(0.1)
    assert mode == 'hold' and pose == held
    m.update(_pkt('peace'), 0.2)
    pose, grip, mode = m.target(0.2)
    assert mode == 'home' and pose == am.NEUTRAL and grip == 'open'


def test_ok_closes_gripper():
    m = HandMapper(debounce=1, alpha=1.0)
    m.update(_pkt('ok'), 0.0)
    assert m.target(0.0)[1] == 'close'


def test_debounce_ignores_single_frame_glitch():
    m = HandMapper(debounce=4, alpha=1.0)
    for i in range(5):
        m.update(_pkt('paa'), i * 0.03)
    assert m.mode == 'follow'
    m.update(_pkt('guu'), 0.2)              # 1フレームだけグーに誤認識
    m.update(_pkt('paa'), 0.23)
    assert m.mode == 'follow'


def test_timeout_means_lost():
    m = HandMapper(debounce=1, alpha=1.0)
    m.update(_pkt(x=0.5), 0.0)
    m.update(_pkt(x=0.8), 0.0)
    before = m.target(0.0)[0]
    pose, _, mode = m.target(1.0)
    assert mode == 'lost' and pose == before


def test_step_toward_limits_speed():
    out = step_toward(dict(am.NEUTRAL), {'turn': 90, 'lean': -5, 'elbow': 0, 'wrist': 0}, 3.0)
    assert out['turn'] == 3.0 and out['lean'] == -3.0


# ---------------- relative (マウス式) ----------------
def _tip_z(m, t):
    return am.arm_points(m.target(t)[0])[-1][1]


def test_relative_first_open_hand_does_not_jump_to_hand_position():
    m = HandMapper(debounce=1, alpha=1.0)
    m.update(_pkt(x=0.85, y=0.8), 0.0)      # 画面の右下でパーを出しても
    pose, _, _ = m.target(0.0)
    assert abs(pose['turn']) < 1e-6          # 右端へ飛ばず START から始まる


def test_relative_moves_by_hand_motion():
    m = HandMapper(debounce=1, alpha=1.0)
    m.update(_pkt(x=0.5), 0.0)
    m.update(_pkt(x=0.6), 0.1)              # 右へ 0.1
    assert m.target(0.1)[0]['turn'] < -20   # ロボットの右へ旋回


def test_relative_clutch_like_a_mouse():
    """ユーザーの報告した操作: パーで上へ → グーで止めて手を下へ → パーで再開しても下がらない."""
    m = HandMapper(debounce=1, alpha=1.0)
    m.update(_pkt(y=0.5), 0.0)
    m.update(_pkt(y=0.25), 0.1)             # パーで上へ
    z_up = _tip_z(m, 0.1)
    m.update(_pkt('guu', y=0.25), 0.2)
    m.update(_pkt('guu', y=0.8), 0.3)       # グーのまま手を下へ戻す
    assert abs(_tip_z(m, 0.3) - z_up) < 1e-6
    m.update(_pkt('paa', y=0.8), 0.4)       # パーで再開
    assert abs(_tip_z(m, 0.4) - z_up) < 1e-6    # 下に飛ばない
    m.update(_pkt('paa', y=0.6), 0.5)       # さらに上へ動かすと
    assert _tip_z(m, 0.5) > z_up                # 続きから上がる


def test_relative_no_windup_at_workspace_edge():
    m = HandMapper(debounce=1, alpha=1.0)
    m.update(_pkt(size=0.2), 0.0)
    for i in range(1, 30):                  # 届かないほど前へ伸ばし続ける
        m.update(_pkt(size=0.2 + 0.05 * i), i * 0.03)
        m.target(i * 0.03)
    edge = am.arm_points(m.target(1.0)[0])[-1][0]
    m.update(_pkt(size=1.65), 1.0)
    m.update(_pkt(size=1.60), 1.03)         # 少し戻すと
    back = am.arm_points(m.target(1.03)[0])[-1][0]
    assert back < edge - 1e-3               # すぐに戻り始める (空振りしない)


def test_relative_ok_then_fist_keeps_gripper_closed():
    m = HandMapper(debounce=1, alpha=1.0)
    m.update(_pkt('ok'), 0.0)
    assert m.target(0.0)[1] == 'close'
    m.update(_pkt('guu'), 0.1)
    assert m.target(0.1)[1] == 'close'
