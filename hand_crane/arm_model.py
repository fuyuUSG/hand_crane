"""CRANE+ V2 の運動学 (順運動学・逆運動学) と安全チェック. ROS に依存しない.

姿勢は「意味的な角度 (度)」で表す:
    turn  : 土台の旋回.  + でロボットから見て左
    lean  : 肩.          + で前 (+x) に倒れる
    elbow : ひじ.        + で前に曲がる
    wrist : 手首.        + で前 (下向き) に曲がる
URDF では joint3, joint4 の回転軸が joint2 と逆 (-y) なので, 関節角への変換で符号を反転する.
"""

import math

ARM_JOINT_NAMES = ['crane_plus_joint1', 'crane_plus_joint2',
                   'crane_plus_joint3', 'crane_plus_joint4']
GRIPPER_JOINT_NAME = 'crane_plus_joint_hand'
GRIPPER_OPEN = -0.5236    # -30 deg (physicalai_teleop と同じ値)
GRIPPER_CLOSE = 0.1745    # +10 deg

SHOULDER_HEIGHT = 0.0454 + 0.026   # 土台上面 -> joint2 [m]
UPPER_ARM = 0.083                  # joint2 -> joint3
FOREARM = 0.0935                   # joint3 -> joint4
HAND = 0.10                        # joint4 -> 指先 (概算)

SAFE_RANGE = {'turn': (-120.0, 120.0), 'lean': (-60.0, 80.0),
              'elbow': (-90.0, 120.0), 'wrist': (-90.0, 100.0)}
MIN_HEIGHT = 0.03                  # 机から各関節・指先までの最低の高さ [m]
KEYS = ('turn', 'lean', 'elbow', 'wrist')
NEUTRAL = {'turn': 0.0, 'lean': 0.0, 'elbow': 0.0, 'wrist': 0.0}


def to_joint_radians(pose):
    return [math.radians(pose['turn']), math.radians(pose['lean']),
            -math.radians(pose['elbow']), -math.radians(pose['wrist'])]


def from_joint_radians(j):
    return {'turn': math.degrees(j[0]), 'lean': math.degrees(j[1]),
            'elbow': -math.degrees(j[2]), 'wrist': -math.degrees(j[3])}


def arm_points(pose):
    """側面から見た [肩, ひじ, 手首, 指先] の (前後 r, 高さ z) [m]."""
    a1 = math.radians(pose['lean'])
    a2 = a1 + math.radians(pose['elbow'])
    a3 = a2 + math.radians(pose['wrist'])
    pts = [(0.0, SHOULDER_HEIGHT)]
    for length, ang in ((UPPER_ARM, a1), (FOREARM, a2), (HAND, a3)):
        r, z = pts[-1]
        pts.append((r + length * math.sin(ang), z + length * math.cos(ang)))
    return pts


def in_range(pose):
    return all(SAFE_RANGE[k][0] <= pose[k] <= SAFE_RANGE[k][1] for k in KEYS)


def is_safe(pose):
    return in_range(pose) and all(z >= MIN_HEIGHT for _, z in arm_points(pose)[1:])


def _two_link_ik(r, z):
    """肩から手首 (joint4) の点 (r, z) への 2リンク逆運動学. 解のリスト [(lean, elbow)] [deg]."""
    vr, vz = r, z - SHOULDER_HEIGHT
    d2 = vr * vr + vz * vz
    c = (d2 - UPPER_ARM ** 2 - FOREARM ** 2) / (2 * UPPER_ARM * FOREARM)
    if c < -1.0 or c > 1.0:
        return []
    theta = math.atan2(vr, vz)              # 垂直から前へ測った角度
    sols = []
    for sign in (1.0, -1.0):                # +: ひじを前に曲げる解を優先
        elbow = sign * math.acos(c)
        lean = theta - math.atan2(FOREARM * math.sin(elbow), UPPER_ARM + FOREARM * math.cos(elbow))
        sols.append((math.degrees(lean), math.degrees(elbow)))
    return sols


def solve_ik(r, z, turn, pitch_pref):
    """指先を (r, z) に, 手の向きをなるべく pitch_pref [deg, 垂直から前へ] にする姿勢を探す.

    手の向きを pitch_pref から少しずつずらしながら, 可動範囲内かつ机にぶつからない解を探す.
    見つからなければ None.
    """
    for off in (0, 10, -10, 20, -20, 30, -30, 45, -45, 60, -60):
        pitch = math.radians(pitch_pref + off)
        wr = r - HAND * math.sin(pitch)
        wz = z - HAND * math.cos(pitch)
        for lean, elbow in _two_link_ik(wr, wz):
            wrist = math.degrees(pitch) - lean - elbow
            pose = {'turn': turn, 'lean': lean, 'elbow': elbow, 'wrist': wrist}
            if is_safe(pose):
                return pose
    return None


def reach_pose(r, z, turn, pitch_pref, shrink_steps=15):
    """届かない目標は肩の方へ少しずつ近づけて解く. 戻り値 (pose or None, 目標を縮めたか)."""
    turn = min(max(turn, SAFE_RANGE['turn'][0]), SAFE_RANGE['turn'][1])
    dr, dz = r, z - SHOULDER_HEIGHT
    reach = 0.97 * (UPPER_ARM + FOREARM + HAND)
    dist = math.hypot(dr, dz)
    pre_shrunk = dist > reach
    if pre_shrunk:                          # まず腕の長さの円の中まで一気に縮める
        dr, dz = dr * reach / dist, dz * reach / dist
    for i in range(shrink_steps + 1):
        s = 0.95 ** i
        pose = solve_ik(dr * s, SHOULDER_HEIGHT + dz * s, turn, pitch_pref)
        if pose is not None:
            return pose, pre_shrunk or i > 0
    return None, True
