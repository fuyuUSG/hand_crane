"""MediaPipe の手の 21 点 (ランドマーク) から, 手の位置・大きさ・ジェスチャーを求める.

ランドマーク番号 (MediaPipe Hands):
    0 手首 / 1-4 親指 / 5-8 人差し指 / 9-12 中指 / 13-16 薬指 / 17-20 小指
    (各指: 付け根 MCP, 第2関節 PIP, 第1関節 DIP, 指先 TIP の順)
"""

import math

WRIST = 0
THUMB_TIP = 4
FINGERS = {  # 名前: (MCP, PIP, TIP)
    'index': (5, 6, 8),
    'middle': (9, 10, 12),
    'ring': (13, 14, 16),
    'pinky': (17, 18, 20),
}
PALM = (0, 5, 9, 13, 17)


def _d(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def analyze(landmarks, aspect=4 / 3):
    """landmarks: [(x, y), ...] 21点 (0〜1 に正規化された画像座標).
    aspect: 画像の 幅/高さ. 距離を正しく測るため x をこの値で引きのばす.

    戻り値: {"x", "y", "size", "pinch", "fingers", "gesture"}
    """
    pts = [(x * aspect, y) for x, y in landmarks]
    wrist = pts[WRIST]
    size = _d(wrist, pts[9])                      # 手首〜中指の付け根
    if size < 1e-6:
        size = 1e-6

    extended = {}
    for name, (mcp, pip, tip) in FINGERS.items():
        # 指先が第2関節より手首から十分遠ければ「伸びている」(手の向きによらない判定)
        extended[name] = _d(wrist, pts[tip]) > _d(wrist, pts[pip]) * 1.15
    pinch = _d(pts[THUMB_TIP], pts[8]) / size     # 親指と人差し指の先の距離 (手の大きさで割る)

    n_ext = sum(extended.values())
    others = extended['middle'] + extended['ring'] + extended['pinky']
    if pinch < 0.35 and others >= 2:
        gesture = 'ok'          # OKサイン: 親指と人差し指で輪, 他は伸ばす
    elif n_ext == 0:
        gesture = 'guu'
    elif extended['index'] and extended['middle'] and not extended['ring'] and not extended['pinky']:
        gesture = 'peace'
    elif n_ext >= 4:
        gesture = 'paa'
    else:
        gesture = 'other'

    cx = sum(landmarks[i][0] for i in PALM) / len(PALM)
    cy = sum(landmarks[i][1] for i in PALM) / len(PALM)
    return {'x': cx, 'y': cy, 'size': size, 'pinch': pinch,
            'fingers': extended, 'gesture': gesture}


def synthetic_hand(cx, cy, scale, fingers=('index', 'middle', 'ring', 'pinky'), pinch=False):
    """テスト・デモ用の作り物の手 (手のひらを正面に向けて立てた形).

    cx, cy: 手のひらの中心, scale: 手首〜中指付け根の長さ (画面の高さ比)
    fingers: 伸ばす指, pinch: True なら親指と人差し指の先をくっつける
    """
    s = scale
    wrist = (cx, cy + 0.6 * s)
    pts = [None] * 21
    pts[0] = wrist
    bases = {'index': -0.35, 'middle': -0.1, 'ring': 0.12, 'pinky': 0.32}
    for name, (mcp, pip, tip) in FINGERS.items():
        bx = cx + bases[name] * s
        by = cy - 0.4 * s
        pts[mcp] = (bx, by)
        if name in fingers:
            pts[pip], pts[pip + 1], pts[tip] = (bx, by - 0.35 * s), (bx, by - 0.6 * s), (bx, by - 0.8 * s)
        else:   # 曲げた指: 手のひら側へ折り返す
            pts[pip], pts[pip + 1], pts[tip] = (bx, by - 0.2 * s), (bx, by + 0.05 * s), (bx, by + 0.2 * s)
    # 親指
    pts[1] = (cx - 0.45 * s, cy + 0.35 * s)
    pts[2] = (cx - 0.6 * s, cy + 0.1 * s)
    pts[3] = (cx - 0.7 * s, cy - 0.1 * s)
    pts[4] = (cx - 0.75 * s, cy - 0.3 * s)
    if pinch:
        ix, iy = pts[8]
        pts[4] = (ix + 0.05 * s, iy + 0.05 * s)
    return pts
