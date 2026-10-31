"""Windows 側で動かすプログラム: Web カメラの手を MediaPipe で認識し, コンテナへ UDP で送る.

    python hand_sender.py              # Web カメラで操縦
    python hand_sender.py --demo       # カメラなしで, 作り物の手を送って通信を確認
    python hand_sender.py --camera 1   # 別のカメラを使う

ウィンドウを選んだ状態で q か ESC で終了.
"""

import argparse
import json
import math
import os
import socket
import sys
import time
import urllib.request

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)        # どこから起動しても同じフォルダの gesture.py を読めるように
from gesture import analyze, synthetic_hand  # noqa: E402

MODEL_URL = ('https://storage.googleapis.com/mediapipe-models/hand_landmarker/'
             'hand_landmarker/float16/latest/hand_landmarker.task')
LABEL = {'paa': 'OPEN : follow', 'ok': 'OK : grab', 'guu': 'FIST : stop',
         'peace': 'PEACE : home', 'other': '?', None: 'no hand'}
COLOR = {'paa': (80, 220, 80), 'ok': (0, 200, 255), 'guu': (60, 60, 255),
         'peace': (255, 160, 60), 'other': (200, 200, 200), None: (120, 120, 120)}


def load_landmarker(model_path):
    from mediapipe.tasks.python import BaseOptions, vision
    if not os.path.exists(model_path):
        print('手の認識モデルをダウンロードしています...（初回のみ, 約8MB）')
        urllib.request.urlretrieve(MODEL_URL, model_path)
    # フォルダ名に日本語が入っていても読めるよう, パスではなく中身(バイト列)で渡す
    with open(model_path, 'rb') as f:
        buf = f.read()
    opts = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_buffer=buf),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_tracking_confidence=0.5)
    return vision.HandLandmarker.create_from_options(opts), vision.HandLandmarksConnections.HAND_CONNECTIONS


def draw(frame, lms, info, connections, fps, target):
    h, w = frame.shape[:2]
    # 操作範囲の目安 (この枠の中で手を動かす)
    cv2.rectangle(frame, (int(0.15 * w), int(0.2 * h)), (int(0.85 * w), int(0.8 * h)), (90, 90, 90), 1)
    g = info['gesture'] if info else None
    col = COLOR.get(g, (200, 200, 200))
    if lms:
        p = [(int(x * w), int(y * h)) for x, y in lms]
        for c in connections or []:
            cv2.line(frame, p[c.start], p[c.end], col, 2)
        for q in p:
            cv2.circle(frame, q, 4, (255, 255, 255), -1)
        cv2.circle(frame, (int(info['x'] * w), int(info['y'] * h)), 10, col, 2)
    cv2.rectangle(frame, (0, 0), (w, 40), (30, 30, 30), -1)
    cv2.putText(frame, LABEL.get(g, '?'), (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2)
    sub = f'{fps:4.1f} fps  -> {target}'
    if info:
        sub += f'  size {info["size"]:.2f}  pinch {info["pinch"]:.2f}'
    cv2.putText(frame, sub, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def demo_landmarks(t):
    """デモ用: 8の字に動く手. 数秒ごとに パー → OK → グー → ピース と形を変える."""
    cx = 0.5 + 0.25 * math.sin(t * 0.8)
    cy = 0.5 + 0.18 * math.sin(t * 1.6)
    scale = 0.2 + 0.06 * math.sin(t * 0.5)
    phase = int(t // 4) % 4
    if phase == 0:
        return synthetic_hand(cx, cy, scale)
    if phase == 1:
        return synthetic_hand(cx, cy, scale, fingers=('middle', 'ring', 'pinky'), pinch=True)
    if phase == 2:
        return synthetic_hand(cx, cy, scale, fingers=())
    return synthetic_hand(cx, cy, scale, fingers=('index', 'middle'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--host', default='127.0.0.1', help='コンテナ側 (Docker Desktop なら 127.0.0.1)')
    ap.add_argument('--port', type=int, default=5005)
    ap.add_argument('--camera', type=int, default=0)
    ap.add_argument('--demo', action='store_true', help='カメラなしで作り物の手を送る')
    ap.add_argument('--headless', action='store_true', help='ウィンドウを出さない')
    ap.add_argument('--seconds', type=float, default=0, help='この秒数で終了 (0 なら無制限)')
    ap.add_argument('--model', default=os.path.join(HERE, 'hand_landmarker.task'))
    args = ap.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    target = f'{args.host}:{args.port}'
    landmarker, connections, cap = None, None, None
    if not args.demo:
        import mediapipe as mp
        landmarker, connections = load_landmarker(args.model)
        cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW if sys.platform == 'win32' else cv2.CAP_ANY)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not cap.isOpened():
            sys.exit(f'カメラ {args.camera} が開けません。--camera 1 などを試してください')

    print(f'{target} へ送信します。ウィンドウで q / ESC を押すと終了')
    t0 = last = time.monotonic()
    fps = 0.0
    while True:
        now = time.monotonic()
        if args.seconds and now - t0 > args.seconds:
            break
        if args.demo:
            frame = np.zeros((480, 640, 3), np.uint8)
            lms = demo_landmarks(now - t0)
            time.sleep(1 / 30)
        else:
            ok, frame = cap.read()
            if not ok:
                print('カメラ画像が取れません'); break
            frame = cv2.flip(frame, 1)                 # 鏡のように左右反転 (操作しやすい)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res = landmarker.detect_for_video(image, int((now - t0) * 1000))
            lms = [(p.x, p.y) for p in res.hand_landmarks[0]] if res.hand_landmarks else None

        h, w = frame.shape[:2]
        info = analyze(lms, aspect=w / h) if lms else None
        pkt = {'hand': info is not None, 't': now}
        if info:
            pkt.update({k: round(info[k], 4) for k in ('x', 'y', 'size', 'pinch')})
            pkt['gesture'] = info['gesture']
        sock.sendto(json.dumps(pkt).encode('utf-8'), (args.host, args.port))

        dt = now - last
        last = now
        fps = 0.9 * fps + 0.1 * (1 / dt if dt > 0 else 0)
        if not args.headless:
            draw(frame, lms, info, connections, fps, target)
            cv2.imshow('hand_sender (q / ESC: quit)', frame)
            if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                break

    if cap:
        cap.release()
    cv2.destroyAllWindows() if not args.headless else None


if __name__ == '__main__':
    main()
