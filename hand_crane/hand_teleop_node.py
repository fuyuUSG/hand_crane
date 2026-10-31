#!/usr/bin/env python3
"""手のジェスチャーで CRANE+ V2 を操縦する ROS 2 ノード (コンテナ側).

Windows 側の hand_sender.py (MediaPipe で手を認識) から UDP で届く手の情報を受け取り,
20Hz で JointTrajectory を publish する.

    ros2 run hand_crane hand_teleop_node
    ros2 run hand_crane hand_teleop_node --ros-args -p invert_turn:=true   # 左右が逆に感じるとき
"""

import json
import socket
import threading
import time

import rclpy
from rclpy.node import Node
from builtin_interfaces.msg import Duration
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from hand_crane import arm_model as am
from hand_crane.mapping import HandMapper, step_toward

ARM_TOPIC = '/crane_plus_arm_controller/joint_trajectory'
GRIPPER_TOPIC = '/crane_plus_gripper_controller/joint_trajectory'
MODE_JA = {'follow': '追従(パー)', 'grab': 'つかむ(OK)', 'hold': '停止(グー)',
           'home': '直立へ(ピース)', 'lost': '手が見えない→停止'}


class HandTeleop(Node):

    def __init__(self):
        super().__init__('hand_teleop')
        self.declare_parameter('port', 5005)
        self.declare_parameter('rate', 20.0)
        self.declare_parameter('max_speed_deg', 60.0)   # 1秒あたりの最大角度
        self.declare_parameter('invert_turn', False)
        self.declare_parameter('smoothing', 0.35)
        self.declare_parameter('debounce', 4)

        self.mapper = HandMapper(
            alpha=self.get_parameter('smoothing').value,
            debounce=self.get_parameter('debounce').value,
            invert_turn=self.get_parameter('invert_turn').value)
        self.rate = float(self.get_parameter('rate').value)
        self.max_step = float(self.get_parameter('max_speed_deg').value) / self.rate

        self.arm_pub = self.create_publisher(JointTrajectory, ARM_TOPIC, 10)
        self.grip_pub = self.create_publisher(JointTrajectory, GRIPPER_TOPIC, 10)
        self.create_subscription(JointState, '/joint_states', self._on_js, 10)
        self.cmd = None                 # 今送っている姿勢 (最初は /joint_states から)
        self.lock = threading.Lock()
        self.packets = 0
        self.last_gesture = '-'

        port = int(self.get_parameter('port').value)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(('0.0.0.0', port))
        self.sock.settimeout(0.2)
        threading.Thread(target=self._recv_loop, daemon=True).start()
        self.create_timer(1.0 / self.rate, self._tick)
        self.create_timer(0.5, self._status)
        self.get_logger().info(f'UDP {port} 番で Windows 側の手の情報を待っています')

    def _on_js(self, msg):
        if self.cmd is not None:
            return
        pos = dict(zip(msg.name, msg.position))
        if all(n in pos for n in am.ARM_JOINT_NAMES):
            self.cmd = am.from_joint_radians([pos[n] for n in am.ARM_JOINT_NAMES])

    def _recv_loop(self):
        while rclpy.ok():
            try:
                data, _ = self.sock.recvfrom(4096)
                pkt = json.loads(data.decode('utf-8'))
            except socket.timeout:
                continue
            except (ValueError, OSError):
                continue
            with self.lock:
                self.mapper.update(pkt, time.monotonic())
                self.packets += 1
                self.last_gesture = pkt.get('gesture', '-') if pkt.get('hand') else 'なし'

    def _tick(self):
        with self.lock:
            target, grip, mode = self.mapper.target(time.monotonic())
        if self.cmd is None:
            self.cmd = dict(am.NEUTRAL)
        self.cmd = step_toward(self.cmd, target, self.max_step)
        self._publish(self.cmd, grip)

    def _publish(self, pose, grip):
        dt = Duration(sec=0, nanosec=int(2.0 / self.rate * 1e9))   # 2周期ぶんで到達
        arm = JointTrajectory()
        arm.joint_names = list(am.ARM_JOINT_NAMES)
        p = JointTrajectoryPoint()
        p.positions = [float(v) for v in am.to_joint_radians(pose)]
        p.time_from_start = dt
        arm.points.append(p)
        g = JointTrajectory()
        g.joint_names = [am.GRIPPER_JOINT_NAME]
        pg = JointTrajectoryPoint()
        pg.positions = [am.GRIPPER_CLOSE if grip == 'close' else am.GRIPPER_OPEN]
        pg.time_from_start = dt
        g.points.append(pg)
        self.arm_pub.publish(arm)
        self.grip_pub.publish(g)

    def _status(self):
        with self.lock:
            mode, n, ges, note = self.mapper.mode, self.packets, self.last_gesture, self.mapper.note
            self.packets = 0
        c = self.cmd or am.NEUTRAL
        print(f'\r[{MODE_JA.get(mode, mode):<12}] 受信 {n * 2:3d}fps  手:{ges:<5} '
              f'turn {c["turn"]:6.1f} lean {c["lean"]:6.1f} elbow {c["elbow"]:6.1f} '
              f'wrist {c["wrist"]:6.1f}  {note}', end='   ', flush=True)


def main():
    rclpy.init()
    node = HandTeleop()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.sock.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
