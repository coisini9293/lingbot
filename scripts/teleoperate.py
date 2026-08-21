#!/usr/bin/env python3
"""遥操作脚本 - 用于测试机械臂和摄像头"""

import subprocess
import sys


def main():
    print("=== 启动遥操作测试 ===")

    cmd = [
        "lerobot-teleoperate",
        "--robot.type=so100_follower",
        "--robot.port=/dev/ttyACM0",
        "--robot.id=so100_follower_arm",
        '--robot.cameras={'
        'camera_top: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, '
        'camera_wrist_left: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}, '
        'camera_wrist_right: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}'
        '}',
        "--teleop.type=so100_leader",
        "--teleop.port=/dev/ttyACM1",
        "--teleop.id=so100_leader_arm",
        "--display_data=true",
    ]

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
