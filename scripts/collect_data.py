#!/usr/bin/env python3
"""数据采集脚本 - 采集遥操作数据用于训练"""

import argparse
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description="采集遥操作数据用于训练")
    parser.add_argument(
        "--task",
        default="将绿色方块放到纸盒里",
        help="任务描述",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=5,
        help="采集轮次",
    )
    parser.add_argument(
        "--repo-id",
        default="hexchip/record-test",
        help="数据集ID",
    )
    args = parser.parse_args()

    print("=== 开始数据采集 ===")
    print(f"任务: {args.task}")
    print(f"采集轮次: {args.num_episodes}")
    print(f"数据集ID: {args.repo_id}")

    cmd = [
        "lerobot-record",
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
        f"--dataset.repo_id={args.repo_id}",
        f"--dataset.num_episodes={args.num_episodes}",
        f'--dataset.single_task={args.task}',
        "--dataset.streaming_encoding=true",
        "--dataset.encoder_threads=2",
    ]

    subprocess.run(cmd, check=True)

    print("=== 数据采集完成 ===")


if __name__ == "__main__":
    main()
