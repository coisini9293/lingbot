#!/usr/bin/env python3
"""生成归一化统计量 - 纯 Python，离线运行"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from launch_utils import run_torchrun
from local_paths import (
    DATASET_PATH,
    LINGBOT_DIR,
    NORM_STATS_FILE,
    VLA_CONFIG,
    get_subprocess_env,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="生成归一化统计量（离线）")
    parser.add_argument("--data-path", default=str(DATASET_PATH), help="本地数据集路径")
    parser.add_argument(
        "--norm-stats-file",
        default=str(NORM_STATS_FILE.relative_to(LINGBOT_DIR)),
        help="输出路径（相对于 lingbot-vla/）",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--micro-batch-size", type=int, default=8)
    parser.add_argument("--gpu", default="0", help="GPU 编号")
    args = parser.parse_args()

    print("=== 生成归一化统计量 ===")
    print(f"数据路径: {args.data_path}")
    print(f"输出文件: {args.norm_stats_file}")

    script_args = [
        str(LINGBOT_DIR / "scripts/compute_norm.py"),
        str(VLA_CONFIG),
        "--data.data_name", "so100",
        "--data.train_path", args.data_path,
        "--data.norm_stats_file", args.norm_stats_file,
        "--data.num_workers", str(args.num_workers),
        "--train.micro_batch_size", str(args.micro_batch_size),
        "--train.output_dir", "output/",
    ]

    run_torchrun(
        script_args,
        cwd=LINGBOT_DIR,
        env=get_subprocess_env(gpu=args.gpu),
    )

    print("=== 归一化统计量生成完成 ===")


if __name__ == "__main__":
    main()
