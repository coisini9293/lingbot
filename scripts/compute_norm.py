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
    POT14_DATASET_PATH,
    POT14_NORM_STATS_FILE,
    POT14_VLA_CONFIG,
    VLA_CONFIG,
    get_subprocess_env,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="生成归一化统计量（离线）")
    parser.add_argument(
        "--preset",
        choices=["so100", "pot14"],
        default="so100",
        help="快捷配置：pot14=自采右臂数据",
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--data-name", default=None)
    parser.add_argument("--data-path", default=None, help="本地数据集路径")
    parser.add_argument(
        "--norm-stats-file",
        default=None,
        help="输出路径（相对于 lingbot-vla/）",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--micro-batch-size", type=int, default=8)
    parser.add_argument("--gpu", default="0", help="GPU 编号")
    args = parser.parse_args()

    if args.preset == "pot14":
        config = Path(args.config) if args.config else POT14_VLA_CONFIG
        data_name = args.data_name or "pot14"
        data_path = args.data_path or str(POT14_DATASET_PATH)
        norm_stats = args.norm_stats_file or str(
            POT14_NORM_STATS_FILE.relative_to(LINGBOT_DIR)
        )
    else:
        config = Path(args.config) if args.config else VLA_CONFIG
        data_name = args.data_name or "so100"
        data_path = args.data_path or str(DATASET_PATH)
        norm_stats = args.norm_stats_file or str(
            NORM_STATS_FILE.relative_to(LINGBOT_DIR)
        )

    print("=== 生成归一化统计量 ===")
    print(f"preset: {args.preset}")
    print(f"config: {config}")
    print(f"data_name: {data_name}")
    print(f"数据路径: {data_path}")
    print(f"输出文件: {norm_stats}")

    script_args = [
        str(LINGBOT_DIR / "scripts/compute_norm.py"),
        str(config),
        "--data.data_name", data_name,
        "--data.train_path", data_path,
        "--data.norm_stats_file", norm_stats,
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
