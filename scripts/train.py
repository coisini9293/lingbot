#!/usr/bin/env python3
"""训练脚本 - 纯 Python，本地离线微调 Lingbot VLA"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from launch_utils import run_torchrun
from local_paths import (
    DATASET_PATH,
    LINGBOT_DIR,
    LINGBOT_VLA_MODEL,
    NORM_STATS_FILE,
    POT14_DATASET_PATH,
    POT14_NORM_STATS_FILE,
    POT14_VLA_CONFIG,
    QWEN_VL_MODEL,
    VLA_CONFIG,
    get_subprocess_env,
    recommended_num_workers,
    validate_local_models,
    validate_norm_stats_file,
    check_gpu_free_memory,
    check_disk_free_space,
    validate_training_env,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="训练 Lingbot VLA（本地离线）")
    parser.add_argument(
        "--preset",
        choices=["so100", "pot14"],
        default="so100",
        help="快捷配置：pot14=自采右臂数据",
    )
    parser.add_argument("--config", default=None, help="VLA yaml 路径，默认随 --preset")
    parser.add_argument("--data-name", default=None, help="robot_configs/<name>.yaml 的 name")
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--norm-stats-file", default=None)
    parser.add_argument("--model-path", default=str(LINGBOT_VLA_MODEL))
    parser.add_argument("--tokenizer-path", default=str(QWEN_VL_MODEL))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--save-steps", type=int, default=None)
    parser.add_argument("--micro-batch-size", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument(
        "--pin-memory",
        choices=["true", "false"],
        default=None,
        help="DataLoader pin_memory（num_workers>0 时建议 true，加速 CPU→GPU）",
    )
    parser.add_argument("--prefetch-factor", type=int, default=None)
    parser.add_argument(
        "--video-backend",
        choices=["pyav", "torchcodec"],
        default=None,
        help="视频解码后端，torchcodec 更快（需 torchcodec 0.5 + FFmpeg）",
    )
    parser.add_argument("--use-compile", choices=["true", "false"], default=None)
    parser.add_argument("--gpu", default="0")
    args = parser.parse_args()

    if args.preset == "pot14":
        config = Path(args.config) if args.config else POT14_VLA_CONFIG
        data_name = args.data_name or "pot14"
        data_path = args.data_path or str(POT14_DATASET_PATH)
        norm_stats = args.norm_stats_file or str(
            POT14_NORM_STATS_FILE.relative_to(LINGBOT_DIR)
        )
        output_dir = args.output_dir or "output/pot14/"
    else:
        config = Path(args.config) if args.config else VLA_CONFIG
        data_name = args.data_name or "so100"
        data_path = args.data_path or str(DATASET_PATH)
        norm_stats = args.norm_stats_file or str(
            NORM_STATS_FILE.relative_to(LINGBOT_DIR)
        )
        output_dir = args.output_dir or "output/"

    env_errors = validate_training_env()
    if env_errors:
        print("=== Conda 环境检查失败 ===")
        for err in env_errors:
            print(f"  - {err}")
        sys.exit(1)

    errors = validate_local_models()
    if errors:
        print("=== 本地模型检查失败 ===")
        for err in errors:
            print(f"  - {err}")
        sys.exit(1)

    norm_path = LINGBOT_DIR / norm_stats
    norm_errors = validate_norm_stats_file(norm_path)
    if norm_errors:
        print("=== 归一化统计量检查失败 ===")
        for err in norm_errors:
            print(f"  - {err}")
        sys.exit(1)

    gpu_errors = check_gpu_free_memory()
    if gpu_errors:
        print("=== GPU 显存检查失败 ===")
        for err in gpu_errors:
            print(f"  - {err}")
        sys.exit(1)

    out_check = Path(output_dir)
    if not out_check.is_absolute():
        out_check = LINGBOT_DIR / out_check
    out_check.mkdir(parents=True, exist_ok=True)
    # 短冒烟（≤10 step）不强制预留完整 checkpoint 空间
    smoke = args.max_steps is not None and args.max_steps <= 10
    disk_errors = check_disk_free_space(
        out_check, min_free_gib=5.0 if smoke else 22.0
    )
    if disk_errors:
        print("=== 磁盘空间检查失败 ===")
        for err in disk_errors:
            print(f"  - {err}")
        sys.exit(1)

    print("=== 开始训练（离线模式） ===")
    print(f"preset:     {args.preset}")
    print(f"config:     {config}")
    print(f"data_name:  {data_name}")
    print(f"VLA 模型:   {args.model_path}")
    print(f"Tokenizer:  {args.tokenizer_path}")
    print(f"数据路径:   {data_path}")
    print(f"归一化统计: {norm_stats}")
    print(f"输出目录:   {output_dir}")
    workers_hint = args.num_workers if args.num_workers is not None else recommended_num_workers()
    print(
        f"数据加载建议: num_workers≈{workers_hint}（CPU 并行解码），"
        "pin_memory=true，video_backend=torchcodec；"
        "GPU 专用于训练，勿重复启动进程"
    )

    script_args = [
        str(LINGBOT_DIR / "tasks/vla/train_lingbotvla.py"),
        str(config),
        "--data.train_path", data_path,
        "--data.data_name", data_name,
        "--data.norm_stats_file", norm_stats,
        "--model.model_path", args.model_path,
        "--model.tokenizer_path", args.tokenizer_path,
        "--train.output_dir", output_dir,
    ]
    optional_flags: list[tuple[str, object]] = [
        ("--train.max_steps", args.max_steps),
        ("--train.save_steps", args.save_steps),
        ("--train.micro_batch_size", args.micro_batch_size),
        ("--train.gradient_accumulation_steps", args.grad_accum),
        ("--data.num_workers", args.num_workers),
        ("--data.pin_memory", args.pin_memory),
        ("--data.prefetch_factor", args.prefetch_factor),
        ("--data.video_backend", args.video_backend),
        ("--train.use_compile", args.use_compile),
    ]
    for flag, value in optional_flags:
        if value is not None:
            script_args.extend([flag, str(value)])

    run_torchrun(
        script_args,
        cwd=LINGBOT_DIR,
        env=get_subprocess_env(gpu=args.gpu),
    )

    loss_file = LINGBOT_DIR / output_dir / "checkpoints" / "loss.jsonl"
    ckpt_dir = LINGBOT_DIR / output_dir / "checkpoints"
    print("=== 训练完成 ===")
    print("产物（均以 global step 为准，不用 epoch 命名）：")
    if loss_file.exists():
        last_step = _read_last_step(loss_file)
        print(f"  loss 日志:     {loss_file}")
        print(
            f"  loss 曲线:     python scripts/plot_training_loss.py "
            f"--loss-file {loss_file.relative_to(LINGBOT_DIR.parent)}"
        )
        if last_step is not None:
            print(f"  checkpoint:    {ckpt_dir / f'global_step_{last_step}/'}")
            print(f"  部署权重:      .../global_step_{last_step}/hf_ckpt/")
    else:
        print(f"  （未找到 {loss_file}，可能训练未写入 loss）")


def _read_last_step(loss_file: Path) -> int | None:
    import json

    last_step: int | None = None
    try:
        for line in loss_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                last_step = int(json.loads(line)["step"])
    except (OSError, json.JSONDecodeError, KeyError, ValueError):
        return None
    return last_step


if __name__ == "__main__":
    main()
