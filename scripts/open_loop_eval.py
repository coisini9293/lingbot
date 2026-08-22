#!/usr/bin/env python3
"""开环评估 - 对比预测动作与示范动作（需在 lingbot-vla/ 目录下解析 robot config）"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from local_paths import (
    LINGBOT_DIR,
    POT14_DATASET_PATH,
    POT14_NORM_STATS_FILE,
    QWEN_VL_MODEL,
    get_subprocess_env,
)


def _resolve_under_lingbot(raw: str | Path, project_root: Path) -> Path:
    """Resolve model/norm paths from project root or lingbot-vla/."""
    path = Path(raw)
    if path.is_absolute():
        return path.resolve()
    # Prefer existing path relative to project root (e.g. lingbot-vla/output/...)
    from_root = (project_root / path).resolve()
    if from_root.exists():
        return from_root
    # Then relative to lingbot-vla/ (e.g. output/pot14/...)
    from_lingbot = (LINGBOT_DIR / path).resolve()
    if from_lingbot.exists():
        return from_lingbot
    # Fall back to lingbot-vla relative (caller may still want a clear error)
    return from_lingbot


def main() -> None:
    parser = argparse.ArgumentParser(description="开环评估（Open-loop eval）")
    parser.add_argument(
        "--model-path",
        default="output/pot14/checkpoints/global_step_1024/hf_ckpt",
        help="hf_ckpt 路径（绝对路径，或相对项目根/lingbot-vla/）",
    )
    parser.add_argument("--robo-name", default="pot14")
    parser.add_argument(
        "--norm-path",
        default=str(POT14_NORM_STATS_FILE),
    )
    parser.add_argument("--data-path", default=str(POT14_DATASET_PATH))
    parser.add_argument("--traj-ids", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--use-length", type=int, default=50)
    parser.add_argument(
        "--save-plot-path",
        default="../logs/open_loop_eval_pot14",
        help="评估曲线输出目录（默认项目根 logs/open_loop_eval_pot14/）",
    )
    parser.add_argument("--use-compile", action="store_true")
    parser.add_argument("--gpu", default="0")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    model_path = _resolve_under_lingbot(args.model_path, project_root)
    norm_path = _resolve_under_lingbot(args.norm_path, project_root)
    data_path = Path(args.data_path)
    if not data_path.is_absolute():
        data_path = (project_root / data_path).resolve()
    else:
        data_path = data_path.resolve()
    plot_path = Path(args.save_plot_path)
    if not plot_path.is_absolute():
        plot_path = (LINGBOT_DIR / plot_path).resolve()
    else:
        plot_path = plot_path.resolve()
    plot_path.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        raise FileNotFoundError(f"模型路径不存在: {model_path}")
    if not norm_path.exists():
        raise FileNotFoundError(f"归一化文件不存在: {norm_path}")

    env = get_subprocess_env(gpu=args.gpu)
    env["QWEN25_PATH"] = str(QWEN_VL_MODEL)

    # 必须传绝对路径：相对路径会被 HuggingFace 当成 repo id
    cmd = [
        sys.executable,
        str(LINGBOT_DIR / "scripts/open_loop_eval.py"),
        "--model_path", str(model_path),
        "--robo_name", args.robo_name,
        "--norm_path", str(norm_path),
        "--data_path", str(data_path),
        "--traj_ids", *[str(t) for t in args.traj_ids],
        "--use_length", str(args.use_length),
        "--save_plot_path", str(plot_path),
    ]
    if args.use_compile:
        cmd.append("--use_compile")

    print("=== 开环评估 ===")
    print(f"模型:     {model_path}")
    print(f"数据:     {data_path}")
    print(f"轨迹 ID:  {args.traj_ids}")
    print(f"输出图:   {plot_path}/")

    subprocess.run(cmd, cwd=LINGBOT_DIR, env=env, check=True)
    print("=== 评估完成 ===")


if __name__ == "__main__":
    main()
