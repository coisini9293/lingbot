#!/usr/bin/env python3
"""开环评估 - 对比预测动作与示范动作（需在 lingbot-vla/ 目录下解析 robot config）"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from local_paths import (
    DATASET_PATH,
    LINGBOT_DIR,
    NORM_STATS_FILE,
    QWEN_VL_MODEL,
    get_subprocess_env,
)


def _rel_to_lingbot(path: Path) -> str:
    """Paths passed to open_loop_eval should be relative to lingbot-vla/ when possible."""
    try:
        return str(path.relative_to(LINGBOT_DIR))
    except ValueError:
        return str(path.resolve())


def main() -> None:
    parser = argparse.ArgumentParser(description="开环评估（Open-loop eval）")
    parser.add_argument(
        "--model-path",
        default="output/checkpoints/global_step_1024/hf_ckpt",
        help="hf_ckpt 路径（相对于 lingbot-vla/ 或绝对路径）",
    )
    parser.add_argument("--robo-name", default="so100")
    parser.add_argument(
        "--norm-path",
        default=str(NORM_STATS_FILE.relative_to(LINGBOT_DIR)),
    )
    parser.add_argument("--data-path", default=str(DATASET_PATH))
    parser.add_argument("--traj-ids", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--use-length", type=int, default=50)
    parser.add_argument(
        "--save-plot-path",
        default="../logs/open_loop_eval",
        help="评估曲线输出目录（默认项目根 logs/open_loop_eval/）",
    )
    parser.add_argument("--use-compile", action="store_true")
    parser.add_argument("--gpu", default="0")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    model_path = Path(args.model_path)
    if not model_path.is_absolute():
        model_path = LINGBOT_DIR / model_path
    norm_path = Path(args.norm_path)
    if not norm_path.is_absolute():
        norm_path = LINGBOT_DIR / norm_path
    data_path = Path(args.data_path)
    if not data_path.is_absolute():
        data_path = project_root / data_path
    plot_path = Path(args.save_plot_path)
    if not plot_path.is_absolute():
        plot_path = (LINGBOT_DIR / plot_path).resolve()
    plot_path.mkdir(parents=True, exist_ok=True)

    env = get_subprocess_env(gpu=args.gpu)
    env["QWEN25_PATH"] = str(QWEN_VL_MODEL)

    cmd = [
        sys.executable,
        str(LINGBOT_DIR / "scripts/open_loop_eval.py"),
        "--model_path", _rel_to_lingbot(model_path),
        "--robo_name", args.robo_name,
        "--norm_path", _rel_to_lingbot(norm_path),
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
