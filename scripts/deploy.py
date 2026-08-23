#!/usr/bin/env python3
"""部署脚本 - 使用本地 Qwen 基座，离线部署训练好的模型"""

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from local_paths import LINGBOT_DIR, QWEN_VL_MODEL, get_subprocess_env


def main() -> None:
    parser = argparse.ArgumentParser(description="部署训练好的 Lingbot VLA 模型（离线）")
    parser.add_argument(
        "--model-path",
        default="output/checkpoints/global_step_1024/hf_ckpt",
        help="训练产出的 hf_ckpt 路径（相对于 lingbot-vla/）",
    )
    parser.add_argument(
        "--qwen-path",
        default=str(QWEN_VL_MODEL),
        help="本地 Qwen 基座路径（用于 tokenizer/processor）",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=6006,
        help="WebSocket 端口（AutoDL 自定义服务常用 6006）",
    )
    parser.add_argument(
        "--norm-path",
        default=None,
        help="归一化统计量 JSON（pot14 建议传 pot14_right_arm.json）",
    )
    args = parser.parse_args()

    env = get_subprocess_env()
    env["QWEN25_PATH"] = args.qwen_path

    print("=== 部署模型（离线模式） ===")
    print(f"微调模型: {args.model_path}")
    print(f"Qwen 基座: {args.qwen_path}")
    print(f"端口:      {args.port}")

    cmd = [
        sys.executable, "-m", "deploy.lingbot_vla_policy",
        "--model_path", args.model_path,
        "--port", str(args.port),
        "--use_compile",
        "--use_length", "16",
    ]
    if args.norm_path:
        cmd.extend(["--norm_path", args.norm_path])

    subprocess.run(cmd, cwd=LINGBOT_DIR, env=env, check=True)


if __name__ == "__main__":
    main()
