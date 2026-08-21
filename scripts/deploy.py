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
    args = parser.parse_args()

    env = get_subprocess_env()
    env["QWEN25_PATH"] = args.qwen_path

    print("=== 部署模型（离线模式） ===")
    print(f"微调模型: {args.model_path}")
    print(f"Qwen 基座: {args.qwen_path}")

    cmd = [
        sys.executable, "-m", "deploy.lingbot_vla_policy",
        "--model_path", args.model_path,
        "--use_compile",
        "--use_length", "16",
    ]

    subprocess.run(cmd, cwd=LINGBOT_DIR, env=env, check=True)


if __name__ == "__main__":
    main()
