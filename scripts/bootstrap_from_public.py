#!/usr/bin/env python3
"""新服务器：下载开训所需资产（不必从 coisini9293/lingbot 再传一遍大文件）

预训练权重本身就在 Hugging Face 公开仓库；flash-attn 有官方预编译 wheel。
国内用 hf-mirror 下载通常比「自己上传再下载」快一个数量级。

用法:
  export HF_ENDPOINT=https://hf-mirror.com
  python scripts/bootstrap_from_public.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models" / "pretrained"
WHEELS_DIR = PROJECT_ROOT / "wheels"

# 公开源（不要重复上传到自己的 dataset）
PUBLIC_MODELS = [
    ("robbyant/lingbot-vla-4b", "lingbot-vla-4b"),
    ("Qwen/Qwen2.5-VL-3B-Instruct", "Qwen2.5-VL-3B-Instruct"),
]

# 与本机 CUDA/Torch 匹配时再改；当前环境是 cu12 + torch2.7/2.8 附近
FLASH_ATTN_WHL_URL = (
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/"
    "flash_attn-2.8.3+cu12torch2.7cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"
)


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> int:
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    WHEELS_DIR.mkdir(parents=True, exist_ok=True)

    # 1) 预训练权重：直接拉公开仓库
    for repo_id, local_name in PUBLIC_MODELS:
        dest = MODELS_DIR / local_name
        if dest.exists() and any(dest.iterdir()):
            print(f"[跳过] 已存在 {dest}", flush=True)
            continue
        print(f"[下载] {repo_id} -> {dest}", flush=True)
        run(
            [
                sys.executable,
                "-m",
                "huggingface_hub.commands.huggingface_cli",
                "download",
                repo_id,
                "--local-dir",
                str(dest),
            ]
        )

    # 2) flash-attn wheel：从官方 Release 拉（可用 ghproxy / 浏览器镜像加速）
    whl_name = Path(FLASH_ATTN_WHL_URL).name
    whl_path = WHEELS_DIR / whl_name
    if whl_path.exists():
        print(f"[跳过] 已存在 {whl_path}", flush=True)
    else:
        print(f"[下载] {FLASH_ATTN_WHL_URL}", flush=True)
        # 优先直连 GitHub；失败可手动换镜像，例如:
        # https://ghfast.top/https://github.com/Dao-AILab/flash-attention/releases/download/...
        run(["curl", "-L", "--fail", "-o", str(whl_path), FLASH_ATTN_WHL_URL])

    print(
        f"""
完成。接下来:
  pip install {whl_path}
  # 把配置里的 model_path / tokenizer_path 指到:
  #   {MODELS_DIR / 'lingbot-vla-4b'}
  #   {MODELS_DIR / 'Qwen2.5-VL-3B-Instruct'}
  # 准备新数据集后:
  #   python scripts/compute_norm.py
  #   python scripts/train.py
"""
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
