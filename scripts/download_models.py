#!/usr/bin/env python3
"""
下载预训练模型并链接到 models/pretrained/（仅需有网时运行一次）。
下载完成后，训练和部署均使用本地路径，无需联网。
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from local_paths import HF_CACHE, MODEL_REPOS, setup_local_model_links


def main() -> None:
    print("=== 下载预训练模型（需联网，仅运行一次） ===")
    print(f"缓存目录: {HF_CACHE}\n")

    for repo_id, local_name in MODEL_REPOS:
        print(f"下载 {repo_id} ({local_name}) ...")
        subprocess.run(["hf", "download", repo_id], check=True)

    print("\n=== 创建本地模型链接 ===")
    for msg in setup_local_model_links():
        print(msg)

    print("\n=== 完成 ===")
    print("本地路径: models/pretrained/<model_name>")
    print("之后运行 python scripts/train.py 即可，无需联网。")


if __name__ == "__main__":
    main()
