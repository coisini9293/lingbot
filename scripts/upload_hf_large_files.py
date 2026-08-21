#!/usr/bin/env python3
"""上传「新机器开训」所需大文件到 HF Dataset: coisini9293/lingbot

目标场景：新服务器下载预训练权重 + flash-attn wheel，再用自己的新数据集微调。
不上传旧 checkpoint（那是上次微调产物，换新数据训练用不到）。

用法（AutoDL / 国内环境）:
  # 1) 一次性 DNS 修复（multipart 完成回调会请求 hf-mirror.org）
  grep -q 'hf-mirror.org' /etc/hosts || echo '160.16.86.14 hf-mirror.org' | sudo tee -a /etc/hosts

  # 2) 登录并上传
  export HF_TOKEN=hf_xxx          # Write 权限
  export HF_ENDPOINT=https://hf-mirror.com
  export HF_HOME=/root/autodl-tmp/.cache/huggingface
  export HF_HUB_DISABLE_XET=1     # 避免走官方 Xet CDN
  python scripts/upload_hf_large_files.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ID = "coisini9293/lingbot"
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 只上传「新机器开训」需要的资产（不含旧 checkpoint）
UPLOAD_ITEMS: list[tuple[Path, str]] = [
    (
        PROJECT_ROOT
        / "flash_attn-2.8.3+cu12torch2.7cxx11abiFALSE-cp312-cp312-linux_x86_64.whl",
        "wheels/flash_attn-2.8.3+cu12torch2.7cxx11abiFALSE-cp312-cp312-linux_x86_64.whl",
    ),
    (
        PROJECT_ROOT
        / ".cache/huggingface/hub/models--robbyant--lingbot-vla-4b"
        / "snapshots/87712223fb463f0f6a6565f8feba64c35418e393",
        "pretrained/lingbot-vla-4b",
    ),
    (
        PROJECT_ROOT
        / ".cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct"
        / "snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3",
        "pretrained/Qwen2.5-VL-3B-Instruct",
    ),
]


def main() -> int:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print("缺少 HF_TOKEN。请先: export HF_TOKEN=hf_xxx")
        return 1

    endpoint = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com").rstrip("/")
    os.environ["HF_ENDPOINT"] = endpoint
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ["HF_HOME"] = os.environ.get(
        "HF_HOME", str(PROJECT_ROOT / ".cache" / "huggingface")
    )

    from huggingface_hub import HfApi

    api = HfApi(endpoint=endpoint, token=token)
    print(f"endpoint={endpoint} user={api.whoami()['name']}", flush=True)

    readme = PROJECT_ROOT / "docs" / "hf_dataset_card.md"
    if readme.exists():
        print(f"[上传] Dataset Card -> README.md", flush=True)
        api.upload_file(
            path_or_fileobj=str(readme),
            path_in_repo="README.md",
            repo_id=REPO_ID,
            repo_type="dataset",
        )

    for local, remote in UPLOAD_ITEMS:
        if not local.exists():
            print(f"[跳过] 不存在: {local}", flush=True)
            continue
        print(f"[上传] {local} -> {remote}", flush=True)
        if local.is_dir():
            api.upload_folder(
                folder_path=str(local),
                path_in_repo=remote,
                repo_id=REPO_ID,
                repo_type="dataset",
            )
        else:
            api.upload_file(
                path_or_fileobj=str(local),
                path_in_repo=remote,
                repo_id=REPO_ID,
                repo_type="dataset",
            )
        print(f"[完成] {remote}", flush=True)

    print(f"全部完成: https://huggingface.co/datasets/{REPO_ID}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
