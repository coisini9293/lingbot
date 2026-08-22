#!/usr/bin/env python3
"""把 pot14 微调权重与配置上传到 Hugging Face（不复制大文件，避免占满磁盘）。

默认仓库: https://huggingface.co/coisini9293/lingbot_pot14

用法:
  conda activate lingbotvla
  cd /root/autodl-tmp
  huggingface-cli login
  python scripts/upload_pot14_to_hf.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from huggingface_hub import HfApi, login

sys.path.insert(0, str(Path(__file__).resolve().parent))
from local_paths import LINGBOT_DIR, PROJECT_ROOT

DEFAULT_REPO = "coisini9293/lingbot_pot14"
DEFAULT_CKPT = (
    LINGBOT_DIR / "output/pot14/checkpoints/global_step_1024/hf_ckpt"
)
DEFAULT_NORM = LINGBOT_DIR / "assets/norm_stats/pot14_right_arm.json"
DEFAULT_ROBOT = LINGBOT_DIR / "configs/robot_configs/pot14.yaml"
DEFAULT_VLA = LINGBOT_DIR / "configs/vla/pot14.yaml"
DEFAULT_CARD = PROJECT_ROOT / "docs/hf_model_card_pot14.md"


def _ensure_login(token: str | None) -> None:
    if token:
        login(token=token)
        return
    try:
        HfApi().whoami()
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            "未登录 Hugging Face。请先执行:\n"
            "  huggingface-cli login\n"
            "或:\n"
            "  export HF_TOKEN=hf_xxx\n"
            f"原始错误: {exc}"
        ) from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="上传 pot14 权重到 Hugging Face")
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    parser.add_argument("--ckpt-dir", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--norm-file", type=Path, default=DEFAULT_NORM)
    parser.add_argument("--robot-config", type=Path, default=DEFAULT_ROBOT)
    parser.add_argument("--vla-config", type=Path, default=DEFAULT_VLA)
    parser.add_argument("--model-card", type=Path, default=DEFAULT_CARD)
    parser.add_argument("--token", default=None, help="HF token；默认读缓存或 HF_TOKEN")
    parser.add_argument(
        "--private",
        action="store_true",
        help="若仓库不存在则创建为私有（已有仓库不改可见性）",
    )
    parser.add_argument(
        "--commit-message",
        default="Upload pot14 step1024 hf_ckpt + configs",
    )
    args = parser.parse_args()

    for p, name in [
        (args.ckpt_dir, "hf_ckpt"),
        (args.norm_file, "norm"),
        (args.robot_config, "robot yaml"),
        (args.vla_config, "vla yaml"),
    ]:
        if not p.exists():
            raise FileNotFoundError(f"缺少 {name}: {p}")

    _ensure_login(args.token)
    api = HfApi()

    try:
        api.repo_info(repo_id=args.repo_id, repo_type="model")
        print(f"仓库已存在: https://huggingface.co/{args.repo_id}")
    except Exception:
        print(f"创建仓库: {args.repo_id}")
        api.create_repo(
            repo_id=args.repo_id,
            repo_type="model",
            private=args.private,
            exist_ok=True,
        )

    # 1) 直接上传 hf_ckpt（不复制到 /tmp）
    print(f"上传权重目录: {args.ckpt_dir} （约 16GB，请耐心等待）")
    api.upload_folder(
        folder_path=str(args.ckpt_dir),
        repo_id=args.repo_id,
        repo_type="model",
        commit_message=f"{args.commit_message} (weights)",
    )

    # 2) 小文件单独上传
    extras: list[tuple[Path, str]] = [
        (args.norm_file, "configs/pot14_right_arm.json"),
        (args.robot_config, "configs/robot_pot14.yaml"),
        (args.vla_config, "configs/vla_pot14.yaml"),
    ]
    if args.model_card.exists():
        extras.append((args.model_card, "README.md"))

    print("上传配置与 README ...")
    for local, remote in extras:
        api.upload_file(
            path_or_fileobj=str(local),
            path_in_repo=remote,
            repo_id=args.repo_id,
            repo_type="model",
            commit_message=f"Add {remote}",
        )
        print(f"  OK {remote}")

    print("=== 上传完成 ===")
    print(f"页面: https://huggingface.co/{args.repo_id}")
    print("下载示例:")
    print(
        f"  huggingface-cli download {args.repo_id} "
        f"--local-dir models/finetuned/lingbot_pot14"
    )


if __name__ == "__main__":
    main()
