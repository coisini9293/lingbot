#!/usr/bin/env python3
"""Convert DCP checkpoint (global_step_*) to HuggingFace hf_ckpt for deploy."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lingbot-vla"))

from dataclasses import dataclass, field

from local_paths import LINGBOT_DIR, VLA_CONFIG
from lingbotvla.checkpoint import ckpt_to_state_dict
from lingbotvla.models import save_model_weights
from lingbotvla.utils.arguments import DataArguments, ModelArguments, TrainingArguments, parse_args, save_args


@dataclass
class Arguments:
    model: ModelArguments = field(default_factory=ModelArguments)
    data: DataArguments = field(default_factory=DataArguments)
    train: TrainingArguments = field(default_factory=TrainingArguments)


def main() -> None:
    parser = argparse.ArgumentParser(description="DCP → hf_ckpt（部署用）")
    parser.add_argument(
        "--checkpoint-dir",
        required=True,
        help="例如 lingbot-vla/output/checkpoints/global_step_1024",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="默认 <checkpoint-dir>/hf_ckpt",
    )
    parser.add_argument(
        "--model-assets-dir",
        default="lingbot-vla/output/model_assets",
    )
    parser.add_argument(
        "--config-yaml",
        default=str(VLA_CONFIG),
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    ckpt_dir = Path(args.checkpoint_dir)
    if not ckpt_dir.is_absolute():
        ckpt_dir = project_root / ckpt_dir
    hf_dir = Path(args.output_dir) if args.output_dir else ckpt_dir / "hf_ckpt"
    if not hf_dir.is_absolute():
        hf_dir = project_root / hf_dir
    assets_dir = Path(args.model_assets_dir)
    if not assets_dir.is_absolute():
        assets_dir = project_root / assets_dir

    if not (ckpt_dir / "model").exists():
        raise FileNotFoundError(f"Missing DCP model dir: {ckpt_dir / 'model'}")
    if not assets_dir.exists():
        raise FileNotFoundError(f"Missing model_assets: {assets_dir}")

    config_yaml = Path(args.config_yaml)
    if not config_yaml.is_absolute():
        config_yaml = project_root / config_yaml

    print("=== DCP → HF ===")
    print(f"输入: {ckpt_dir}")
    print(f"输出: {hf_dir}")

    state_dict = ckpt_to_state_dict(
        save_checkpoint_path=str(ckpt_dir),
        output_dir=str(ckpt_dir.parent.parent),
        ckpt_manager="dcp",
    )

    hf_dir.mkdir(parents=True, exist_ok=True)
    for item in assets_dir.iterdir():
        dest = hf_dir / item.name
        if item.is_dir():
            shutil.copytree(item, dest, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dest)

    save_model_weights(
        str(hf_dir),
        state_dict,
        model_assets=None,
        save_dtype=torch.float32,
    )

    sys.argv = ["", str(config_yaml)]
    cli_args = parse_args(Arguments)
    save_args(cli_args, str(hf_dir))

    print(f"=== 完成: {hf_dir} ===")
    print("部署: python scripts/deploy.py --model-path", hf_dir)


if __name__ == "__main__":
    main()
