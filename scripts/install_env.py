#!/usr/bin/env python3
"""安装 lingbot-vla 训练环境（纯 Python，替代 lingbot-vla/install.sh）"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from local_paths import LINGBOT_DIR, PROJECT_ROOT


def run(cmd: list[str], *, cwd: Path | None = None) -> None:
    print(f"\n>>> {' '.join(cmd)}")
    subprocess.run(cmd, cwd=cwd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="安装 LingBot-VLA 训练依赖")
    parser.add_argument(
        "--skip-flash-attn",
        action="store_true",
        help="跳过 flash-attn 安装（显存优化可选）",
    )
    parser.add_argument(
        "--flash-attn-wheel",
        default=str(
            PROJECT_ROOT
            / "flash_attn-2.8.3+cu12torch2.7cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"
        ),
        help="flash-attn 预编译 wheel 路径",
    )
    args = parser.parse_args()

    print("=== 安装 LeRobot ===")
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "https://github.com/huggingface/lerobot/archive/refs/tags/v0.4.2.tar.gz",
        ]
    )

    print("\n=== 初始化 git submodule ===")
    run(["git", "submodule", "update", "--init", "--recursive"], cwd=LINGBOT_DIR)

    print("\n=== 安装 lingbot-vla ===")
    run([sys.executable, "-m", "pip", "install", "-e", "."], cwd=LINGBOT_DIR)
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-e",
            "./lingbotvla/models/vla/vision_models/lingbot-depth/",
            "--no-deps",
        ],
        cwd=LINGBOT_DIR,
    )
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-e",
            "./lingbotvla/models/vla/vision_models/MoGe/",
        ],
        cwd=LINGBOT_DIR,
    )

    if not args.skip_flash_attn:
        wheel = Path(args.flash_attn_wheel)
        if wheel.exists():
            print("\n=== 安装 flash-attn（预编译 wheel）===")
            run([sys.executable, "-m", "pip", "install", str(wheel)])
        else:
            print(f"\n[WARN] flash-attn wheel 不存在: {wheel}")
            print("请下载 wheel 后重试，或使用 --skip-flash-attn 跳过")

    print("\n=== 安装完成 ===")
    print("验证: python -c \"import lingbotvla; print('OK')\"")


if __name__ == "__main__":
    main()
