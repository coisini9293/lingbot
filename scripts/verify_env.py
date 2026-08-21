#!/usr/bin/env python3
"""训练前环境自检：逐项检查，失败时给出修复建议。"""

from __future__ import annotations

import glob
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from local_paths import (
    DATASET_PATH,
    LINGBOT_DIR,
    LINGBOT_VLA_MODEL,
    NORM_STATS_FILE,
    QWEN_VL_MODEL,
)

# torch 2.7.x 必须配 torchcodec 0.5（0.6 对应 torch 2.8，会报 _convert_to_tensor）
TORCH_TORCHCODEC_PAIRS: dict[str, str] = {
    "2.7": "0.5",
    "2.8": "0.6",
    "2.9": "0.8",
}


def check(name: str, ok: bool, detail: str = "", fix: str = "") -> bool:
    mark = "OK" if ok else "FAIL"
    print(f"[{mark}] {name}" + (f" — {detail}" if detail else ""))
    if not ok and fix:
        print(f"      修复: {fix}")
    return ok


def main() -> None:
    all_ok = True

    print("=== LingBot-VLA 训练环境自检 ===\n")

    conda_env = __import__("os").environ.get("CONDA_DEFAULT_ENV", "(未激活)")
    all_ok &= check(
        "Conda 环境 lingbotvla",
        conda_env == "lingbotvla",
        conda_env,
        "conda activate lingbotvla",
    )

    # Python
    v = sys.version_info
    all_ok &= check(
        "Python 3.12",
        v.major == 3 and v.minor == 12,
        f"{v.major}.{v.minor}.{v.micro}",
        "conda create -n lingbotvla python=3.12 -y",
    )

    # PyTorch
    try:
        import torch

        torch_ver = torch.__version__
        cuda_ver = torch.version.cuda or "N/A"
        arch = torch.cuda.get_arch_list()
        has_sm120 = "sm_120" in arch
        has_sm90 = "sm_90" in arch
        gpu_ok = torch.cuda.is_available()
        if gpu_ok:
            t = torch.randn(2, 2, device="cuda")
            gpu_ok = t.device.type == "cuda"
        all_ok &= check(
            "PyTorch + CUDA",
            gpu_ok and (has_sm120 or has_sm90),
            f"{torch_ver}, cuda={cuda_ver}, arch={arch[-3:]}",
            "Blackwell 用 cu128: pip install torch==2.7.1 ... --index-url https://download.pytorch.org/whl/cu128",
        )
    except Exception as exc:
        all_ok &= check("PyTorch + CUDA", False, str(exc))

    # torchcodec 版本匹配
    try:
        import torchcodec

        tc_ver = torchcodec.__version__
        torch_major_minor = ".".join(torch.__version__.split("+")[0].split(".")[:2])
        expected = TORCH_TORCHCODEC_PAIRS.get(torch_major_minor, "?")
        tc_match = tc_ver.split(".")[0] == expected.split(".")[0] and (
            tc_ver.startswith(expected) or expected.startswith(tc_ver.split(".")[0])
        )
        # 精确：2.7 -> 0.5
        if torch_major_minor == "2.7":
            tc_match = tc_ver.startswith("0.5") or tc_ver.startswith("0.4") or tc_ver.startswith("0.3")
        all_ok &= check(
            "torchcodec 版本",
            tc_match,
            f"torchcodec={tc_ver}, torch={torch_major_minor} → 需要 torchcodec {expected}",
            f'pip install "torchcodec=={expected}" -i https://mirrors.aliyun.com/pypi/simple/',
        )
    except ImportError:
        all_ok &= check(
            "torchcodec",
            False,
            "未安装",
            'pip install "torchcodec==0.5" -i https://mirrors.aliyun.com/pypi/simple/',
        )

    # FFmpeg
    try:
        r = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, check=False)
        ff_ok = r.returncode == 0
        ff_line = r.stdout.splitlines()[0] if r.stdout else "not found"
        all_ok &= check(
            "FFmpeg",
            ff_ok,
            ff_line[:60] if ff_ok else "",
            "conda install -c conda-forge ffmpeg=7 -y",
        )
    except FileNotFoundError:
        all_ok &= check("FFmpeg", False, "command not found", "conda install -c conda-forge ffmpeg=7 -y")

    # torchcodec 实际解码
    try:
        from torchcodec.decoders import VideoDecoder

        videos = glob.glob(str(DATASET_PATH / "videos/**/*.mp4"), recursive=True)
        if videos:
            dec = VideoDecoder(videos[0], device="cpu")
            _ = dec[0]
            all_ok &= check("torchcodec 视频解码", True, videos[0][-40:])
        else:
            all_ok &= check("torchcodec 视频解码", False, "找不到 mp4", f"检查数据集 {DATASET_PATH}")
    except Exception as exc:
        all_ok &= check(
            "torchcodec 视频解码",
            False,
            str(exc)[:120],
            "torch 2.7 请用 torchcodec==0.5，并安装 FFmpeg",
        )

    # flash-attn
    try:
        import flash_attn  # noqa: F401

        all_ok &= check("flash-attn", True)
    except Exception as exc:
        all_ok &= check(
            "flash-attn",
            False,
            str(exc)[:100],
            "pip install flash_attn-2.8.3+cu12torch2.7...whl（须与当前 torch 版本匹配）",
        )

    # 核心包
    try:
        import transformers
        import lerobot
        import lingbotvla

        all_ok &= check(
            "transformers / lerobot / lingbotvla",
            transformers.__version__ == "4.51.3",
            f"transformers={transformers.__version__}",
            'pip install "transformers==4.51.3"',
        )
    except Exception as exc:
        all_ok &= check("核心 Python 包", False, str(exc))

    # 模型
    for name, path in [("lingbot-vla-4b", LINGBOT_VLA_MODEL), ("Qwen2.5-VL", QWEN_VL_MODEL)]:
        ok = path.exists() and (path / "config.json").exists()
        weights = list(path.glob("*.safetensors")) if path.exists() else []
        if name == "lingbot-vla-4b":
            ok = ok and bool(weights)
        all_ok &= check(
            f"本地模型 {name}",
            ok,
            f"{len(weights)} weight file(s)" if weights else str(path),
            "python scripts/download_models.py && python scripts/setup_local_models.py",
        )

    # 数据集
    all_ok &= check(
        "数据集",
        DATASET_PATH.exists(),
        str(DATASET_PATH),
    )

    # 归一化
    if NORM_STATS_FILE.exists():
        data = json.loads(NORM_STATS_FILE.read_text(encoding="utf-8"))
        ns_ok = "norm_stats" in data
        all_ok &= check(
            "归一化统计量格式",
            ns_ok,
            str(NORM_STATS_FILE.relative_to(LINGBOT_DIR)),
            "python scripts/compute_norm.py",
        )
    else:
        all_ok &= check("归一化统计量", False, "文件不存在", "python scripts/compute_norm.py")

    print()
    if all_ok:
        print("=== 全部通过，可以运行 python scripts/train.py ===")
        sys.exit(0)
    else:
        print("=== 存在失败项，请先修复再训练。详见 docs/environment_setup.md ===")
        sys.exit(1)


if __name__ == "__main__":
    main()
