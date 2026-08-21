"""Local model and data path constants for offline usage."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LINGBOT_DIR = PROJECT_ROOT / "lingbot-vla"
HF_CACHE = PROJECT_ROOT / ".cache/huggingface/hub"
PRETRAINED_DIR = PROJECT_ROOT / "models/pretrained"

# Pretrained models (offline, no HuggingFace Hub access needed)
LINGBOT_VLA_MODEL = PRETRAINED_DIR / "lingbot-vla-4b"
QWEN_VL_MODEL = PRETRAINED_DIR / "Qwen2.5-VL-3B-Instruct"

# Dataset
DATASET_PATH = PROJECT_ROOT / "data/raw/svla_so101_pickplace"
NORM_STATS_FILE = LINGBOT_DIR / "assets/norm_stats/so100_svla.json"

# pot14 自采数据（转换后）
POT14_DATASET_PATH = PROJECT_ROOT / "data/processed/pot14_right_arm"
POT14_NORM_STATS_FILE = LINGBOT_DIR / "assets/norm_stats/pot14_right_arm.json"
POT14_VLA_CONFIG = LINGBOT_DIR / "configs/vla/pot14.yaml"

VLA_CONFIG = LINGBOT_DIR / "configs/vla/so100.yaml"

# HuggingFace repo → local directory name
MODEL_REPOS: list[tuple[str, str]] = [
    ("robbyant/lingbot-vla-4b", "lingbot-vla-4b"),
    ("Qwen/Qwen2.5-VL-3B-Instruct", "Qwen2.5-VL-3B-Instruct"),
]

OFFLINE_ENV = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "TOKENIZERS_PARALLELISM": "false",
    # AutoDL 默认 OMP_NUM_THREADS=0 会导致 libgomp 报错
    "OMP_NUM_THREADS": "4",
}


def get_subprocess_env(*, gpu: str | None = None) -> dict[str, str]:
    env = {**os.environ, **OFFLINE_ENV}
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    return env


def find_hf_snapshot(repo_id: str) -> Path | None:
    cache_dir = HF_CACHE / f"models--{repo_id.replace('/', '--')}" / "snapshots"
    if not cache_dir.exists():
        return None
    snapshots = sorted(cache_dir.iterdir())
    return snapshots[-1] if snapshots else None


def setup_local_model_links() -> list[str]:
    """Link HF cache snapshots to models/pretrained/. Returns status messages."""
    PRETRAINED_DIR.mkdir(parents=True, exist_ok=True)
    messages: list[str] = []

    for repo_id, local_name in MODEL_REPOS:
        snapshot = find_hf_snapshot(repo_id)
        target = PRETRAINED_DIR / local_name

        if snapshot is None:
            messages.append(f"[SKIP] {local_name}: not found in {HF_CACHE}")
            continue

        if target.is_symlink() or target.exists():
            if target.is_symlink():
                target.unlink()
            elif target.is_dir():
                messages.append(f"[SKIP] {local_name}: already exists at {target}")
                continue

        target.symlink_to(snapshot)
        messages.append(f"[OK]   {local_name} -> {snapshot}")

    return messages


def validate_norm_stats_file(path: Path) -> list[str]:
    errors: list[str] = []
    if not path.exists():
        errors.append(f"Missing norm stats file: {path}")
        return errors
    import json

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        errors.append(f"Invalid JSON in norm stats file: {path} ({exc})")
        return errors
    if "norm_stats" not in data:
        errors.append(
            f"Norm stats file has wrong format (missing 'norm_stats' key): {path}. "
            "Run: python scripts/compute_norm.py"
        )
    return errors


def check_gpu_free_memory(*, min_free_gib: float = 20.0) -> list[str]:
    """Return warnings if GPU is occupied (e.g. previous train not exited)."""
    try:
        import torch

        if not torch.cuda.is_available():
            return ["CUDA 不可用"]
        free, total = torch.cuda.mem_get_info(0)
        free_gib = free / (1024**3)
        total_gib = total / (1024**3)
        used_gib = total_gib - free_gib
        if free_gib < min_free_gib:
            return [
                f"GPU 显存不足: 空闲 {free_gib:.1f}GB / 共 {total_gib:.1f}GB "
                f"(已用 {used_gib:.1f}GB)。"
                "可能有上次训练未退出。请先执行: pkill -f train_lingbotvla.py"
            ]
    except Exception as exc:
        return [f"无法检查 GPU 显存: {exc}"]
    return []


def validate_local_models() -> list[str]:
    errors: list[str] = []
    for name, path in [
        ("lingbot-vla-4b", LINGBOT_VLA_MODEL),
        ("Qwen2.5-VL-3B-Instruct", QWEN_VL_MODEL),
    ]:
        if not path.exists():
            errors.append(f"Missing local model: {path} ({name})")
        elif not (path / "config.json").exists():
            errors.append(f"Incomplete local model (no config.json): {path}")

    if LINGBOT_VLA_MODEL.exists() and not list(LINGBOT_VLA_MODEL.glob("*.safetensors")):
        errors.append(
            f"lingbot-vla-4b weights missing (*.safetensors). "
            f"Download incomplete: {LINGBOT_VLA_MODEL}"
        )
    return errors


def recommended_num_workers(*, cap: int = 8) -> int:
    """Parallel CPU workers for video decode; avoids oversubscribing cores."""
    cpus = os.cpu_count() or 4
    return max(2, min(cap, cpus // 2))


def check_disk_free_space(
    path: Path | None = None,
    *,
    min_free_gib: float = 22.0,
) -> list[str]:
    """Warn if data disk cannot fit one DCP checkpoint (~21GB)."""
    import shutil

    target = path or PROJECT_ROOT
    usage = shutil.disk_usage(target)
    free_gib = usage.free / (1024**3)
    if free_gib < min_free_gib:
        return [
            f"数据盘空间不足: {target} 空闲 {free_gib:.1f}GB，"
            f"保存 checkpoint 建议至少 {min_free_gib:.0f}GB。"
            "可删除 output/*/checkpoints 或 pip cache purge。"
        ]
    return []


EXPECTED_CONDA_ENV = "lingbotvla"


def validate_training_env() -> list[str]:
    """Ensure train.py runs inside the lingbotvla conda env."""
    import sys

    errors: list[str] = []
    conda_env = os.environ.get("CONDA_DEFAULT_ENV", "")
    if conda_env and conda_env != EXPECTED_CONDA_ENV:
        errors.append(
            f"当前 Conda 环境是 '{conda_env}'，不是 '{EXPECTED_CONDA_ENV}'。"
            f"请先执行: conda activate {EXPECTED_CONDA_ENV}"
        )
    try:
        import lingbotvla  # noqa: F401
    except ImportError:
        errors.append(
            f"当前 Python ({sys.executable}) 未安装 lingbotvla。"
            f"请先: conda activate {EXPECTED_CONDA_ENV} && cd lingbot-vla && pip install -e ."
        )
    try:
        import wandb  # noqa: F401
    except ImportError:
        errors.append(
            f"缺少 wandb。请先: conda activate {EXPECTED_CONDA_ENV} && pip install wandb"
        )
    return errors
