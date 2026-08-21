#!/usr/bin/env python3
"""Link cached HF models to models/pretrained/ for offline usage."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from local_paths import LINGBOT_VLA_MODEL, QWEN_VL_MODEL, setup_local_model_links


def main() -> None:
    print("=== Setup local model links ===")
    for msg in setup_local_model_links():
        print(msg)

    print("\n=== Verify ===")
    for name, path in [
        ("lingbot-vla-4b", LINGBOT_VLA_MODEL),
        ("Qwen2.5-VL-3B-Instruct", QWEN_VL_MODEL),
    ]:
        if (path / "config.json").exists():
            weights = list(path.glob("*.safetensors"))
            status = f"config.json OK, {len(weights)} weight file(s)"
            print(f"[OK]   {name}: {status}")
        else:
            print(f"[WARN] {name}: config.json missing — download may be incomplete")


if __name__ == "__main__":
    main()
