#!/usr/bin/env python3
"""Plot training loss from loss.jsonl (x-axis: global step, not epoch)."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LossRecord:
    step: int
    loss: float
    lr: float
    epoch: int | None
    grad_norm: float | None
    step_time: float | None


def load_loss_records(loss_file: Path) -> list[LossRecord]:
    records: list[LossRecord] = []
    with loss_file.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            records.append(
                LossRecord(
                    step=int(raw["step"]),
                    loss=float(raw["loss"]),
                    lr=float(raw.get("lr", 0.0)),
                    epoch=int(raw["epoch"]) if "epoch" in raw else None,
                    grad_norm=float(raw["grad_norm"]) if "grad_norm" in raw else None,
                    step_time=float(raw["step_time"]) if "step_time" in raw else None,
                )
            )
    return records


def default_output_path(project_root: Path, last_step: int) -> Path:
    return project_root / "logs" / f"training_loss_step{last_step}.png"


def plot_loss(records: list[LossRecord], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    steps = [r.step for r in records]
    losses = [r.loss for r in records]
    last_step = steps[-1]
    last_epoch = next((r.epoch for r in reversed(records) if r.epoch is not None), None)

    title = f"Training Loss (step 1–{last_step}"
    if last_epoch is not None:
        title += f", ~{last_epoch} epoch"
    title += ")"

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, losses, color="#2E86AB", linewidth=2, label="Training Loss")
    ax.set_xlabel("Global Step")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def resolve_loss_file(project_root: Path, loss_file: Path) -> Path:
    if loss_file.is_absolute():
        return loss_file
    return project_root / loss_file


def resolve_output_path(
    project_root: Path,
    output: str | None,
    *,
    last_step: int,
    suffix_step: bool,
) -> Path:
    if output is None:
        return default_output_path(project_root, last_step)
    output_path = Path(output)
    if not output_path.is_absolute():
        output_path = project_root / output_path
    if suffix_step and "step" not in output_path.stem.lower():
        output_path = output_path.with_name(f"{output_path.stem}_step{last_step}{output_path.suffix}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot training loss from loss.jsonl (use global step on x-axis)"
    )
    parser.add_argument(
        "--loss-file",
        default="lingbot-vla/output/checkpoints/loss.jsonl",
        help="Path to loss.jsonl",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output image path (default: logs/training_loss_step{N}.png)",
    )
    parser.add_argument(
        "--no-suffix-step",
        action="store_true",
        help="Do not append _step{N} to a custom --output filename",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    loss_file = resolve_loss_file(project_root, Path(args.loss_file))

    if not loss_file.exists():
        raise FileNotFoundError(f"Loss file not found: {loss_file}")

    records = load_loss_records(loss_file)
    if not records:
        raise ValueError(f"No loss records found in {loss_file}")

    last_step = records[-1].step
    output_path = resolve_output_path(
        project_root,
        args.output,
        last_step=last_step,
        suffix_step=not args.no_suffix_step,
    )

    plot_loss(records, output_path)
    print(f"Saved loss curve to {output_path}")
    print(f"  x-axis: global step 1..{last_step}")
    if records[-1].epoch is not None:
        steps_per_epoch = max(1, math.ceil(last_step / records[-1].epoch))
        print(
            f"  note: epoch in loss.jsonl is auxiliary only "
            f"(~{steps_per_epoch} steps/epoch for this run)"
        )


if __name__ == "__main__":
    main()
