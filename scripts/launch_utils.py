"""Replace train.sh — launch distributed jobs via torchrun in pure Python."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def count_gpus(cuda_visible_devices: str | None) -> int:
    if cuda_visible_devices is not None and cuda_visible_devices.strip():
        return len([g for g in cuda_visible_devices.split(",") if g.strip()])
    result = subprocess.run(
        ["nvidia-smi", "-L"],
        capture_output=True,
        text=True,
        check=True,
    )
    return len([line for line in result.stdout.splitlines() if line.strip()])


def run_torchrun(
    script_args: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    log_file: Path | None = None,
) -> None:
    """Launch a script with torchrun (equivalent to lingbot-vla/train.sh)."""
    merged_env = {**os.environ, **(env or {})}
    merged_env.setdefault("TOKENIZERS_PARALLELISM", "false")

    nproc = count_gpus(merged_env.get("CUDA_VISIBLE_DEVICES"))
    nnodes = merged_env.get("NNODES", "1")
    node_rank = merged_env.get("NODE_RANK", "0")
    master_addr = merged_env.get("MASTER_ADDR", "0.0.0.0")
    master_port = merged_env.get("MASTER_PORT", "62500")

    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nnodes={nnodes}",
        f"--nproc-per-node={nproc}",
        f"--node-rank={node_rank}",
        f"--master-addr={master_addr}",
        f"--master-port={master_port}",
        *script_args,
    ]

    print(f"[torchrun] {nproc} GPU(s), cwd={cwd}")
    print(f"[torchrun] {' '.join(script_args)}")

    if log_file is None:
        log_file = cwd / "log.txt"

    # 实时打印子进程输出（避免看起来“卡在 torchrun 那一行”）
    log_lines: list[str] = []
    with subprocess.Popen(
        cmd,
        cwd=cwd,
        env=merged_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    ) as proc:
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            log_lines.append(line)
        returncode = proc.wait()

    log_file.write_text("".join(log_lines), encoding="utf-8")

    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, cmd, output="".join(log_lines))
