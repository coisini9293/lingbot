#!/usr/bin/env bash
# 新服务器一键拉取开训资产（公开源，不依赖自己上传的 dataset）
set -euo pipefail

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/models/pretrained" "$ROOT/wheels"

echo "=== 1) 预训练权重（公开仓库，走镜像）==="
hf download robbyant/lingbot-vla-4b \
  --local-dir "$ROOT/models/pretrained/lingbot-vla-4b"
hf download Qwen/Qwen2.5-VL-3B-Instruct \
  --local-dir "$ROOT/models/pretrained/Qwen2.5-VL-3B-Instruct"

echo "=== 2) flash-attn wheel（GitHub Release）==="
WHL="flash_attn-2.8.3+cu12torch2.7cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"
URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/${WHL}"
# 若 GitHub 慢，改用加速镜像：
# URL="https://ghfast.top/${URL}"
curl -L --fail -o "$ROOT/wheels/${WHL}" "$URL"

echo "=== 3) 安装 wheel ==="
pip install "$ROOT/wheels/${WHL}"

echo "完成。代码从 GitHub 拉：https://github.com/coisini9293/lingbot"
echo "然后换新数据集 -> compute_norm -> train"
