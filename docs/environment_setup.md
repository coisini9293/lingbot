# LingBot-VLA 训练前置环境（AutoDL）

> 项目路径：`/root/autodl-tmp`
> 数据集：`svla_so101_pickplace`（50 episodes，11939 frames）

**安装完成后先跑自检：**

```bash
conda activate lingbotvla
cd /root/autodl-tmp
python scripts/verify_env.py
```

全部 `[OK]` 后再训练。

---

## 一、版本对应表（最容易装错）

| 组件                     | 正确版本                      | 常见错误                        | 错误现象                                        |
| ------------------------ | ----------------------------- | ------------------------------- | ----------------------------------------------- |
| PyTorch（Blackwell GPU） | **2.7.1+cu128**         | cu126                           | `sm_120 not compatible` / `no kernel image` |
| **torchcodec**     | **0.5**（配 torch 2.7） | **0.6.0**（配 torch 2.8） | `_convert_to_tensor` 刷屏 / 训练 Step 0 失败  |
| flash-attn               | 2.8.3 wheel（torch 2.7）      | 与 torch 2.8 混用               | `undefined symbol`                            |
| transformers             | 4.51.3                        | 其他版本                        | 离线模式报错                                    |
| numpy                    | 1.26.4（推荐）                | 2.x                             | 部分依赖告警                                    |
| FFmpeg                   | conda-forge**7.x**      | 未安装                          | `libavutil.so.xx not found`                   |

> **关键**：文档里曾写 `torchcodec==0.6.0` 是错的。
> 官方对照：[torchcodec 兼容表](https://github.com/meta-pytorch/torchcodec#installing-torchcodec) — **torch 2.7 → torchcodec 0.3/0.4/0.5**。

---

## 二、训练前置检查清单

| # | 条件                                | 验证                                     |
| - | ----------------------------------- | ---------------------------------------- |
| 1 | Conda`lingbotvla` Python 3.12     | `python --version`                     |
| 2 | PyTorch 2.7.1+cu128 + sm_120        | `python scripts/verify_env.py`         |
| 3 | **torchcodec 0.5** + FFmpeg 7 | 同上（含实际 mp4 解码测试）              |
| 4 | flash-attn 2.8.3                    | `python -c "import flash_attn"`        |
| 5 | transformers 4.51.3                 | `pip show transformers`                |
| 6 | lingbot-vla / lerobot 已安装        | `python -c "import lingbotvla"`        |
| 7 | 本地模型 + safetensors              | `python scripts/setup_local_models.py` |
| 8 | 归一化 JSON 含`norm_stats` 键     | `python scripts/compute_norm.py`       |

### 环境变量（`~/.bashrc`）

```bash
export HF_HOME=/root/autodl-tmp/.cache/huggingface
export PIP_CACHE_DIR=/root/autodl-tmp/.cache/pip
export QWEN25_PATH=/root/autodl-tmp/models/pretrained/Qwen2.5-VL-3B-Instruct
export OMP_NUM_THREADS=4
```

---

## 三、完整安装命令

```bash
cd /root/autodl-tmp
export HF_HOME=/root/autodl-tmp/.cache/huggingface
export PIP_CACHE_DIR=/root/autodl-tmp/.cache/pip
export OMP_NUM_THREADS=4

conda create -n lingbotvla python=3.12 -y
conda activate lingbotvla

# 1. PyTorch（Blackwell 必须 cu128；国内加阿里云镜像）
pip install --force-reinstall \
  torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
  --index-url https://download.pytorch.org/whl/cu128 \
  --extra-index-url https://mirrors.aliyun.com/pypi/simple/

# 2. FFmpeg（torchcodec 运行时依赖）
conda install -c conda-forge ffmpeg=7 -y

# 3. LeRobot + lingbot-vla
pip install "https://github.com/huggingface/lerobot/archive/refs/tags/v0.4.2.tar.gz"
cd /root/autodl-tmp/lingbot-vla
git submodule update --init --recursive
pip install -e .
pip install -e ./lingbotvla/models/vla/vision_models/lingbot-depth/ --no-deps
pip install -e ./lingbotvla/models/vla/vision_models/MoGe/
cd /root/autodl-tmp

# 4. 训练依赖
pip install -r requirements_train.txt -i https://mirrors.aliyun.com/pypi/simple/
pip install \
  "transformers==4.51.3" \
  "datasets==3.6.0" \
  "numpy==1.26.4" \
  "torchcodec==0.5" \
  "huggingface-hub>=0.30.0,<1.0" \
  -i https://mirrors.aliyun.com/pypi/simple/

# 5. flash-attn（预编译 wheel，匹配 torch 2.7）
wget -P /root/autodl-tmp \
  https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.7cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
pip install /root/autodl-tmp/flash_attn-2.8.3+cu12torch2.7cxx11abiFALSE-cp312-cp312-linux_x86_64.whl

# 6. 自检
python scripts/verify_env.py
```

---

## 四、模型与数据准备

```bash
conda activate lingbotvla
cd /root/autodl-tmp

# 模型（有网时一次）
export HF_HOME=/root/autodl-tmp/.cache/huggingface
export HF_ENDPOINT=https://hf-mirror.com
python scripts/download_models.py
python scripts/setup_local_models.py

# 归一化（必须用官方脚本，勿手写 JSON）
python scripts/compute_norm.py --gpu 0
```

---

## 五、训练

**同一时间只能跑一个训练。** 重复启动会让旧进程占满显存，GPU 空转浪费。

重跑前先释放 GPU：

```bash
pkill -f train_lingbotvla.py
nvidia-smi   # Used 应接近 0 MiB
```

### 资源利用原则（避免浪费，而非少用）

| 资源           | 应该做什么                                       | 常见浪费                                             |
| -------------- | ------------------------------------------------ | ---------------------------------------------------- |
| **GPU**  | 模型前向/反向、梯度更新                          | 重复启动训练占 79GB；DataLoader worker 误初始化 CUDA |
| **CPU**  | `num_workers` 并行解码 mp4、拼 batch           | `num_workers=0` 导致 GPU 等数据空转                |
| **内存** | `pin_memory` + 适度 `prefetch_factor=2` 预取 | `prefetch_factor` 过大（如 8）多占 RAM 收益很小    |

默认配置（`configs/vla/so100.yaml`）已按上述原则设置：`num_workers=4`、`pin_memory=true`、`video_backend=torchcodec`。Worker 进程强制只用 CPU（代码内 `worker_init_fn`），主进程独占 GPU。

```bash
conda activate lingbotvla
cd /root/autodl-tmp
python scripts/verify_env.py   # 先确认全 OK

# 冒烟 50 步（save_steps=0：只在结束时存一次，避免中途占满 50G 盘）
python scripts/train.py \
    --max-steps 50 --save-steps 0 \
    --output-dir output/smoke_test \
    --micro-batch-size 1 --grad-accum 4 \
    --use-compile false --gpu 0

# 正式训练（yaml 默认 save_steps=0，仅 1024 步结束时保存 1 个 checkpoint）
python scripts/train.py --gpu 0
```

若 CPU 核心多、仍感觉 GPU 利用率低，可适当加大 `--num-workers`（建议不超过 `CPU核心数/2`）。若内存紧张，再降 `--prefetch-factor` 或 `--num-workers`。

---

## 五（附）、Step 与 Epoch 的区别

训练日志里会同时出现 **Step** 和 **Epoch**，容易混淆。本节说明二者含义、换算关系，以及本项目**为什么一律以 Step 为准**。

### 1. 分别是什么？

| 术语 | 英文 | 含义 |
|------|------|------|
| **Step（步）** | global step / training step | 模型**更新参数 1 次**，计为 1 step |
| **Epoch（轮 / 遍）** | epoch | 把**整个训练数据集从头到尾完整看过 1 遍**，计为 1 epoch |

**类比：**

- **Epoch** = 把整本习题册**从头到尾做 1 遍**
- **Step** = 每做完**一组题并改一次学习方法**，算 1 step

习题册有 373 组题 → 做完整本 1 遍 = 1 epoch = 373 step（在本项目 batch 设置下）。

### 2. 1 个 Step 内部在做什么？

配置 `micro_batch_size: 2`、`gradient_accumulation_steps: 16` 时，**每 1 step** 会：

1. 连续取 **16 次** micro-batch（每次 2 条样本）
2. 每次做前向 + 反向，梯度**累加**
3. 16 次累加完成后，**更新 1 次**模型权重 → 这就是 **1 step**

因此 step 是「真正学进去一次」的最小计数单位，比 epoch 更细、更精确。

### 3. 本项目的换算（SO-100 数据集）

| 量 | 数值 | 计算方式 |
|----|------|---------|
| 数据集帧数 | 11,939 | `meta/info.json` |
| 有效 batch | 32 | `micro_batch_size(2) × grad_accum(16)` |
| **1 epoch** | **≈ 373 step** | ⌊11939 ÷ 32⌋ |
| **正式训练 `max_steps: 1024`** | **≈ 2.7 epoch** | 1024 ÷ 373 |
| **若跑满 20 epoch** | **≈ 7460 step** | 20 × 373 |

训练可能在第 2.7 个 epoch 中间就停在 step 1024，**epoch 往往不是整数**，不适合作为唯一刻度。

### 4. 训练由谁控制停止？

LingBot-VLA 用两个参数（至少设一个）：

| 参数 | 本项目设置 | 效果 |
|------|-----------|------|
| `max_steps` | **1024** | 跑满 1024 step 就停 ✅（当前生效） |
| `num_train_epochs` | 未设置（视为无限） | 不按轮数停止 |

因此：**你关心的是 step，不是「一共几轮」**。终端进度条在 `max_steps` 模式下也会显示 `Step: 512/1024` 这类信息。

原论文 GM-100 对比实验写的是 **20 epoch**；官方开源 yaml（如 `real_load20000h.yaml`）则常用 **max_steps: 40000**。两种写法都可以，但**同一项目里应统一用一种做主刻度**——本项目选 **step**。

### 5. 为什么 checkpoint、曲线图都用 Step？

| 产物 | 正确做法 | 不推荐 |
|------|---------|--------|
| Checkpoint 目录 | `global_step_1024/` | `epoch_3/`（可能只训到 2.7 epoch） |
| Loss 曲线 X 轴 | **Global Step** | 仅按 Epoch（丢失 step 级细节） |
| Loss 图文件名 | `training_loss_step1024.png` | `training_loss_epoch3.png` |
| 部署路径 | `.../global_step_1024/hf_ckpt/` | 与 step 不对齐会搞混版本 |

`loss.jsonl` 里虽有 `"epoch"` 字段，只是**辅助信息**；画图、存盘、部署都以 `"step"` 为准。

### 6. 常用命令（step 对齐）

```bash
# 画 loss 曲线（自动命名 training_loss_step{N}.png）
python scripts/plot_training_loss.py \
    --loss-file lingbot-vla/output/checkpoints/loss.jsonl

# 部署（step 编号与 checkpoint 一致）
python scripts/deploy.py \
    --model-path lingbot-vla/output/checkpoints/global_step_1024/hf_ckpt
```

### 7. 一句话记忆

> **Step = 学了几次（更新几次参数）；Epoch = 数据集看了几遍。**  
> 本项目用 `max_steps` 控制训练，所以**看日志、存模型、画曲线、部署，全部对齐 Step。**

更通俗的说明见 [lingbot_vla_guide.md — Step 与 Epoch](lingbot_vla_guide.md#step-与-epoch-训练进度怎么理解)。

---

## 六、常见问题（按实际报错）

| 报错                                                 | 根因                                            | 修复                                                  |
| ---------------------------------------------------- | ----------------------------------------------- | ----------------------------------------------------- |
| `libavutil.so.xx not found`                        | 未装 FFmpeg                                     | `conda install -c conda-forge ffmpeg=7 -y`          |
| **`torchcodec_ns::_convert_to_tensor`**      | **torchcodec 0.6 + torch 2.7 版本不匹配** | **`pip install torchcodec==0.5`**             |
| `Could not load libtorchcodec`                     | FFmpeg 或 torchcodec 未装                       | FFmpeg + torchcodec 0.5                               |
| `KeyError: 'norm_stats'`                           | 归一化 JSON 格式错                              | `python scripts/compute_norm.py`                    |
| `flash_attn ... undefined symbol`                  | flash-attn 与 torch 版本不匹配                  | 重装对应 wheel；勿升级 torch 到 2.8 却不换 flash-attn |
| `sm_120 not compatible`                            | PyTorch cu126                                   | 换**cu128**                                     |
| **`CUDA out of memory`**（另一进程占 ~79GB） | **重复启动训练**，旧进程未退出            | `pkill -f train_lingbotvla.py` 后只启动一次         |
| 终端停在`[torchrun]` 很久                          | 子进程在加载模型 / 旧版输出缓冲                 | `watch nvidia-smi` 看显存；勿重复启动               |
| GPU 利用率低、step 很慢                              | 数据解码跟不上（num_workers 过少）              | 加大`--num-workers`（建议 ≤ CPU/2）                |
| **`unexpected pos ... inline_container`** / Step N 保存失败 | **数据盘满**（每个 checkpoint ~21GB） | 删旧 checkpoint；正式训练用 `save_steps: 0` |
| 终端刷屏相同错误                                     | DataLoader 每个样本重复打印                     | 修根因；调试时可 `--num-workers 0` 定位              |

---

## 七、磁盘说明

| 磁盘        | 路径                 | 用途                   |
| ----------- | -------------------- | ---------------------- |
| 系统盘 30GB | `/`                | conda                  |
| 数据盘 50GB | `/root/autodl-tmp` | 代码、数据、模型、输出 |

**Checkpoint 体积（实测）：**

| 内容 | 大小 | 部署是否需要 |
|------|------|-------------|
| DCP `model/` | ~16 GB | 否（可转 hf 后删） |
| DCP `optimizer/` | ~25 GB | 否（**已默认不保存** `save_optimizer: false`） |
| `hf_ckpt/` | ~16 GB | **是** |

70G 数据盘扣除 HF 缓存（~23G）后，一次完整训练峰值约 **39G**（16+16+日志）。若仍报空间不足：

```bash
# 训练完成后，已有 hf_ckpt 时可删 DCP model 释放 16G
rm -rf lingbot-vla/output/checkpoints/global_step_*/model

# 或手动从 DCP 转 hf（训练因磁盘满未生成 hf_ckpt 时）
LOCAL_RANK=0 RANK=0 WORLD_SIZE=1 python scripts/convert_checkpoint_to_hf.py \
  --checkpoint-dir lingbot-vla/output/checkpoints/global_step_1024
```

正式训练配置（`so100.yaml`）已设置：

- `save_steps: 0` — 仅在 `max_steps=1024` 结束时保存一次  
- `save_optimizer: false` — 不存 optimizer，省 ~25GB  
- `max_checkpoints_to_keep: 1` — 自动删旧 checkpoint

```bash
pip cache purge && conda clean -a -y
df -h /root/autodl-tmp
```
