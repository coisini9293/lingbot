# LingBot-VLA SO-100 完整流程总结

> 从数据获取 → 数据处理 → 训练 → 结果分析 → 评估 → 部署  
> 项目路径：`/root/autodl-tmp`  
> 实测数据集：`svla_so101_pickplace`（50 episodes，11,939 frames）

---

## 一、流程总览

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        LingBot-VLA SO-100 全流程                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  ① 环境搭建          ② 数据获取          ③ 数据处理                        │
│  install_env.py       真机采集 / HF 下载    compute_norm.py                 │
│  verify_env.py        collect_data.py     robot_configs/so100.yaml        │
│       ↓                    ↓                     ↓                        │
│  ④ 模型准备          ⑤ 微调训练          ⑥ 结果分析                        │
│  download_models.py   train.py             plot_training_loss.py        │
│  setup_local_models   so100.yaml             loss.jsonl                   │
│       ↓                    ↓                     ↓                        │
│  ⑦ 开环评估          ⑧ 真机部署          ⑨（可选）Checkpoint 转换          │
│  open_loop_eval.py    deploy.py            convert_checkpoint_to_hf.py    │
│                                                                           │
└─────────────────────────────────────────────────────────────────────────┘
```

**本项目实际路径**：没有真机，使用 HuggingFace 开源数据集 `svla_so101_pickplace` 在 AutoDL 上完成 **1024 step 微调**，并完成开环评估。

---

## 二、环境搭建

### 2.1 硬件与磁盘

| 资源 | 配置 | 说明 |
|------|------|------|
| GPU | NVIDIA Blackwell（sm_120） | 必须 PyTorch **cu128** |
| 显存 | 单卡 ~80GB 可用 | 本项目单卡训练 |
| 数据盘 | `/root/autodl-tmp`（70GB） | 代码、数据、模型、输出 |
| 系统盘 | `/`（30GB） | Conda 环境 |

### 2.2 关键软件版本

| 组件 | 版本 | 备注 |
|------|------|------|
| Python | 3.12 | Conda 环境 `lingbotvla` |
| PyTorch | 2.7.1+cu128 | Blackwell 必须 cu128 |
| torchcodec | **0.5** | 配 torch 2.7，勿用 0.6 |
| flash-attn | 2.8.3 | 预编译 wheel |
| transformers | 4.51.3 | 离线推理 |
| FFmpeg | 7.x | torchcodec 依赖 |

详细安装命令见 [environment_setup.md](environment_setup.md)。

### 2.3 环境变量

```bash
export HF_HOME=/root/autodl-tmp/.cache/huggingface
export PIP_CACHE_DIR=/root/autodl-tmp/.cache/pip
export QWEN25_PATH=/root/autodl-tmp/models/pretrained/Qwen2.5-VL-3B-Instruct
export OMP_NUM_THREADS=4
```

### 2.4 自检（训练前必跑）

```bash
conda activate lingbotvla
cd /root/autodl-tmp
python scripts/verify_env.py
```

全部 `[OK]` 后再继续。脚本会检查：Conda 环境、PyTorch/CUDA、torchcodec 解码、flash-attn、本地模型、归一化 JSON、GPU 显存、磁盘空间。

---

## 三、数据获取

### 3.1 两条路径

| 路径 | 适用场景 | 命令/来源 |
|------|---------|----------|
| **A. 开源数据集（本项目）** | 无机械臂，云端练手 | HuggingFace `svla_so101_pickplace` |
| **B. 真机采集** | 有自己的 SO-100 臂 | `scripts/collect_data.py` |

### 3.2 路径 A：下载开源数据集

本项目数据已放在：

```
/root/autodl-tmp/data/raw/svla_so101_pickplace/
├── meta/
│   ├── info.json          # 数据集元信息
│   └── stats.json         # LeRobot 统计
├── data/                  # parquet 动作/状态
├── videos/                # mp4 摄像头录像
└── README.md
```

**数据集规模（`meta/info.json`）：**

| 字段 | 值 |
|------|-----|
| 机器人类型 | `so100_follower` |
| Episode 数 | 50 |
| 总帧数 | 11,939 |
| FPS | 30 |
| 任务数 | 1 |
| 动作维度 | 6（5 关节 + 1 夹爪） |
| 摄像头 | 2 路：`observation.images.up`、`observation.images.side` |
| 分辨率 | 640×480，AV1 编码 |

若需重新下载（有网时）：

```bash
export HF_HOME=/root/autodl-tmp/.cache/huggingface
export HF_ENDPOINT=https://hf-mirror.com   # 国内镜像
huggingface-cli download <dataset_repo> --local-dir data/raw/svla_so101_pickplace
```

### 3.3 路径 B：真机遥操作采集

需要：SO-100 leader/follower 双臂、USB 摄像头、串口连接。

```bash
python scripts/collect_data.py \
  --task "将绿色方块放到纸盒里" \
  --num-episodes 5 \
  --repo-id hexchip/record-test
```

采集内容（LeRobot 格式）：
- **视觉**：多路摄像头 mp4
- **动作**：leader 操控的 6 维关节指令
- **状态**：follower 当前关节位置
- **语言**：任务描述文本

建议每个任务 **5–20 个 episode**，动作规范、覆盖不同初始位置。

---

## 四、数据处理

训练前需完成两件事：**机器人配置映射** 和 **归一化统计量计算**。

### 4.1 机器人配置（摄像头/关节映射）

文件：`lingbot-vla/configs/robot_configs/so100.yaml`

本数据集只有 2 个摄像头，配置做了映射：

| 训练用名称 | 数据集原始键 |
|-----------|-------------|
| `camera_top` | `observation.images.up` |
| `camera_wrist_left` | `observation.images.side` |

动作分为：
- `action.arm.position`：5 维关节（**相对当前 state 的增量**，`subtract_state: True`）
- `action.effector.position`：1 维夹爪（绝对值）

### 4.2 归一化统计量

VLA 训练要求动作/状态在统一数值范围。必须运行官方脚本生成 JSON，**不可手写**。

```bash
conda activate lingbotvla
cd /root/autodl-tmp
python scripts/compute_norm.py --gpu 0
```

输出：`lingbot-vla/assets/norm_stats/so100_svla.json`

文件必须包含顶层键 `norm_stats`，内含各模态的 `mean` / `std`（或 min/max，取决于 `norm_type`）。训练配置 `so100.yaml` 中：

```yaml
data:
  norm_type: meanstd
  norm_stats_file: assets/norm_stats/so100_svla.json
```

### 4.3 训练配置

主配置：`lingbot-vla/configs/vla/so100.yaml`

| 参数 | 值 | 含义 |
|------|-----|------|
| `micro_batch_size` | 2 | 每次前向样本数 |
| `gradient_accumulation_steps` | 16 | 梯度累加次数 |
| **有效 batch** | **32** | 2 × 16 |
| `max_steps` | 1024 | 训练停止条件 |
| `lr` | 5e-5 | 学习率 |
| `num_workers` | 4 | CPU 并行解码视频 |
| `video_backend` | torchcodec | 视频解码后端 |
| `save_steps` | 0 | 仅在结束时存 checkpoint |
| `save_optimizer` | false | 不存 optimizer（省 ~25GB） |

**Step 与 Epoch 换算：**

```
1 epoch ≈ ⌊11939 ÷ 32⌋ = 373 step
1024 step ≈ 2.7 epoch
```

本项目**一律以 global step 为准**（checkpoint 名、loss 图、部署路径）。详见 [environment_setup.md — Step 与 Epoch](environment_setup.md#五附step-与-epoch-的区别)。

---

## 五、模型准备

### 5.1 预训练权重

| 模型 | 用途 | 本地路径 |
|------|------|---------|
| `robbyant/lingbot-vla-4b` | VLA 基座（4B 参数） | `models/pretrained/lingbot-vla-4b` |
| `Qwen/Qwen2.5-VL-3B-Instruct` | 视觉-语言 tokenizer/processor | `models/pretrained/Qwen2.5-VL-3B-Instruct` |

### 5.2 下载与离线链接

```bash
cd /root/autodl-tmp
export HF_HOME=/root/autodl-tmp/.cache/huggingface
python scripts/download_models.py      # 有网，仅一次
python scripts/setup_local_models.py   # 链接 HF cache → models/pretrained/
```

之后训练/评估/部署均走本地路径，设置 `HF_HUB_OFFLINE=1` 离线运行。

---

## 六、训练

### 6.1 冒烟测试（可选，50 step）

```bash
conda activate lingbotvla
cd /root/autodl-tmp
python scripts/verify_env.py

python scripts/train.py \
  --max-steps 50 --save-steps 0 \
  --output-dir output/smoke_test \
  --micro-batch-size 1 --grad-accum 4 \
  --use-compile false --gpu 0
```

### 6.2 正式训练（1024 step）

```bash
python scripts/train.py --gpu 0
```

等价于读取 `configs/vla/so100.yaml` 默认参数。训练入口在 `lingbot-vla/tasks/vla/train_lingbotvla.py`，由 `scripts/train.py` 通过 torchrun 启动。

**注意事项：**
- 同一时间只跑一个训练进程
- 重跑前先：`pkill -f train_lingbotvla.py`
- 第一步较慢（~54s，含 compile 预热），稳定后约 **4.6 s/step**
- 1024 step 总时长约 **1.3 小时**（单卡实测）

### 6.3 训练产物

| 产物 | 路径 | 大小 |
|------|------|------|
| Loss 日志 | `lingbot-vla/output/checkpoints/loss.jsonl` | ~200KB |
| DCP Checkpoint | `lingbot-vla/output/checkpoints/global_step_1024/model/` | ~16GB |
| HF 部署权重 | `.../global_step_1024/hf_ckpt/` | ~16GB |
| 训练配置快照 | `lingbot-vla/output/lingbotvla_cli.yaml` | 小 |

若训练结束时磁盘满、未自动生成 `hf_ckpt`：

```bash
python scripts/convert_checkpoint_to_hf.py \
  --checkpoint-dir lingbot-vla/output/checkpoints/global_step_1024
```

已有 `hf_ckpt` 后可删 DCP `model/` 释放 16GB：

```bash
rm -rf lingbot-vla/output/checkpoints/global_step_1024/model
```

---

## 七、结果分析

### 7.1 Loss 曲线

```bash
python scripts/plot_training_loss.py \
  --loss-file lingbot-vla/output/checkpoints/loss.jsonl
```

默认输出：`logs/training_loss_step1024.png`（X 轴为 **Global Step**）。

### 7.2 本项目实测训练指标

| 指标 | Step 1 | Step 1024 | 说明 |
|------|--------|-----------|------|
| Loss | 0.856 | 0.085 | 训练损失（非 accuracy） |
| Grad Norm | 3.60 | 0.22 | 梯度范数，收敛正常 |
| LR | 5e-5 | 5e-5 | constant 策略 |
| Step Time | 53.6s | 4.5s | 首步含 compile 预热 |
| 总 Step 数 | — | 1024 | loss.jsonl 共 1024 行 |

**Loss 下降趋势：** 0.86 → 0.08 量级，说明模型在示范数据上拟合良好。  
**注意：** 训练 Loss 低 ≠ 真机一定成功；论文指标为 **SR（成功率）** 和 **PS（进度分数）**，需闭环/真机验证。

### 7.3 如何解读 Loss

- Loss 是模型预测动作与示范动作在**归一化空间**的差异（类似回归 MSE）
- 没有「准确率」概念；看曲线是否平稳下降、是否过拟合
- 对比不同 checkpoint 时，用相同 step 编号（如 `global_step_1024`）

---

## 八、评估（开环 Open-Loop Eval）

开环评估：用数据集中的历史画面作为输入，对比 **模型预测动作** 与 **示范真值动作**，机械臂不实际运动。

### 8.1 运行命令

**必须从项目根目录**使用包装脚本（自动切换工作目录到 `lingbot-vla/`）：

```bash
conda activate lingbotvla
cd /root/autodl-tmp

python scripts/open_loop_eval.py \
  --model-path lingbot-vla/output/checkpoints/global_step_1024/hf_ckpt \
  --traj-ids 0 1 2 \
  --use-length 50
```

**不要**直接在 `/root/autodl-tmp` 下运行 `lingbot-vla/scripts/open_loop_eval.py`，否则会报：

```
FileNotFoundError: configs/robot_configs/so100.yaml
```

### 8.2 评估指标

| 指标 | 含义 |
|------|------|
| **MSE** | 预测与真值动作的均方误差（反归一化后） |
| **MAE** | 平均绝对误差 |
| **曲线图** | 各关节维度 gt vs pred 时序对比 |

### 8.3 本项目实测结果

| 轨迹 ID | MSE | MAE | 曲线图 |
|---------|-----|-----|--------|
| 0 | 14.31 | 2.17 | `logs/open_loop_eval/0.png` |
| 1 | 150.54 | 5.59 | `logs/open_loop_eval/1.png` |
| 平均（0+1） | 82.42 | 3.88 | — |

推理速度：约 **0.47–0.49 秒/步**（模型加载后）。

**解读建议：**
- 优先看曲线图：预测与真值**趋势是否一致**，而非只看 MSE 绝对值
- MSE 在反归一化动作空间计算，数值偏大属正常
- 不同轨迹难度不同，单条轨迹差不代表整体失败

### 8.4 与论文指标的区别

| 评估类型 | 输入 | 输出指标 | 是否需要真机 |
|---------|------|---------|-------------|
| 训练 Loss | 批量训练样本 | loss.jsonl | 否 |
| 开环评估 | 历史轨迹画面 | MSE/MAE + 曲线 | 否 |
| 闭环/真机 | 实时摄像头 | SR、PS | **是** |

---

## 九、部署（真机推理）

训练完成后，用 HF 格式权重启动推理服务：

```bash
conda activate lingbotvla
cd /root/autodl-tmp

python scripts/deploy.py \
  --model-path lingbot-vla/output/checkpoints/global_step_1024/hf_ckpt
```

内部调用 `deploy.lingbot_vla_policy`，支持 WebSocket 远程推理：机械臂端发图片，GPU 服务器返回动作。

部署前确认：
- `QWEN25_PATH` 指向本地 Qwen 权重
- `hf_ckpt/` 目录完整（4 个 safetensors 分片 + config）

---

## 十、完整命令速查（复制即用）

```bash
conda activate lingbotvla
cd /root/autodl-tmp

# 0. 环境
python scripts/verify_env.py

# 1. 模型（首次有网）
python scripts/download_models.py
python scripts/setup_local_models.py

# 2. 数据处理
python scripts/compute_norm.py --gpu 0

# 3. 训练
python scripts/train.py --gpu 0

# 4. Loss 曲线
python scripts/plot_training_loss.py \
  --loss-file lingbot-vla/output/checkpoints/loss.jsonl

# 5. 开环评估
python scripts/open_loop_eval.py --traj-ids 0 1 2

# 6. 真机部署
python scripts/deploy.py \
  --model-path lingbot-vla/output/checkpoints/global_step_1024/hf_ckpt
```

---

## 十一、目录与文件对照

```
/root/autodl-tmp/
├── data/raw/svla_so101_pickplace/     # 训练数据（83MB）
├── models/pretrained/                  # 预训练模型（symlink → HF cache）
├── lingbot-vla/
│   ├── configs/vla/so100.yaml         # 训练主配置
│   ├── configs/robot_configs/so100.yaml
│   ├── assets/norm_stats/so100_svla.json
│   └── output/checkpoints/
│       ├── loss.jsonl
│       └── global_step_1024/
│           ├── model/                 # DCP（可删）
│           └── hf_ckpt/               # 部署用（~16GB）
├── logs/
│   ├── training_loss_step1024.png
│   └── open_loop_eval/{0,1,2}.png
├── scripts/                           # 全部 Python 入口
└── docs/
    ├── full_workflow_summary.md       # 本文档
    ├── environment_setup.md           # 环境详解
    └── lingbot_vla_guide.md           # VLA 原理通俗说明
```

---

## 十二、常见问题

| 现象 | 原因 | 处理 |
|------|------|------|
| `configs/robot_configs/so100.yaml` 找不到 | cwd 不在 `lingbot-vla/` | 用 `scripts/open_loop_eval.py` |
| `KeyError: 'norm_stats'` | 归一化 JSON 格式错 | `python scripts/compute_norm.py` |
| `torchcodec _convert_to_tensor` 刷屏 | torchcodec 0.6 与 torch 2.7 不匹配 | `pip install torchcodec==0.5` |
| CUDA OOM / 显存被占满 | 重复启动训练 | `pkill -f train_lingbotvla.py` |
| checkpoint 保存失败 | 磁盘满（每个 ~21GB） | `save_steps: 0` + `save_optimizer: false` |
| 训练结束无 hf_ckpt | 磁盘满 | `convert_checkpoint_to_hf.py` 手动转换 |
| base 环境缺 wandb | 未 activate lingbotvla | `conda activate lingbotvla` |

---

## 十三、相关文档

| 文档 | 内容 |
|------|------|
| [README.md](../README.md) | 项目入口与脚本索引 |
| [environment_setup.md](environment_setup.md) | 环境安装、Step/Epoch、磁盘、排错 |
| [lingbot_vla_guide.md](lingbot_vla_guide.md) | VLA 原理、数据采集、部署概念 |

---

*最后更新：2026-06-28 · 基于 AutoDL 实测 1024 step 训练与开环评估结果整理*
