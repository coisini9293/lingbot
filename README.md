# Lingbot-VLA SO-100 训练部署指南

基于 [芯电厂教程](https://hexchip.com/archives/lingbotvla-so100-train-deploy) 的完整工作流。

## 项目结构

```
Lingbot-VLA/
├── lerobot/                    # LeRobot 数据采集环境
├── lingbot-vla/                # 训练环境（含 configs/、deploy/）
├── data/raw/                   # 数据集（svla_so101_pickplace）
├── models/pretrained/          # 本地预训练模型
├── scripts/                    # 全部 Python 脚本（无 shell 依赖）
├── deploy/                     # 机械臂部署测试
├── logs/                       # 训练日志与曲线图
└── docs/                       # 文档
```

## 配置文件

使用 `lingbot-vla/configs/` 下的配置：

| 文件 | 说明 |
|------|------|
| `lingbot-vla/configs/vla/so100.yaml` | VLA 训练配置 |
| `lingbot-vla/configs/robot_configs/so100.yaml` | 机器人配置（2 摄像头） |
| `lingbot-vla/assets/norm_stats/so100_svla.json` | 归一化统计量 |

## 完整流程（纯 Python，离线）

```bash
cd /root/autodl-tmp
conda activate lingbotvla

# 0. 自检 + 准备
python scripts/verify_env.py          # 环境必须全 OK
python scripts/download_models.py     # 首次有网
python scripts/setup_local_models.py
python scripts/compute_norm.py

# 3. 冒烟测试（50 步，仅结束时保存 checkpoint）
python scripts/train.py --max-steps 50 --save-steps 0 \
    --output-dir output/smoke_test \
    --micro-batch-size 1 --grad-accum 4 \
    --use-compile false

# 4. 正式训练（1024 step，结束时保存 global_step_1024/）
python scripts/train.py --gpu 0

# 5. 生成 Loss 曲线（X 轴为 step，自动命名 training_loss_step{N}.png）
python scripts/plot_training_loss.py \
    --loss-file lingbot-vla/output/checkpoints/loss.jsonl

# 5. 开环评估（预测 vs 示范动作，输出 MSE/MAE + 曲线图）
python scripts/open_loop_eval.py \
  --model-path lingbot-vla/output/checkpoints/global_step_1024/hf_ckpt \
  --traj-ids 0 1 2

# 6. 真机部署（路径与 step 对齐）
python scripts/deploy.py \
    --model-path lingbot-vla/output/checkpoints/global_step_1024/hf_ckpt
```

## 脚本说明

| 脚本 | 用途 |
|------|------|
| `scripts/local_paths.py` | 路径常量 + 离线环境变量 |
| `scripts/launch_utils.py` | torchrun 启动器 |
| `scripts/verify_env.py` | **训练前环境自检（必跑）** |
| `scripts/install_env.py` | 安装 lingbot-vla 依赖 |
| `scripts/setup_local_models.py` | 链接 HF 缓存到 models/pretrained/ |
| `scripts/download_models.py` | 下载模型（有网时运行一次） |
| `scripts/compute_norm.py` | 生成归一化统计量 |
| `scripts/train.py` | 微调训练 |
| `scripts/deploy.py` | 启动推理服务器 |
| `scripts/plot_training_loss.py` | Loss 曲线图（英文标签） |
| `scripts/collect_data.py` | 真实机械臂数据采集 |
| `scripts/teleoperate.py` | 遥操作测试（需硬件） |

## 文档

| 文档 | 说明 |
|------|------|
| **[docs/full_workflow_summary.md](docs/full_workflow_summary.md)** | **完整流程总结（数据→训练→评估，推荐阅读）** |
| [docs/environment_setup.md](docs/environment_setup.md) | 环境配置与命令详解 |
| [docs/environment_setup.md#五附step-与-epoch-的区别](docs/environment_setup.md) | Step 与 Epoch 区别 |
| [docs/lingbot_vla_guide.md](docs/lingbot_vla_guide.md) | VLA 原理通俗说明 |

## 注意事项

1. 所有脚本均为 Python，不再依赖 `train.sh` 等 shell 脚本
2. YAML 配置文件作为**第一个位置参数**传入，不要用 `--config`
3. 训练输出在数据盘 `lingbot-vla/output/`；**checkpoint、loss 图、部署路径均以 global step 命名**，不要按 epoch（见 [Step 与 Epoch 说明](docs/environment_setup.md#五附step-与-epoch-的区别)）
4. 模型缓存建议放在 `autodl-tmp/.cache/huggingface/`
5. **训练前务必完成环境检查**：见 [docs/environment_setup.md](docs/environment_setup.md) 第一节清单（PyTorch cu128、FFmpeg、flash-attn、归一化统计量等）

### 训练产物对照（step 制）

| 产物 | 路径示例 |
|------|---------|
| Loss 日志 | `lingbot-vla/output/checkpoints/loss.jsonl` |
| Checkpoint | `lingbot-vla/output/checkpoints/global_step_1024/` |
| HF 部署权重 | `.../global_step_1024/hf_ckpt/` |
| Loss 曲线图 | `logs/training_loss_step1024.png` |
