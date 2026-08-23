 # pot14 完整指南：权重托管、训练、云端推理、本地串口

仓库权重：[coisini9293/lingbot_pot14](https://huggingface.co/coisini9293/lingbot_pot14)

适用场景：自采 CSV+MKV → 微调 LingBot-VLA → AutoDL 推理 → 本地串口控臂。

---

## 一、上传权重到 Hugging Face

在 **AutoDL 训练机**上执行（需已有 `hf_ckpt`）：

```bash
conda activate lingbotvla
cd /root/autodl-tmp

# 首次登录（浏览器复制 token，或 export HF_TOKEN=hf_xxx）
huggingface-cli login

# 上传约 16GB：权重 + norm + robot/vla 配置 + model card
python scripts/upload_pot14_to_hf.py --repo-id coisini9293/lingbot_pot14
```

可选参数：

```bash
# 仓库不存在时创建为私有
python scripts/upload_pot14_to_hf.py --private

# 指定其它 step 目录
python scripts/upload_pot14_to_hf.py \
  --ckpt-dir lingbot-vla/output/pot14/checkpoints/global_step_1024/hf_ckpt
```

上传后页面：https://huggingface.co/coisini9293/lingbot_pot14

**不必上传：** `global_step_*/model/`（DCP 训练格式）、原始 `data/data/`、预训练基座（公开可下）。

---

## 二、从 Hugging Face 下载权重

### 2.1 微调后的 pot14 权重

```bash
conda activate lingbotvla
cd /root/autodl-tmp

# 国内建议镜像
export HF_ENDPOINT=https://hf-mirror.com

huggingface-cli download coisini9293/lingbot_pot14 \
  --local-dir models/finetuned/lingbot_pot14
```

下载后建议把配置拷回训练代码期望位置：

```bash
mkdir -p lingbot-vla/assets/norm_stats
cp models/finetuned/lingbot_pot14/configs/pot14_right_arm.json \
   lingbot-vla/assets/norm_stats/pot14_right_arm.json

cp models/finetuned/lingbot_pot14/configs/robot_pot14.yaml \
   lingbot-vla/configs/robot_configs/pot14.yaml

cp models/finetuned/lingbot_pot14/configs/vla_pot14.yaml \
   lingbot-vla/configs/vla/pot14.yaml
```

### 2.2 推理还需要的 Qwen 处理器

```bash
huggingface-cli download Qwen/Qwen2.5-VL-3B-Instruct \
  --local-dir models/pretrained/Qwen2.5-VL-3B-Instruct

export QWEN25_PATH=/root/autodl-tmp/models/pretrained/Qwen2.5-VL-3B-Instruct
```

### 2.3 若还要继续微调（不是只推理）

还要基座 VLA：

```bash
huggingface-cli download robbyant/lingbot-vla-4b \
  --local-dir models/pretrained/lingbot-vla-4b
# 或用项目脚本：
python scripts/download_models.py
python scripts/setup_local_models.py
```

---

## 三、添加数据并训练

### 3.1 数据格式（原始采集）

放到 `data/data/`：

| 文件 | 含义 |
|------|------|
| `data_XXX.csv` | 50Hz，14 维关节（左 7 + 右 7） |
| `data_XXX_metadata.json` | 任务名、标定 |
| `N.mkv` | 1280×720 三路拼接视频 |

详细字段与转换逻辑见 [pot14_dataset_finetune.md](pot14_dataset_finetune.md)。

### 3.2 转换 → 归一化 → 训练

```bash
conda activate lingbotvla
cd /root/autodl-tmp
python scripts/verify_env.py

# 1) 原始 → LeRobot
python scripts/convert_pot14_to_lerobot.py --overwrite
# 输出: data/processed/pot14_right_arm/

# 2) 归一化
python scripts/compute_norm.py --preset pot14

# 3) 训练（默认 1024 step ≈ 1.02 epoch）
python scripts/train.py --preset pot14 --gpu 0

# 想多训几轮，例如约 3 epoch：
# python scripts/train.py --preset pot14 --gpu 0 --max-steps 3009
```

### 3.3 训练后评估

```bash
# Loss 曲线
python scripts/plot_training_loss.py \
  --loss-file lingbot-vla/output/pot14/checkpoints/loss.jsonl \
  --output logs/pot14_training_loss.png

# 开环评估（必须用绝对路径或修好的包装脚本）
export OMP_NUM_THREADS=4
export QWEN25_PATH=/root/autodl-tmp/models/pretrained/Qwen2.5-VL-3B-Instruct

python scripts/open_loop_eval.py \
  --model-path /root/autodl-tmp/lingbot-vla/output/pot14/checkpoints/global_step_1024/hf_ckpt \
  --robo-name pot14 \
  --norm-path /root/autodl-tmp/lingbot-vla/assets/norm_stats/pot14_right_arm.json \
  --data-path /root/autodl-tmp/data/processed/pot14_right_arm \
  --traj-ids 0 1 2 \
  --save-plot-path /root/autodl-tmp/logs/open_loop_eval_pot14
```

训完可再次执行「第一节」上传新权重到 HF。

---

## 四、云端推理（AutoDL 跑模型）

AutoDL **没有物理串口**。云端只负责：收图像+关节状态 → 推理 → 回动作。

### 4.1 启动服务

```bash
conda activate lingbotvla
cd /root/autodl-tmp
export QWEN25_PATH=/root/autodl-tmp/models/pretrained/Qwen2.5-VL-3B-Instruct
export OMP_NUM_THREADS=4

# 若权重在 HF 下载目录：
python scripts/deploy.py \
  --model-path /root/autodl-tmp/models/finetuned/lingbot_pot14 \
  --qwen-path "$QWEN25_PATH"

# 或本地训练产物：
# python scripts/deploy.py \
#   --model-path output/pot14/checkpoints/global_step_1024/hf_ckpt
```

默认监听 **WebSocket `0.0.0.0:8006`**。

### 4.2 AutoDL 端口

在 AutoDL 控制台把自定义服务端口 **8006** 映射出来，记下：

```text
ws://<公网IP或域名>:<映射端口>
```

本机可用健康检查（若开启 HTTP health）：

```bash
curl http://<公网IP>:<端口>/healthz
```

### 4.3 算力说明

| 项目 | 建议 |
|------|------|
| 设备 | NVIDIA GPU（代码写死 CUDA） |
| 显存 | 推理约 10–20GB（远小于训练） |
| M4 Mac | **不能直接跑本仓库 deploy**；可做本地串口客户端 |
| 延迟 | 开环实测约 0.5 秒/次动作块 |

---

## 五、本地串口 + 连云端推理

### 5.1 分工

```text
本地电脑
  ├─ USB 串口 → 机械臂
  ├─ USB 摄像头 → 画面
  └─ WebSocket 客户端 → AutoDL:8006

AutoDL
  └─ deploy.py 推理服务
```

### 5.2 通信约定（与现有 server 一致）

1. 本地组装 `observation`（图像、关节状态、任务文本）
2. msgpack 发给服务器
3. 收到 `action` chunk（一次多步，deploy 默认 `use_length=16`）
4. 本地按串口协议逐帧写出

**推荐客户端（pot14 专用）：** [`deploy/pot14_local_client.py`](../deploy/pot14_local_client.py)

- 观测键：`observation.images.top/left/right` + `observation.state`（7 维）+ `task`
- 支持 AutoDL 公网 `https://...:8443`（自动转 `wss://`）
- `--mode mock-latency`：模拟串口，测推理延迟并写 `logs/pot14_runs/*.jsonl`
- `--mode serial`：真实串口（写 ESP32 RAW：`P`+14×ADC+`*`+CRC16；读 key=value 状态行）

```bash
# Mac：查蓝牙串口
python deploy/pot14_local_client.py --list-ports

# 1) AutoDL 上先启动服务（内部仍监听 6006）
# python scripts/deploy.py --model-path ... --port 6006 --norm-path ...

# 2) Mac 上测延迟（不接真机）
python deploy/pot14_local_client.py --mode mock-latency \
  --ws-url https://<你的AutoDL公网域名>:8443 \
  --steps 20

# 3) Mac 上接真机
python deploy/pot14_local_client.py --mode serial \
  --serial-port /dev/cu.XXXX \
  --ws-url https://<你的AutoDL公网域名>:8443 \
  --task "完成任务：杯子"
```

仓库里旧的 `deploy/test.py` 是 **SO-100（6 维）** 示例，**不要用于 pot14**。

### 5.3 关键对齐点（否则控臂会偏）

| 项 | 要求 |
|----|------|
| 关节定义 | 与 `robot_pot14.yaml` 一致：前 6 维臂 + 第 7 维 effector |
| 角度单位 | 训练用 `relative_rad`（相对标定零点） |
| 相机 | 三路裁剪方式与转换脚本一致（俯视+左右） |
| 任务文本 | 与训练类似：`完成任务：杯子` |
| 归一化 | 服务端加载 `pot14_right_arm.json` |

启动 deploy 时建议显式传 norm（若包装脚本未带）：

```bash
cd lingbot-vla
python -m deploy.lingbot_vla_policy \
  --model_path /root/autodl-tmp/models/finetuned/lingbot_pot14 \
  --norm_path /root/autodl-tmp/lingbot-vla/assets/norm_stats/pot14_right_arm.json \
  --use_length 16 \
  --port 8006
```

---

## 六、常用命令速查

```bash
# --- 上传 ---
huggingface-cli login
python scripts/upload_pot14_to_hf.py

# --- 下载 ---
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download coisini9293/lingbot_pot14 \
  --local-dir models/finetuned/lingbot_pot14

# --- 新数据训练 ---
python scripts/convert_pot14_to_lerobot.py --overwrite
python scripts/compute_norm.py --preset pot14
python scripts/train.py --preset pot14 --gpu 0

# --- 云端推理 ---
export QWEN25_PATH=.../Qwen2.5-VL-3B-Instruct
python scripts/deploy.py \
  --model-path models/finetuned/lingbot_pot14
```

---

## 七、相关文档

| 文档 | 内容 |
|------|------|
| [pot14_dataset_finetune.md](pot14_dataset_finetune.md) | 数据转换细节 |
| [environment_setup.md](environment_setup.md) | 环境安装、磁盘、排错 |
| [full_workflow_summary.md](full_workflow_summary.md) | SO-100 全流程（对照用） |
| [hf_model_card_pot14.md](hf_model_card_pot14.md) | 上传到 HF 的 model card 原文 |

---

*仓库: [coisini9293/lingbot_pot14](https://huggingface.co/coisini9293/lingbot_pot14)*

## 八、本地前端控制台（可选）

推荐使用带 GSAP 动画的控制台（`frontend-design` + `gsap-core`）：

```bash
# Mac
pip install fastapi uvicorn numpy msgpack websockets opencv-python pyserial pyobjc-framework-AVFoundation
cd lingbot-main
python deploy/pot14_console_server.py
# 浏览器打开 http://127.0.0.1:7860
```

功能要点：
- **相机**：用 AVFoundation 列出三路 USB（排除 FaceTime）；网页下拉自行指定 top / left / right
- **串口**：下拉选择（优先 `cu.usbserial*`）
- **动作限速**：默认 **15 Hz**；「每轴最大步进」默认 **0（无限制）**，需要时可再调小
- 列设备：`python deploy/list_avf_cameras.py`

AutoDL 需先 `deploy.py --port 6006` 并保持运行。
若客户端报 `keepalive ping timeout`：同步最新代码后**重启** AutoDL 上的 deploy。
 cd /root/autodl-tmp/lingbot


 conda activate lingbotvla
cd /root/autodl-tmp/lingbot

python scripts/deploy.py \
  --model-path /root/autodl-tmp/lingbot/models/finetuned/lingbot_pot14 \
  --qwen-path /root/autodl-tmp/lingbot/models/pretrained/Qwen2.5-VL-3B-Instruct \
  --norm-path /root/autodl-tmp/lingbot/lingbot-vla/assets/norm_stats/pot14_right_arm.json \
  --port 6006