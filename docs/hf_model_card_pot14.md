---
license: apache-2.0
tags:
  - robotics
  - vla
  - lingbot-vla
  - pot14
library_name: transformers
---

# lingbot_pot14

LingBot-VLA 在 **pot14 自采右臂数据**上微调到 `global_step_1024` 的部署权重。

- 基座: [robbyant/lingbot-vla-4b](https://huggingface.co/robbyant/lingbot-vla-4b)
- 任务: 杯子等（训练文本形如 `完成任务：杯子`）
- 动作: 7 维单活动臂（6 关节 + 1 effector）
- 相机: top / left / right（由拼接画面裁剪）

## 仓库内容

| 路径 | 说明 |
|------|------|
| `model-*.safetensors` + `config.json` 等 | HF 格式权重（`hf_ckpt`） |
| `configs/pot14_right_arm.json` | 归一化统计量 |
| `configs/robot_pot14.yaml` | 机器人特征映射 |
| `configs/vla_pot14.yaml` | 训练配置参考 |

## 下载

```bash
export HF_ENDPOINT=https://hf-mirror.com   # 国内可选
huggingface-cli download coisini9293/lingbot_pot14 \
  --local-dir models/finetuned/lingbot_pot14
```

推理还需要 Qwen 处理器：

```bash
huggingface-cli download Qwen/Qwen2.5-VL-3B-Instruct \
  --local-dir models/pretrained/Qwen2.5-VL-3B-Instruct
```

完整训练 / 云端推理 / 本地串口说明见项目文档  
`docs/pot14_full_guide.md`。
