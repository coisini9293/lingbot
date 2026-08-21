# 新服务器开训指南（公开源下载，不上传大文件）

适合场景：换一台新机器（如 AutoDL 开卡实例），用**新数据集**做微调。

## 结论先说

| 资产 | 怎么处理 | 原因 |
|------|----------|------|
| 代码 / 配置 / 脚本 | 从 GitHub 拉 | 体积小 |
| 预训练权重 `lingbot-vla-4b` | 从公开 HF 仓库下 | 已公开，不必自己再传 |
| `Qwen2.5-VL-3B-Instruct` | 从公开 HF 仓库下 | 同上 |
| `flash-attn` wheel | 从官方 GitHub Release 下 | 上传镜像极慢，官方包即可 |
| 旧 `checkpoints/global_step_*` | **不要** | 那是旧任务产物；新数据要重新训 |
| 自己的 HF Dataset | 可选，仅放说明文档 | 大文件上传走 LFS，国内很慢 |

公开地址：

- 代码：https://github.com/coisini9293/lingbot
- VLA 权重：https://huggingface.co/robbyant/lingbot-vla-4b
- Qwen：https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct
- flash-attn 2.8.3 Release：https://github.com/Dao-AILab/flash-attention/releases/tag/v2.8.3

## 新机器最短路径

```bash
# 0) 国内镜像（下载用）
export HF_ENDPOINT=https://hf-mirror.com

# 1) 拉代码
git clone https://github.com/coisini9293/lingbot.git
cd lingbot
conda activate lingbotvla   # 或按 docs/environment_setup.md 新建环境

# 2) 一键下权重 + wheel 并安装 flash-attn
bash scripts/bootstrap_new_server.sh

# 若 GitHub 下 wheel 失败，把脚本里的 URL 换成加速镜像，例如:
# https://ghfast.top/https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.7cxx11abiFALSE-cp312-cp312-linux_x86_64.whl

# 3) 环境自检
python scripts/verify_env.py

# 4) 放入你的新数据集后，重新算归一化（必须）
python scripts/compute_norm.py

# 5) 微调
python scripts/train.py --gpu 0
```

等价 Python 入口：`python scripts/bootstrap_from_public.py`

## 下载后目录对应

| 本地路径 | 来源 |
|----------|------|
| `models/pretrained/lingbot-vla-4b/` | `robbyant/lingbot-vla-4b` |
| `models/pretrained/Qwen2.5-VL-3B-Instruct/` | `Qwen/Qwen2.5-VL-3B-Instruct` |
| `wheels/flash_attn-*.whl` | Dao-AILab Release |

训练配置见 `lingbot-vla/configs/vla/so100.yaml`，把 `model_path` / `tokenizer_path` 指到上述本地目录（或继续用 HF repo id，由缓存解析）。

## 为什么不推荐往自己的 Dataset 传大文件

1. AutoDL 等环境常**无法直连** `huggingface.co`，只能走 `hf-mirror.com`
2. 大文件走 LFS multipart，完成回调域名曾出现 `hf-mirror.org` 解析失败
3. 实测上传 `flash_attn`（约 256MB）可慢到数小时；预训练权重约 20GB+ 更不现实
4. 这些文件本来就有稳定公开源，重复上传无收益

若仍要上传小文件/说明到 Dataset，可用 `scripts/upload_hf_large_files.py`（已改成默认不传 checkpoint），并设置：

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
export HF_TOKEN=hf_xxx
# 可选：echo '160.16.86.14 hf-mirror.org' >> /etc/hosts
```

## 和旧机器产物的关系

- **要带走的**：本仓库代码、你的新采集数据、自己的配置改动  
- **不必带走的**：`.cache/`、`lingbot-vla/output/checkpoints/`、本地 `flash_attn-*.whl`（可再下）  
- **换新数据必做**：`compute_norm.py`，不要复用旧 `norm_stats`（除非数据分布真的一样）

## 相关文档

- [environment_setup.md](environment_setup.md) — 环境依赖与自检清单  
- [full_workflow_summary.md](full_workflow_summary.md) — 完整训练/评估流程  
- [hf_dataset_card.md](hf_dataset_card.md) — Dataset 卡片说明（可选）
