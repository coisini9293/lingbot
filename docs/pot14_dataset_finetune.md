# pot14 自采数据微调说明

## 背景

原始采集在 `data/data/`：

| 文件 | 内容 |
|------|------|
| `data_XXX.csv` | 50Hz 关节（14 维：左臂 7 + 右臂 7） |
| `data_XXX_metadata.json` | 任务名、标定 baseline、关节映射 |
| `N.mkv` | 1280×720 **三路拼接**视频（上俯视 / 下左 / 下右） |

**不能**直接拿来训 Lingbot-VLA。需要先转成 **LeRobot v3.0**，并写对应 robot/vla 配置。

模型本身支持多种机械臂；本仓库更接近的现成模板是 **`robotwin`（双臂+三相机）**，而不是 SO-100。  
当前 15 条数据统计上 **全部是右臂主运动**，因此落地策略定为：

> 自动检测活动臂 → 映射为 **7 维单臂**（6 臂关节 + 1 effector）+ **三路相机**。

角度字段：使用 `*_relative_rad`（相对标定零点的**绝对角**）；`*_delta_rad` 为帧间增量，转换时不用。

---

## 已添加的文件

| 路径 | 作用 |
|------|------|
| `scripts/convert_pot14_to_lerobot.py` | CSV+MKV → LeRobot |
| `lingbot-vla/configs/robot_configs/pot14.yaml` | 特征映射（7 维 + 3 相机） |
| `lingbot-vla/configs/vla/pot14.yaml` | 训练配置 |
| `scripts/local_paths.py` | `POT14_*` 路径常量 |
| `scripts/train.py --preset pot14` | 训练入口 |
| `scripts/compute_norm.py --preset pot14` | 归一化入口 |

---

## 一键流程（开卡后）

```bash
conda activate lingbotvla
cd /root/autodl-tmp

# 1) 可选：只看左右臂判定
python scripts/convert_pot14_to_lerobot.py --dry-run

# 2) 转换（全量；可先 --limit 1 试跑）
python scripts/convert_pot14_to_lerobot.py --overwrite
# 输出: data/processed/pot14_right_arm/
# 报告: data/processed/pot14_arm_side_report.json

# 3) 归一化（需 GPU 环境与依赖就绪）
python scripts/compute_norm.py --preset pot14

# 4) 训练（本说明不执行；开卡后再跑）
python scripts/train.py --preset pot14 --gpu 0
```

### 常用参数

```bash
# 强制只用右臂 / 左臂（默认 auto）
python scripts/convert_pot14_to_lerobot.py --force-side right

# 俯视区域高度占比（默认 0.5；若裁切偏了可调）
python scripts/convert_pot14_to_lerobot.py --top-ratio 0.55

# 输出 fps（默认 30；关节 50Hz / 视频 60fps 会对齐到该 fps）
python scripts/convert_pot14_to_lerobot.py --fps 30
```

---

## 转换细节

1. **活动臂检测**：比较左右 `relative_rad` 的行程和；一侧远大于另一侧则选该侧。当前 15/15 为 `right`。  
2. **状态/动作**：`observation.state` = 当前绝对角；`action` = 下一帧绝对角；训练时手臂 `subtract_state: true` 会变成增量。  
3. **相机裁剪**：`H×W=720×1280`，按 `--top-ratio` 横切，再左右对半；缩放到 240×320。  
4. **时间对齐**：按输出 fps 取时间戳，分别最近邻采样 CSV 行与视频帧。  
5. **任务文本**：metadata 的 `task`（如「杯子」）写成 `完成任务：杯子`。

---

## 与 robotwin / so100 的关系

| | so100 | robotwin | pot14（本方案） |
|--|-------|----------|----------------|
| 臂 | 单臂 6 | 双臂 14+2 | 单活动臂 7 |
| 相机 | 2 | 3 | 3（裁自拼接） |
| 是否直接可用 | 否 | 骨架可参考 | **专用配置** |

以后若出现真正的左/右混采：保持 `--force-side auto` 即可自动选臂并映射到同一套 7 维；若要双臂同时控，需另写 14 维配置（可再从 robotwin 扩展）。

---

## 注意

- 转换产物较大（视频），默认在 `data/processed/`，勿提交 Git。  
- 左臂在现有数据里几乎不动；若你目视觉得「像左臂」，以 CSV 侧别报告为准，或检查 CAN1/CAN2 命名是否反了。  
- 未跑训练；开卡后按上面 3→4 步执行即可。
