#!/usr/bin/env python3
"""将 pot14 自采数据 (CSV + 三路拼接 MKV) 转为 LeRobot v3.0，供 Lingbot-VLA 微调。

默认策略（与当前 15 条数据统计一致）:
  - 用 *_relative_rad 作为绝对角（相对标定零点）
  - 自动检测活动臂；映射为 7 维 [joint1..joint6 + joint7(as effector)]
  - 将 1280x720 拼接画面裁成 top / left / right 三路相机
  - 按时间把 50Hz 关节与 60fps 视频对齐到统一 fps（默认 30）

用法:
  conda activate lingbotvla
  python scripts/convert_pot14_to_lerobot.py
  python scripts/convert_pot14_to_lerobot.py --limit 1   # 先转 1 条验证
  python scripts/convert_pot14_to_lerobot.py --dry-run   # 只分析左右臂，不写数据集
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
# 原始采集默认在 data/raw/pot14；若仍用旧路径 data/data，请传 --src
DEFAULT_SRC = ROOT / "data" / "raw" / "pot14"
DEFAULT_OUT = ROOT / "data" / "processed" / "pot14_right_arm"

# 拼接布局（1280x720）: 上=俯视，下左=左侧，下右=右侧
# 若你的分割线不是正中间，可用 --top-ratio 调整
DEFAULT_TOP_RATIO = 0.5
DEFAULT_FPS = 30
IMAGE_HW = (240, 320)  # H, W — 训练时还会再缩到 224

LEFT_JOINTS = [
    "p0_left_joint_1_relative_rad",
    "p1_left_joint_2_relative_rad",
    "p2_left_joint_3_relative_rad",
    "p3_left_joint_4_relative_rad",
    "p4_left_joint_5_relative_rad",
    "p5_left_joint_6_relative_rad",
    "p6_left_joint_7_relative_rad",
]
RIGHT_JOINTS = [
    "p7_right_joint_1_relative_rad",
    "p8_right_joint_2_relative_rad",
    "p9_right_joint_3_relative_rad",
    "p10_right_joint_4_relative_rad",
    "p11_right_joint_5_relative_rad",
    "p12_right_joint_6_relative_rad",
    "p13_right_joint_7_relative_rad",
]

JOINT_NAMES = [
    "joint_1.pos",
    "joint_2.pos",
    "joint_3.pos",
    "joint_4.pos",
    "joint_5.pos",
    "joint_6.pos",
    "joint_7.pos",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert pot14 CSV+MKV to LeRobot v3")
    p.add_argument("--src", type=Path, default=DEFAULT_SRC)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--repo-id", default="local/pot14_right_arm")
    p.add_argument("--fps", type=int, default=DEFAULT_FPS)
    p.add_argument("--top-ratio", type=float, default=DEFAULT_TOP_RATIO)
    p.add_argument("--limit", type=int, default=0, help="只转换前 N 条，0=全部")
    p.add_argument("--dry-run", action="store_true", help="只输出左右臂分析，不写数据集")
    p.add_argument(
        "--force-side",
        choices=["auto", "left", "right"],
        default="auto",
        help="强制使用哪条臂；auto 按运动幅度选择",
    )
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def list_episodes(src: Path) -> list[dict]:
    eps = []
    for csv_path in sorted(src.glob("data_*.csv")):
        m = re.match(r"data_(\d+)\.csv$", csv_path.name)
        if not m:
            continue
        idx = int(m.group(1))
        meta_path = src / f"data_{idx:03d}_metadata.json"
        mkv_path = src / f"{idx}.mkv"
        if not meta_path.exists() or not mkv_path.exists():
            print(f"[跳过] 缺文件: {csv_path.name}")
            continue
        eps.append(
            {
                "index": idx,
                "csv": csv_path,
                "meta": meta_path,
                "mkv": mkv_path,
            }
        )
    return eps


def load_csv_joints(csv_path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    with open(csv_path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"空 CSV: {csv_path}")
    cols = list(rows[0].keys())
    for c in LEFT_JOINTS + RIGHT_JOINTS:
        if c not in cols:
            raise KeyError(f"{csv_path.name} 缺少列 {c}")
    elapsed = np.array([float(r["elapsed_s"]) for r in rows], dtype=np.float64)
    left = np.array([[float(r[c]) for c in LEFT_JOINTS] for r in rows], dtype=np.float32)
    right = np.array([[float(r[c]) for c in RIGHT_JOINTS] for r in rows], dtype=np.float32)
    return elapsed, left, right


def classify_side(left: np.ndarray, right: np.ndarray) -> tuple[str, float, float]:
    left_range = float(np.ptp(left, axis=0).sum())
    right_range = float(np.ptp(right, axis=0).sum())
    if right_range < 0.05 and left_range < 0.05:
        side = "none"
    elif right_range > left_range * 5:
        side = "right"
    elif left_range > right_range * 5:
        side = "left"
    else:
        side = "both"
    return side, left_range, right_range


def select_arm(
    left: np.ndarray,
    right: np.ndarray,
    force_side: str,
) -> tuple[np.ndarray, str, float, float]:
    detected, l_r, r_r = classify_side(left, right)
    if force_side == "left":
        return left, "left", l_r, r_r
    if force_side == "right":
        return right, "right", l_r, r_r
    if detected == "left":
        return left, "left", l_r, r_r
    if detected == "right":
        return right, "right", l_r, r_r
    if detected == "both":
        # 双臂都动时仍取运动更大的一侧，并打印警告
        if r_r >= l_r:
            return right, "right(both)", l_r, r_r
        return left, "left(both)", l_r, r_r
    return right, "right(fallback)", l_r, r_r


def crop_views(frame: np.ndarray, top_ratio: float) -> dict[str, np.ndarray]:
    h, w = frame.shape[:2]
    y_split = int(h * top_ratio)
    x_mid = w // 2
    top = frame[0:y_split, :]
    left = frame[y_split:h, 0:x_mid]
    right = frame[y_split:h, x_mid:w]
    return {"top": top, "left": left, "right": right}


def resize_bgr(img: np.ndarray, hw: tuple[int, int] = IMAGE_HW) -> np.ndarray:
    out_h, out_w = hw
    return cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_AREA)


def build_features(hw: tuple[int, int] = IMAGE_HW) -> dict:
    h, w = hw
    img = {
        "dtype": "video",
        "shape": (h, w, 3),
        "names": ["height", "width", "channels"],
    }
    return {
        "observation.state": {
            "dtype": "float32",
            "shape": (7,),
            "names": JOINT_NAMES,
        },
        "action": {
            "dtype": "float32",
            "shape": (7,),
            "names": JOINT_NAMES,
        },
        "observation.images.top": dict(img),
        "observation.images.left": dict(img),
        "observation.images.right": dict(img),
    }


def nearest_index(times: np.ndarray, t: float) -> int:
    return int(np.clip(np.searchsorted(times, t), 0, len(times) - 1))


def convert_one_episode(
    dataset,
    ep: dict,
    *,
    fps: int,
    top_ratio: float,
    force_side: str,
) -> dict:
    meta = json.loads(ep["meta"].read_text(encoding="utf-8"))
    elapsed, left, right = load_csv_joints(ep["csv"])
    joints, side_used, l_range, r_range = select_arm(left, right, force_side)

    cap = cv2.VideoCapture(str(ep["mkv"]))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {ep['mkv']}")
    n_vid = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_fps = float(cap.get(cv2.CAP_PROP_FPS) or 60.0)
    duration = float(elapsed[-1]) if len(elapsed) else 0.0
    if duration <= 0:
        duration = max(0.0, (n_vid - 1) / vid_fps)

    n_out = max(1, int(duration * fps) + 1)
    task = str(meta.get("task") or "task")
    # 简单中文任务 → 稍完整指令，便于 VLA；原 task 仍写入报告
    task_text = f"完成任务：{task}" if task else "完成操作任务"

    # 预读全部帧会爆内存；按需随机 seek（mkv 可能慢），改为顺序解码 + 索引映射
    # 先按输出时间表计算需要的视频帧号，再顺序扫一遍
    needed_vid_idx = []
    csv_idx_list = []
    for i in range(n_out):
        t = i / fps
        csv_idx_list.append(nearest_index(elapsed, t))
        needed_vid_idx.append(int(np.clip(round(t * vid_fps), 0, max(0, n_vid - 1))))

    frames_cache: dict[int, np.ndarray] = {}
    unique_needed = sorted(set(needed_vid_idx))
    u_set = set(unique_needed)
    cur = 0
    ok, frame = cap.read()
    while ok and cur <= unique_needed[-1]:
        if cur in u_set:
            frames_cache[cur] = frame
        cur += 1
        ok, frame = cap.read()
    cap.release()

    for i in range(n_out):
        j = csv_idx_list[i]
        state = joints[j]
        action = joints[min(j + 1, len(joints) - 1)]
        vidx = needed_vid_idx[i]
        if vidx not in frames_cache:
            # 回退：用已有最近帧
            nearest = min(frames_cache.keys(), key=lambda k: abs(k - vidx)) if frames_cache else None
            if nearest is None:
                raise RuntimeError(f"视频无可用帧: {ep['mkv']}")
            raw = frames_cache[nearest]
        else:
            raw = frames_cache[vidx]
        views = crop_views(raw, top_ratio)
        frame_dict = {
            "observation.state": state.astype(np.float32),
            "action": action.astype(np.float32),
            "observation.images.top": resize_bgr(views["top"]),
            "observation.images.left": resize_bgr(views["left"]),
            "observation.images.right": resize_bgr(views["right"]),
            "task": task_text,
        }
        dataset.add_frame(frame_dict)

    dataset.save_episode()
    return {
        "episode_index": ep["index"],
        "task": task,
        "task_text": task_text,
        "side_used": side_used,
        "left_range": l_range,
        "right_range": r_range,
        "csv_rows": int(len(joints)),
        "out_frames": n_out,
        "video_frames": n_vid,
        "video_fps": vid_fps,
    }


def main() -> int:
    args = parse_args()
    eps = list_episodes(args.src)
    if not eps:
        print(f"未找到 episode: {args.src}")
        return 1
    if args.limit > 0:
        eps = eps[: args.limit]

    report = []
    print(f"发现 {len(list_episodes(args.src))} 条，本次处理 {len(eps)} 条")
    print(f"{'ep':>3} {'side':<14} {'L_range':>8} {'R_range':>8} task")
    for ep in eps:
        elapsed, left, right = load_csv_joints(ep["csv"])
        meta = json.loads(ep["meta"].read_text(encoding="utf-8"))
        joints, side, l_r, r_r = select_arm(left, right, args.force_side)
        print(f"{ep['index']:3d} {side:<14} {l_r:8.4f} {r_r:8.4f} {meta.get('task')}")
        report.append(
            {
                "episode_index": ep["index"],
                "detected_or_forced_side": side,
                "left_range": l_r,
                "right_range": r_r,
                "task": meta.get("task"),
                "csv_rows": len(joints),
            }
        )

    sides = Counter(r["detected_or_forced_side"].split("(")[0] for r in report)
    print("侧别统计:", dict(sides))

    report_path = args.out.parent / "pot14_arm_side_report.json"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"侧别报告: {report_path}")

    if args.dry_run:
        print("dry-run 结束，未写入 LeRobot 数据集")
        return 0

    if args.out.exists():
        if not args.overwrite:
            print(f"输出已存在: {args.out} （加 --overwrite 可重建）")
            return 1
        shutil.rmtree(args.out)

    # 延迟 import，dry-run 不需要 lerobot
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        features=build_features(IMAGE_HW),
        root=args.out,
        robot_type="pot14_single_arm",
        use_videos=True,
        image_writer_threads=4,
        image_writer_processes=0,
    )

    convert_report = []
    for ep in eps:
        print(f"\n=== 转换 episode {ep['index']} ===", flush=True)
        info = convert_one_episode(
            dataset,
            ep,
            fps=args.fps,
            top_ratio=args.top_ratio,
            force_side=args.force_side,
        )
        convert_report.append(info)
        print(
            f"完成 ep{info['episode_index']}: side={info['side_used']} "
            f"frames={info['out_frames']} task={info['task']}",
            flush=True,
        )

    dataset.finalize()
    out_report = args.out / "conversion_report.json"
    out_report.write_text(json.dumps(convert_report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n转换完成: {args.out}")
    print(f"报告: {out_report}")
    print("下一步（有 GPU 时）:")
    print("  python scripts/compute_norm.py --data-path data/processed/pot14_right_arm \\")
    print("      --norm-stats-file assets/norm_stats/pot14_right_arm.json")
    print("  python scripts/train.py  # 需先把配置指到 pot14")
    return 0


if __name__ == "__main__":
    sys.exit(main())
