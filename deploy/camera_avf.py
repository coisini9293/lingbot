#!/usr/bin/env python3
"""macOS AVFoundation 摄像头枚举与打开（OpenCV CAP_AVFOUNDATION）。

Mac 上 OpenCV 默认 index 会与 FaceTime 搅在一起；应以
AVCaptureDevice.devicesWithMediaType_(Video) 的顺序为 avf_index。
"""

from __future__ import annotations

import logging
import platform
from typing import Any

logger = logging.getLogger("camera_avf")

_BUILTIN_KEYS = (
    "facetime",
    "continuity",
    "iphone",
    "ipad",
    "desk view",
    "center stage",
    "isight",
)


def is_builtin_camera(name: str, unique_id: str = "") -> bool:
    blob = f"{name} {unique_id}".lower()
    return any(k in blob for k in _BUILTIN_KEYS)


def list_avf_devices(*, exclude_builtin: bool = True) -> list[dict[str, Any]]:
    """列出视频设备。每项含 index(=avf_index)、name、unique_id、is_builtin。"""
    if platform.system() != "Darwin":
        return _list_opencv_fallback(exclude_builtin=exclude_builtin)

    try:
        import AVFoundation  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Mac 需要: python3 -m pip install pyobjc-framework-AVFoundation"
        ) from exc

    media = AVFoundation.AVMediaTypeVideo
    devices = AVFoundation.AVCaptureDevice.devicesWithMediaType_(media) or []
    out: list[dict[str, Any]] = []
    for i, dev in enumerate(devices):
        name = str(dev.localizedName() or "")
        uid = str(dev.uniqueID() or "")
        model = str(dev.modelID() or "") if hasattr(dev, "modelID") else ""
        mfr = str(dev.manufacturer() or "") if hasattr(dev, "manufacturer") else ""
        builtin = is_builtin_camera(name, uid)
        if exclude_builtin and builtin:
            continue
        out.append(
            {
                "index": i,  # OpenCV CAP_AVFOUNDATION 打开时用此序号
                "avf_index": i,
                "name": name,
                "unique_id": uid,
                "model": model,
                "manufacturer": mfr,
                "is_builtin": builtin,
                "role_hint": None,
            }
        )
    for j, cam in enumerate(out[:3]):
        roles = ("top", "left", "right")
        cam["role_hint"] = roles[j]
    return out


def suggest_camera_indices(cameras: list[dict[str, Any]] | None = None) -> list[int]:
    cams = cameras if cameras is not None else list_avf_devices(exclude_builtin=True)
    idxs = [int(c["index"]) for c in cams[:3]]
    while len(idxs) < 3:
        idxs.append(len(idxs))
    return idxs


def open_avf_capture(
    index: int,
    *,
    width: int = 640,
    height: int = 480,
    fps: float = 15.0,
) -> Any:
    """按 AVFoundation 序号打开相机，并压到较低分辨率减轻 USB 带宽。"""
    import cv2  # type: ignore

    af = getattr(cv2, "CAP_AVFOUNDATION", None)
    if platform.system() == "Darwin" and af is not None:
        cap = cv2.VideoCapture(index, af)
    else:
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开摄像头 avf_index={index}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    return cap


def _list_opencv_fallback(*, exclude_builtin: bool) -> list[dict[str, Any]]:
    """非 Mac 或无 PyObjC 时的兜底枚举。"""
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise ImportError("需要 opencv-python") from exc

    found: list[dict[str, Any]] = []
    for idx in range(8):
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            continue
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            continue
        name = f"camera_{idx}"
        builtin = False
        if exclude_builtin and builtin:
            continue
        found.append(
            {
                "index": idx,
                "avf_index": idx,
                "name": name,
                "unique_id": str(idx),
                "model": "",
                "manufacturer": "",
                "is_builtin": False,
                "width": w,
                "height": h,
                "role_hint": None,
            }
        )
    for j, cam in enumerate(found[:3]):
        cam["role_hint"] = ("top", "left", "right")[j]
    return found
