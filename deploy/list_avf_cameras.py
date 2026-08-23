#!/usr/bin/env python3
"""CLI：列出 AVFoundation 摄像头。预览请用控制台网页。

  python deploy/list_avf_cameras.py
"""

from __future__ import annotations

from camera_avf import list_avf_devices, suggest_camera_indices


def main() -> int:
    print("=== AVFoundation 视频设备（已排除 FaceTime）===\n")
    cams = list_avf_devices(exclude_builtin=True)
    if not cams:
        print("未找到。检查：pip install pyobjc-framework-AVFoundation；系统设置→相机权限；退出 OBS")
        return 1
    for c in cams:
        print(f"avf={c['index']}  {c['name']}")
        print(f"  unique_id: {c.get('unique_id')}")
        print()
    print(f"合计: {len(cams)}")
    print("建议 --camera-indices", " ".join(str(i) for i in suggest_camera_indices(cams)))
    print("前端可自行选择 top/left/right 对应哪一路 avf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
