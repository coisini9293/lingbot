#!/usr/bin/env python3
"""pot14 本地控制台后端（FastAPI）+ 静态前端。

Mac:
  pip install fastapi uvicorn python-multipart
  python deploy/pot14_console_server.py
  # 打开 http://127.0.0.1:7860
"""

from __future__ import annotations

import logging
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Generator, Literal

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from pot14_local_client import (
    DEFAULT_ACTION_FPS,
    DEFAULT_MAX_JOINT_STEP,
    DEFAULT_ROBO_NAME,
    DEFAULT_TASK,
    DEFAULT_WS_URL,
    MockArmIO,
    RunRecorder,
    SerialArmIO,
    StepRecord,
    WebsocketClientPolicy,
    _as_float_or_none,
    build_pot14_obs,
    flatten_action_chunk,
    hardware_jog_test,
    limit_action_step,
    list_serial_ports,
    list_usb_cameras,
    normalize_ws_url,
    suggest_camera_indices,
)

STATIC_DIR = Path(__file__).resolve().parent / "web"
LOG_DIR = Path(__file__).resolve().parent.parent / "logs" / "pot14_runs"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("pot14_console")

app = FastAPI(title="pot14 console")


class CameraHub:
    """浏览器 MJPEG 预览：独占打开摄像头。

    Mac 上同一路相机不能被预览流和枚举同时打开；刷新列表前必须 pause 释放。
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._caps: dict[int, Any] = {}
        self._paused = False
        self._epoch = 0
        self._jpeg_quality = 70

    def pause(self) -> None:
        """停止预览占用并释放全部 VideoCapture（枚举/推理前调用）。"""
        with self._lock:
            self._paused = True
            self._epoch += 1
            self._release_all_unlocked()

    def resume(self) -> None:
        with self._lock:
            self._paused = False

    @property
    def epoch(self) -> int:
        with self._lock:
            return self._epoch

    def _release_all_unlocked(self) -> None:
        for idx, cap in list(self._caps.items()):
            try:
                cap.release()
            except Exception:  # noqa: BLE001
                pass
            logger.info("已释放预览相机 index=%s", idx)
        self._caps.clear()

    def close(self) -> None:
        with self._lock:
            self._paused = True
            self._epoch += 1
            self._release_all_unlocked()

    def _ensure_cap(self, index: int) -> Any | None:
        if self._paused:
            return None
        cap = self._caps.get(index)
        if cap is not None and cap.isOpened():
            return cap
        if cap is not None:
            try:
                cap.release()
            except Exception:  # noqa: BLE001
                pass
            self._caps.pop(index, None)
        try:
            from camera_avf import open_avf_capture

            cap = open_avf_capture(index, width=640, height=480, fps=12)
        except Exception as exc:  # noqa: BLE001
            logger.warning("预览打开 avf=%s 失败: %s", index, exc)
            return None
        self._caps[index] = cap
        return cap

    def read_jpeg(self, index: int, max_width: int = 480) -> bytes | None:
        import cv2  # type: ignore

        with self._lock:
            if self._paused:
                return None
            cap = self._ensure_cap(index)
            if cap is None:
                return None
            ok, frame = cap.read()
            if not ok or frame is None:
                return None
            h, w = frame.shape[:2]
            if w > max_width:
                scale = max_width / float(w)
                frame = cv2.resize(frame, (max_width, max(1, int(h * scale))))
            ok, buf = cv2.imencode(
                ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self._jpeg_quality]
            )
            if not ok:
                return None
            return buf.tobytes()

    def mjpeg_generator(self, index: int, fps: float = 8.0) -> Generator[bytes, None, None]:
        interval = 1.0 / max(fps, 1.0)
        blank = (
            b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
            + _blank_jpeg()
            + b"\r\n"
        )
        my_epoch = self.epoch
        try:
            while True:
                if self._paused or self.epoch != my_epoch:
                    break
                t0 = time.perf_counter()
                jpeg = self.read_jpeg(index)
                if jpeg is None:
                    yield blank
                else:
                    yield (
                        b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                        + jpeg
                        + b"\r\n"
                    )
                elapsed = time.perf_counter() - t0
                time.sleep(max(0.0, interval - elapsed))
        except GeneratorExit:
            raise
        finally:
            logger.debug("mjpeg 结束 index=%s epoch=%s", index, my_epoch)


def _blank_jpeg() -> bytes:
    """1x1 灰图占位（摄像头忙/暂停时）。"""
    import cv2  # type: ignore

    img = np.full((120, 160, 3), 40, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    return buf.tobytes() if ok else b""


camera_hub = CameraHub()
_run_lock = threading.Lock()


class RunRequest(BaseModel):
    mode: Literal["test", "serial"] = "test"
    ws_url: str = DEFAULT_WS_URL
    task: str = DEFAULT_TASK
    steps: int = Field(default=160, ge=1, le=300)
    serial_port: str = ""
    camera_indices: list[int] = Field(default_factory=lambda: [0, 1, 2])
    execute_chunk: bool = False
    action_fps: float = Field(default=15.0, ge=0.5, le=60.0)
    max_joint_step: float = Field(default=0.0, ge=0.0, le=1.0)
    mock_state: bool = False


class JogRequest(BaseModel):
    serial_port: str
    joint_index: int = Field(default=0, ge=0, le=6)
    peak_delta_adc: int = Field(default=160, ge=30, le=300)
    step_adc: int = Field(default=40, ge=1, le=64)
    hold_seconds: float = Field(default=1.0, ge=0.2, le=5.0)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/defaults")
def api_defaults() -> dict[str, Any]:
    # 枚举前释放预览占用，避免 Mac 上 USB 相机被占后只剩 FaceTime
    camera_hub.pause()
    try:
        time.sleep(0.4)
        cams = list_usb_cameras(exclude_builtin=True)
        return {
            "ws_url": DEFAULT_WS_URL,
            "task": DEFAULT_TASK,
            "steps": 160,
            "camera_indices": suggest_camera_indices(cams),
            "action_fps": 15.0,
            "max_joint_step": 0.0,
            "cameras": cams,
        }
    finally:
        camera_hub.resume()


@app.get("/api/ports")
def api_ports() -> dict[str, Any]:
    return {"ports": list_serial_ports()}


@app.post("/api/cameras/release")
def api_cameras_release() -> dict[str, str]:
    """前端刷新/切换前调用：打断 MJPEG 并释放设备。"""
    camera_hub.pause()
    time.sleep(0.35)
    return {"ok": "released"}


@app.post("/api/jog")
def api_jog(req: JogRequest) -> dict[str, Any]:
    """不走模型：右臂单轴缓升/缓降，验证串口能否控臂。"""
    if not req.serial_port.strip():
        return {"ok": False, "error": "请选择串口"}
    if not _run_lock.acquire(blocking=False):
        return {"ok": False, "error": "已有任务在跑，请先等结束"}
    try:
        camera_hub.pause()
        time.sleep(0.2)
        result = hardware_jog_test(
            port=req.serial_port.strip(),
            joint_index=req.joint_index,
            peak_delta_adc=req.peak_delta_adc,
            step_adc=req.step_adc,
            hold_seconds=req.hold_seconds,
        )
        return result
    except Exception as exc:  # noqa: BLE001
        logger.exception("jog 失败")
        return {"ok": False, "error": str(exc), "traceback": traceback.format_exc()}
    finally:
        camera_hub.resume()
        _run_lock.release()


@app.get("/api/cameras")
def api_cameras() -> dict[str, Any]:
    try:
        camera_hub.pause()
        time.sleep(0.35)
        try:
            cams = list_usb_cameras(exclude_builtin=True)
            note = None
            if len(cams) < 3:
                note = (
                    f"目前只检测到 {len(cams)} 路非 FaceTime 相机，推理需要 3 路。"
                    "请退出 OBS 后点「刷新摄像头」。"
                )
            return {
                "cameras": cams,
                "suggested_indices": suggest_camera_indices(cams),
                "note": note,
            }
        finally:
            camera_hub.resume()
    except Exception as exc:  # noqa: BLE001
        camera_hub.resume()
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/camera/{index}/mjpeg")
def api_camera_mjpeg(index: int) -> StreamingResponse:
    """实时 MJPEG 流（仅用于预览；推理开始时会 pause）。"""
    if index < 0 or index > 64:
        raise HTTPException(status_code=400, detail="非法 camera index")
    return StreamingResponse(
        camera_hub.mjpeg_generator(index),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


@app.post("/api/run")
def api_run(req: RunRequest) -> dict[str, Any]:
    if not _run_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="已有任务在运行")
    lines: list[str] = []
    camera_hub.pause()
    try:
        uri = normalize_ws_url(req.ws_url.strip())
        lines.append(
            f"mode={req.mode} ws={uri} steps={req.steps} "
            f"action_fps={req.action_fps} max_joint_step={req.max_joint_step} "
            f"mock_state={req.mock_state}"
        )

        if req.mode == "test":
            arm = MockArmIO(seed=0)
            run_name = "mock_latency_web"
            do_chunk = req.execute_chunk
        else:
            if not req.serial_port.strip():
                raise ValueError("实机模式需要 serial_port")
            if len(req.camera_indices) != 3:
                raise ValueError("camera_indices 需要 3 个整数")
            arm = SerialArmIO(
                port=req.serial_port.strip(),
                camera_indices=tuple(int(x) for x in req.camera_indices),
                mock_state=bool(req.mock_state),
            )
            run_name = "serial_web"
            do_chunk = True

        client = WebsocketClientPolicy(uri=uri, connect_timeout=90.0, insecure_ssl=True)
        recorder = RunRecorder(LOG_DIR, run_name=run_name)

        arm.connect()
        last_cmd: np.ndarray | None = None
        try:
            lines.append("正在 reset（服务端首次加载可能较慢）…")
            reset_resp = client.reset(DEFAULT_ROBO_NAME)
            lines.append(f"reset OK keys={list(reset_resp.keys())}")
            for step in range(req.steps):
                try:
                    state = arm.read_state()
                    images = arm.read_images()
                    # 简单健康检查：全黑图通常表示相机没打开
                    black = all(float(img.mean()) < 1.0 for img in images.values())
                    if black:
                        raise RuntimeError(
                            "三路图像几乎全黑：相机未打开或仍被预览占用。"
                            "请先点「关闭预览/准备实机」后再跑。"
                        )
                    obs = build_pot14_obs(images=images, state=state, task=req.task)
                    lines.append(f"[{step+1}/{req.steps}] 采图完成，请求推理…")
                    action_chunk = client.infer(obs)
                    rtt_ms = float(action_chunk.get("_client_rtt_ms", -1))
                    timing = action_chunk.get("server_timing") or {}
                    actions = flatten_action_chunk(action_chunk)
                    if do_chunk:
                        for action in actions:
                            limited = limit_action_step(
                                action, last_cmd, req.max_joint_step
                            )
                            arm.write_action(limited)
                            last_cmd = limited
                            if req.action_fps > 0:
                                time.sleep(1.0 / req.action_fps)
                    else:
                        limited = limit_action_step(
                            actions[0], last_cmd, req.max_joint_step
                        )
                        arm.write_action(limited)
                        last_cmd = limited
                    record = StepRecord(
                        step=step,
                        client_rtt_ms=rtt_ms,
                        server_infer_ms=_as_float_or_none(timing.get("infer_ms")),
                        server_prev_total_ms=_as_float_or_none(
                            timing.get("prev_total_ms")
                        ),
                        action_shape=list(actions.shape),
                        state=[float(x) for x in state.tolist()],
                        first_action=[float(x) for x in actions[0].tolist()],
                        ok=True,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.exception("step fail")
                    record = StepRecord(
                        step=step,
                        client_rtt_ms=-1.0,
                        server_infer_ms=None,
                        server_prev_total_ms=None,
                        action_shape=[],
                        state=[],
                        first_action=[],
                        ok=False,
                        error=str(exc),
                    )
                recorder.add(record)
                infer = (
                    f"{record.server_infer_ms:.1f}ms"
                    if record.server_infer_ms is not None
                    else "n/a"
                )
                status = "OK" if record.ok else f"FAIL:{record.error}"
                lines.append(
                    f"[{step+1}/{req.steps}] rtt={record.client_rtt_ms:.1f}ms "
                    f"infer={infer} action={record.action_shape} {status}"
                )
                if not record.ok:
                    break
        finally:
            arm.disconnect()
            client.close()

        summary = recorder.finalize(
            extra={
                "task": req.task,
                "ws_uri": uri,
                "mode": req.mode,
                "action_fps": req.action_fps,
                "max_joint_step": req.max_joint_step,
            }
        )
        return {
            "ok": True,
            "log": "\n".join(lines),
            "summary": summary,
            "jsonl": str(recorder.jsonl_path),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "log": "\n".join(lines) + "\n" + traceback.format_exc(),
            "summary": {"error": str(exc)},
            "jsonl": "",
        }
    finally:
        camera_hub.resume()
        _run_lock.release()


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def main() -> None:
    import uvicorn

    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    uvicorn.run(app, host="127.0.0.1", port=7860, log_level="info")


if __name__ == "__main__":
    main()
