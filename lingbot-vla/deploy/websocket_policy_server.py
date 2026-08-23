import asyncio
import http
import json
import logging
import os
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .msgpack_numpy import Packer, unpackb
import websockets.asyncio.server as _server
import websockets.frames

logger = logging.getLogger(__name__)

# 默认写到数据盘，便于 AutoDL 上排查
_DEFAULT_SERVER_LOG_DIR = Path(
    os.environ.get(
        "POT14_SERVER_LOG_DIR",
        "/root/autodl-tmp/lingbot/logs/pot14_server_runs",
    )
)


def _to_list(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.astype(np.float32).reshape(-1).tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _image_summary(arr: Any) -> dict[str, Any] | None:
    if not isinstance(arr, np.ndarray):
        return None
    a = arr.astype(np.float32)
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "mean": float(a.mean()) if a.size else 0.0,
        "std": float(a.std()) if a.size else 0.0,
        "min": float(a.min()) if a.size else 0.0,
        "max": float(a.max()) if a.size else 0.0,
    }


def _action_summary(action: dict[str, Any]) -> dict[str, Any]:
    """汇总动作幅度，方便看是否落在死区内。"""
    out: dict[str, Any] = {"keys": []}
    chunks: list[np.ndarray] = []
    for key, value in action.items():
        if key in ("server_timing",) or key.startswith("_"):
            continue
        if not isinstance(value, np.ndarray):
            continue
        out["keys"].append(key)
        chunks.append(value.astype(np.float32))
        flat = value.astype(np.float32).reshape(-1)
        out[key] = {
            "shape": list(value.shape),
            "abs_max": float(np.max(np.abs(flat))) if flat.size else 0.0,
            "abs_mean": float(np.mean(np.abs(flat))) if flat.size else 0.0,
            "first": _to_list(value[0] if value.ndim >= 1 else value),
        }
    if chunks:
        # 拼成 (T, D) 看整段 chunk
        try:
            joined = np.concatenate(
                [c.reshape(c.shape[0], -1) if c.ndim >= 2 else c.reshape(1, -1) for c in chunks],
                axis=-1,
            )
            out["chunk_shape"] = list(joined.shape)
            out["chunk_abs_max"] = float(np.max(np.abs(joined)))
            out["chunk_abs_mean"] = float(np.mean(np.abs(joined)))
            out["first_action"] = joined[0].tolist()
        except Exception as exc:  # noqa: BLE001
            out["join_error"] = str(exc)
    return out


def _obs_summary(obs: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "keys": sorted(list(obs.keys())),
        "task": obs.get("task"),
        "reset": bool(obs.get("reset", False)),
        "robo_name": obs.get("robo_name"),
    }
    state = obs.get("observation.state")
    if isinstance(state, np.ndarray):
        summary["state"] = _to_list(state)
        summary["state_abs_max"] = float(np.max(np.abs(state.astype(np.float32))))
    images: dict[str, Any] = {}
    for key, value in obs.items():
        if "image" in key.lower() or key.startswith("observation.images"):
            info = _image_summary(value)
            if info is not None:
                images[key] = info
    if images:
        summary["images"] = images
    return summary


class ServerInferRecorder:
    """服务端每步推理落盘 JSONL，用于排查动作过小/观测异常。"""

    def __init__(self, log_dir: Path | None = None) -> None:
        self.log_dir = Path(log_dir) if log_dir is not None else _DEFAULT_SERVER_LOG_DIR
        self.log_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.jsonl_path = self.log_dir / f"server_infer_{stamp}.jsonl"
        self.step = 0
        self._fp = self.jsonl_path.open("a", encoding="utf-8")
        logger.info("Server infer log -> %s", self.jsonl_path)

    def close(self) -> None:
        try:
            self._fp.close()
        except Exception:  # noqa: BLE001
            pass

    def write(self, record: dict[str, Any]) -> None:
        self._fp.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._fp.flush()

    def log_step(
        self,
        *,
        remote: str,
        obs: dict[str, Any],
        action: dict[str, Any],
        infer_ms: float,
        error: str | None = None,
    ) -> dict[str, Any]:
        self.step += 1
        action_info = _action_summary(action) if error is None else {}
        record = {
            "type": "infer",
            "step": self.step,
            "ts_utc": datetime.now(timezone.utc).isoformat(),
            "remote": remote,
            "infer_ms": infer_ms,
            "obs": _obs_summary(obs),
            "action": action_info,
            "error": error,
        }
        self.write(record)
        # 终端一行摘要，方便当场看幅度
        abs_max = action_info.get("chunk_abs_max")
        logger.info(
            "[infer #%d] %.1fms  state_abs_max=%s  action_abs_max=%s  task=%r",
            self.step,
            infer_ms,
            (record["obs"].get("state_abs_max")),
            abs_max,
            record["obs"].get("task"),
        )
        if isinstance(abs_max, (int, float)) and abs_max < 0.03:
            logger.warning(
                "[infer #%d] 动作幅度很小 (abs_max=%.5f rad)。"
                "若对应 ADC|Δ|<20，STM32 会死区不动作。",
                self.step,
                float(abs_max),
            )
        return record


class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        policy,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
        *,
        log_dir: str | Path | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        self._log_dir = Path(log_dir) if log_dir is not None else _DEFAULT_SERVER_LOG_DIR
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
            # AutoDL 反代 + 首次冷启动推理可能 >30s；禁用服务端 ping，
            # 避免 keepalive ping timeout 把还在推理的连接掐断。
            ping_interval=None,
            ping_timeout=None,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection):
        remote = str(websocket.remote_address)
        logger.info(f"Connection from {remote} opened")
        packer = Packer()
        recorder = ServerInferRecorder(self._log_dir)
        recorder.write(
            {
                "type": "session_open",
                "ts_utc": datetime.now(timezone.utc).isoformat(),
                "remote": remote,
                "log_file": str(recorder.jsonl_path),
            }
        )

        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        loop = asyncio.get_running_loop()
        try:
            while True:
                try:
                    start_time = time.monotonic()
                    obs = unpackb(await websocket.recv())
                    # 推理前先摘要：policy.infer 会原地把 numpy 改成 torch
                    obs_for_log = obs if isinstance(obs, dict) else {}
                    obs_summary = _obs_summary(obs_for_log)

                    infer_time = time.monotonic()
                    # 推理放到线程池，避免阻塞事件循环 → 客户端 keepalive ping 超时
                    action = await loop.run_in_executor(None, self._policy.infer, obs)
                    infer_time = time.monotonic() - infer_time

                    if not isinstance(action, dict):
                        action = {"action": action}

                    action["server_timing"] = {
                        "infer_ms": infer_time * 1000,
                    }
                    if prev_total_time is not None:
                        # We can only record the last total time since we also want to include the send time.
                        action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                    action_info = _action_summary(action)
                    recorder.step += 1
                    record = {
                        "type": "infer",
                        "step": recorder.step,
                        "ts_utc": datetime.now(timezone.utc).isoformat(),
                        "remote": remote,
                        "infer_ms": infer_time * 1000.0,
                        "obs": obs_summary,
                        "action": action_info,
                        "error": None,
                    }
                    recorder.write(record)
                    abs_max = action_info.get("chunk_abs_max")
                    logger.info(
                        "[infer #%d] %.1fms  state_abs_max=%s  action_abs_max=%s  task=%r",
                        recorder.step,
                        infer_time * 1000.0,
                        obs_summary.get("state_abs_max"),
                        abs_max,
                        obs_summary.get("task"),
                    )
                    if isinstance(abs_max, (int, float)) and abs_max < 0.03:
                        logger.warning(
                            "[infer #%d] 动作幅度很小 (abs_max=%.5f rad)。"
                            "若对应 ADC|Δ|<20，STM32 会死区不动作。",
                            recorder.step,
                            float(abs_max),
                        )

                    await websocket.send(packer.pack(action))
                    prev_total_time = time.monotonic() - start_time

                except websockets.ConnectionClosed:
                    logger.info(f"Connection from {remote} closed")
                    break
                except Exception:
                    err = traceback.format_exc()
                    try:
                        recorder.log_step(
                            remote=remote,
                            obs=obs if isinstance(obs, dict) else {},
                            action={},
                            infer_ms=-1.0,
                            error=err,
                        )
                    except Exception:  # noqa: BLE001
                        pass
                    await websocket.send(err)
                    await websocket.close(
                        code=websockets.frames.CloseCode.INTERNAL_ERROR,
                        reason="Internal server error. Traceback included in previous frame.",
                    )
                    raise
        finally:
            recorder.write(
                {
                    "type": "session_close",
                    "ts_utc": datetime.now(timezone.utc).isoformat(),
                    "remote": remote,
                    "steps": recorder.step,
                }
            )
            recorder.close()
            logger.info("Session log saved: %s", recorder.jsonl_path)


def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    # Continue with the normal request handling.
    return None
