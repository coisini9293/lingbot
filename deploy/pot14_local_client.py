#!/usr/bin/env python3
"""pot14 本地客户端：串口(或模拟) + WebSocket 云端推理。

功能：
1. 真实模式：读串口状态 / 相机 → 发到 AutoDL → 写回串口动作
2. 模拟模式：假串口输入输出，测服务器推理延迟并落盘
3. 终端打印关键 timing，JSONL 保存完整运行记录

示例（Mac）：
  python deploy/pot14_local_client.py --list-ports
  python deploy/pot14_local_client.py --mode mock-latency \\
    --ws-url https://u1087324-85uh-a5fac6ab.weste.seetacloud.com:8443
  python deploy/pot14_local_client.py --mode serial \\
    --serial-port /dev/cu.XXXX --ws-url wss://...:8443
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import functools
import glob
import math

import msgpack
import numpy as np
import websockets.sync.client

# ---------------------------------------------------------------------------
# msgpack + numpy（与服务端 deploy/msgpack_numpy 一致）
# ---------------------------------------------------------------------------


def pack_array(obj: Any) -> Any:
    if isinstance(obj, (np.ndarray, np.generic)) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")
    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }
    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }
    return obj


def unpack_array(obj: Any) -> Any:
    if isinstance(obj, dict) and b"__ndarray__" in obj:
        return np.ndarray(
            buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"]
        )
    if isinstance(obj, dict) and b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


Packer = functools.partial(msgpack.Packer, default=pack_array)
unpackb = functools.partial(msgpack.unpackb, object_hook=unpack_array)

logger = logging.getLogger("pot14_client")

# pot14：7 维（6 臂 + 1 effector），三路相机（与 robot_configs/pot14.yaml origin_keys 一致）
STATE_DIM = 7
IMAGE_KEYS = (
    "observation.images.top",
    "observation.images.left",
    "observation.images.right",
)
DEFAULT_IMAGE_HW = (240, 320)  # H, W — 与转换脚本默认一致；服务端还会 resize 到 224
DEFAULT_TASK = "完成任务：杯子"
DEFAULT_WS_URL = "wss://u1087324-85uh-a5fac6ab.weste.seetacloud.com:8443"
DEFAULT_ROBO_NAME = "pot14"
# 动作下发限速：过快会拉高电机电流；默认 5Hz，可用 --fps / 前端调节
DEFAULT_ACTION_FPS = 5.0
# 相邻两帧关节指令最大变化（弧度）；0 表示不限制
DEFAULT_MAX_JOINT_STEP = 0.0  # 0 = 不限制相邻帧关节步进

# pot14 串口：12-bit ADC → relative_rad（与采集 CSV 一致：counts * ±2π/4096）
# 固件（arm14_pot_receiver / STM32 UART7）合法 ADC 为 0000..4095
_POT14_ADC_COUNTS = 4096
_POT14_ADC_MAX_VALUE = 4095
_POT14_COUNTS_TO_RAD = 2.0 * math.pi / _POT14_ADC_COUNTS
# 右臂 P7..P13 → joint1..7；后两轴符号与数据集一致为负
_POT14_JOINT_SIGNS = np.array([1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0], dtype=np.float32)
_POT14_STATE_KEYS = tuple(f"P{i}" for i in range(7, 14))  # P7..P13
_POT14_ALL_KEYS = tuple(f"P{i}" for i in range(14))  # P0..P13
# ESP32 接收帧：P + 14*4digits + * + 4hex CRC16（不含换行共 62 字节）
_POT14_POT_COUNT = 14
_POT14_ADC_DIGITS = 4
_POT14_FRAME_DATA_LEN = 1 + _POT14_POT_COUNT * _POT14_ADC_DIGITS  # 57
_POT14_FRAME_LEN = _POT14_FRAME_DATA_LEN + 1 + 4  # 62


def parse_pot14_serial_line(line: str) -> dict[str, float]:
    """解析状态遥测行（key=value），如 RAW_ADC,P0=.. 或 b=2032,P6=..,P7=.."""
    out: dict[str, float] = {}
    for part in line.strip().split(","):
        part = part.strip()
        if not part or "=" not in part:
            continue
        key, raw = part.split("=", 1)
        key = key.strip()
        try:
            out[key] = float(raw.strip())
        except ValueError:
            continue
    return out


def crc16_ccitt_esp32(data: bytes) -> int:
    """与 arm14_pot_receiver / STM32 相同的 CRC16-CCITT（初值 0xFFFF，多项式 0x1021）。"""
    crc = 0xFFFF
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc


def build_esp32_pot_raw_frame(adc14: np.ndarray) -> str:
    """构造 ESP32/STM32 控制帧（含换行）。

    格式（见 arm14_pot_receiver.ino）：
      P + 14×4位十进制 ADC(0000..4095) + * + 4位十六进制 CRC16 + \\n
    P0..P6 左臂，P7..P13 右臂。接收端校验 CRC 后原样转发到 UART7。
    """
    adc = np.asarray(adc14, dtype=np.int32).reshape(-1)
    if adc.shape[0] != _POT14_POT_COUNT:
        raise ValueError(f"需要 {_POT14_POT_COUNT} 路 ADC，当前 {adc.shape[0]}")
    adc = np.clip(adc, 0, _POT14_ADC_MAX_VALUE)
    body = "P" + "".join(f"{int(v):04d}" for v in adc.tolist())
    if len(body) != _POT14_FRAME_DATA_LEN:
        raise RuntimeError(f"帧数据长度异常 {len(body)} != {_POT14_FRAME_DATA_LEN}")
    crc = crc16_ccitt_esp32(body.encode("ascii"))
    frame = f"{body}*{crc:04X}"
    if len(frame) != _POT14_FRAME_LEN:
        raise RuntimeError(f"整帧长度异常 {len(frame)} != {_POT14_FRAME_LEN}")
    return frame + "\n"


def adc_to_relative_rad(
    adc: np.ndarray,
    zero_adc: np.ndarray,
    *,
    signs: np.ndarray = _POT14_JOINT_SIGNS,
) -> np.ndarray:
    """raw ADC → 相对标定零点的弧度（训练用 observation.state）。"""
    adc = np.asarray(adc, dtype=np.float64).reshape(-1)
    zero = np.asarray(zero_adc, dtype=np.float64).reshape(-1)
    counts = adc - zero
    counts = (counts + _POT14_ADC_COUNTS / 2) % _POT14_ADC_COUNTS - _POT14_ADC_COUNTS / 2
    return (signs * _POT14_COUNTS_TO_RAD * counts).astype(np.float32)


def relative_rad_to_adc(
    rad: np.ndarray,
    zero_adc: np.ndarray,
    *,
    signs: np.ndarray = _POT14_JOINT_SIGNS,
) -> np.ndarray:
    """relative_rad → 目标 ADC（写回串口）。"""
    rad = np.asarray(rad, dtype=np.float64).reshape(-1)
    zero = np.asarray(zero_adc, dtype=np.float64).reshape(-1)
    counts = rad / (signs * _POT14_COUNTS_TO_RAD)
    adc = zero + counts
    adc = np.mod(adc, _POT14_ADC_COUNTS)
    return np.clip(np.rint(adc), 0, _POT14_ADC_MAX_VALUE).astype(np.int32)


def open_pot14_serial(port: str, baudrate: int = 115200, timeout: float = 1.0) -> Any:
    """打开串口；尽量关掉 DTR/RTS，减少 ESP32 因开串口自动复位。"""
    import serial  # type: ignore

    ser = serial.Serial()
    ser.port = port
    ser.baudrate = baudrate
    ser.timeout = timeout
    ser.dsrdtr = False
    ser.rtscts = False
    ser.dtr = False
    ser.rts = False
    ser.open()
    try:
        ser.reset_input_buffer()
        ser.reset_output_buffer()
    except Exception:  # noqa: BLE001
        pass
    return ser


def serial_set_control_mode(ser: Any, mode: int) -> None:
    """发送端固件：1=遥操作，2=模型（屏蔽电位器）。"""
    if mode not in (1, 2):
        raise ValueError("mode 只能是 1 或 2")
    payload = f"{mode}\n".encode("ascii")
    ser.write(payload)
    try:
        ser.flush()
    except Exception:  # noqa: BLE001
        pass
    logger.info("[SerialArm] 已请求控制模式 MODE=%d", mode)


def wait_sender_ready(ser: Any, timeout_s: float = 12.0) -> dict[str, Any]:
    """等发送端打出 RAW_ADC / STATUS（蓝牙重连后才会稳定控臂）。"""
    deadline = time.monotonic() + timeout_s
    saw_raw = False
    status_line = ""
    bt_on = False
    mode = None
    while time.monotonic() < deadline:
        line = ser.readline().decode("utf-8", errors="ignore").strip()
        if not line:
            continue
        if line.startswith("RAW_ADC") or (",P7=" in line and "P13=" in line):
            saw_raw = True
        if line.startswith("STATUS") or "BT=" in line or line.startswith("[MODE]"):
            status_line = line
            if "BT=ON" in line:
                bt_on = True
            if "MODE=2" in line or "MODE] 2" in line:
                mode = 2
            elif "MODE=1" in line or "MODE] 1" in line:
                mode = 1
        if saw_raw and bt_on:
            return {
                "ready": True,
                "bt": True,
                "mode": mode,
                "status": status_line,
            }
    return {
        "ready": saw_raw,
        "bt": bt_on,
        "mode": mode,
        "status": status_line,
        "hint": "未看到 BT=ON：开串口可能触发了 ESP32 复位，请再等几秒或断电重插后重试",
    }

from camera_avf import (  # noqa: E402
    list_avf_devices,
    open_avf_capture,
    suggest_camera_indices,
)

# 兼容旧调用名
list_usb_cameras = list_avf_devices


def normalize_ws_url(url: str) -> str:
    uri = url.strip().rstrip("/")
    if uri.startswith("https://"):
        return "wss://" + uri[len("https://") :]
    if uri.startswith("http://"):
        return "ws://" + uri[len("http://") :]
    if uri.startswith("ws://") or uri.startswith("wss://"):
        return uri
    return "wss://" + uri


def list_serial_ports() -> list[str]:
    ports = sorted(
        set(
            glob.glob("/dev/cu.*")
            + glob.glob("/dev/tty.*")
            + glob.glob("/dev/ttyACM*")
            + glob.glob("/dev/ttyUSB*")
            + glob.glob("/dev/tty.usb*")
        )
    )

    def _rank(p: str) -> tuple[int, str]:
        name = p.rsplit("/", 1)[-1].lower()
        if "bluetooth" in name or "incoming" in name or "debug-console" in name:
            return (2, p)
        # CH340 等 USB 串口优先
        if "usbserial" in name or "wchusbserial" in name or "usbmodem" in name:
            return (0, p)
        if p.startswith("/dev/cu."):
            return (1, p)
        return (2, p)

    return sorted(ports, key=_rank)


def limit_action_step(
    action: np.ndarray,
    prev: np.ndarray | None,
    max_joint_step: float,
) -> np.ndarray:
    """限制相邻动作每轴变化量，降低电流冲击。max_joint_step<=0 表示不限制。"""
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    if max_joint_step <= 0 or prev is None:
        return action
    prev_arr = np.asarray(prev, dtype=np.float32).reshape(-1)
    if prev_arr.shape != action.shape:
        return action
    delta = np.clip(action - prev_arr, -max_joint_step, max_joint_step)
    return (prev_arr + delta).astype(np.float32)


def build_pot14_obs(
    *,
    images: dict[str, np.ndarray],
    state: np.ndarray,
    task: str,
) -> dict[str, Any]:
    """组装服务端期望的 observation（origin_keys 命名）。"""
    state = np.asarray(state, dtype=np.float32).reshape(-1)
    if state.shape[0] != STATE_DIM:
        raise ValueError(f"state 必须是 ({STATE_DIM},)，当前 {state.shape}")

    obs: dict[str, Any] = {"observation.state": state, "task": task}
    for key in IMAGE_KEYS:
        if key not in images:
            raise KeyError(f"缺少图像键 {key}")
        img = images[key]
        if img.dtype != np.uint8 or img.ndim != 3 or img.shape[-1] != 3:
            raise ValueError(f"{key} 需 uint8 HWC RGB，当前 dtype={img.dtype} shape={img.shape}")
        obs[key] = img
    return obs


def flatten_action_chunk(action_chunk: dict[str, Any]) -> np.ndarray:
    """把服务端返回的 action 特征拼成 (T, 7)。"""
    if action_chunk.get("action") is None and "action" in action_chunk:
        raise RuntimeError("服务端返回 action=None（通常是 reset 响应）")

    if "action" in action_chunk and action_chunk["action"] is not None:
        arr = np.asarray(action_chunk["action"], dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[-1] != STATE_DIM:
            raise ValueError(f"action 末维应为 {STATE_DIM}，当前 {arr.shape}")
        return arr

    # 分特征返回时拼起来
    parts = []
    for key in ("action.arm.position", "action.effector.position"):
        if key in action_chunk:
            parts.append(np.asarray(action_chunk[key], dtype=np.float32))
    if not parts:
        raise KeyError(f"无法解析动作键，收到: {list(action_chunk.keys())}")
    arr = np.concatenate(parts, axis=-1)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[-1] != STATE_DIM:
        raise ValueError(f"拼接后动作维应为 {STATE_DIM}，当前 {arr.shape}")
    return arr


class WebsocketClientPolicy:
    def __init__(
        self,
        uri: str,
        api_key: Optional[str] = None,
        connect_timeout: float = 30.0,
        *,
        insecure_ssl: bool = True,
    ) -> None:
        self._uri = normalize_ws_url(uri)
        self._api_key = api_key
        self._insecure_ssl = insecure_ssl
        self._packer = Packer()
        self._ws, self._server_metadata = self._wait_for_server(connect_timeout)

    @property
    def uri(self) -> str:
        return self._uri

    def get_server_metadata(self) -> Dict:
        return self._server_metadata

    def _ssl_arg(self) -> Any:
        """AutoDL seetacloud 常用自签证书，默认不校验。"""
        if not self._uri.startswith("wss://"):
            return None
        if not self._insecure_ssl:
            return True  # 系统默认校验证书
        import ssl

        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return ctx

    def _wait_for_server(
        self, connect_timeout: float
    ) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
        logger.info("连接 WebSocket: %s (insecure_ssl=%s)", self._uri, self._insecure_ssl)
        if self._insecure_ssl and self._uri.startswith("wss://"):
            logger.warning(
                "已关闭 SSL 证书校验（AutoDL 代理常见自签证书）。仅用于个人调试。"
            )
        deadline = time.monotonic() + connect_timeout
        last_err: Exception | None = None
        while time.monotonic() < deadline:
            try:
                headers = (
                    {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
                )
                conn = websockets.sync.client.connect(
                    self._uri,
                    compression=None,
                    max_size=None,
                    additional_headers=headers,
                    open_timeout=10,
                    ssl=self._ssl_arg(),
                    # 首次推理可能 >30s；禁用客户端 keepalive，避免 ping 超时断线
                    ping_interval=None,
                    close_timeout=10,
                )
                metadata = unpackb(conn.recv())
                logger.info(
                    "已连接，server metadata keys=%s",
                    list(metadata) if isinstance(metadata, dict) else type(metadata),
                )
                return conn, metadata
            except Exception as exc:  # noqa: BLE001 — 启动阶段需重试
                last_err = exc
                logger.warning("等待服务器... (%s)", exc)
                time.sleep(2)
        raise ConnectionError(f"无法在 {connect_timeout:.0f}s 内连接 {self._uri}: {last_err}")

    def infer(self, obs: Dict) -> Dict:
        import websockets.exceptions as ws_exc

        t0 = time.perf_counter()
        last_err: Exception | None = None
        response: Any = None
        for attempt in range(3):
            try:
                self._ws.send(self._packer.pack(obs))
                response = self._ws.recv()
                break
            except ws_exc.ConnectionClosed as exc:
                last_err = exc
                logger.warning(
                    "WebSocket 断线 (%s)，第 %d/3 次重连后重试推理…",
                    exc,
                    attempt + 1,
                )
                try:
                    self._ws.close()
                except Exception:  # noqa: BLE001
                    pass
                self._ws, self._server_metadata = self._wait_for_server(30.0)
                try:
                    self._ws.send(
                        self._packer.pack({"reset": True, "robo_name": DEFAULT_ROBO_NAME})
                    )
                    _ = self._ws.recv()
                except Exception as reset_exc:  # noqa: BLE001
                    logger.warning("重连后 reset 失败: %s", reset_exc)
        else:
            raise ConnectionError(f"推理失败，WebSocket 多次断线: {last_err}") from last_err

        rtt_ms = (time.perf_counter() - t0) * 1000.0
        if isinstance(response, str):
            raise RuntimeError(f"推理服务器错误:\n{response}")
        out = unpackb(response)
        out["_client_rtt_ms"] = rtt_ms
        return out

    def reset(self, robo_name: str = DEFAULT_ROBO_NAME) -> Dict:
        logger.info("发送 reset robo_name=%s", robo_name)
        return self.infer({"reset": True, "robo_name": robo_name})

    def close(self) -> None:
        try:
            self._ws.close()
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------------------
# 串口抽象：真实 / 模拟
# ---------------------------------------------------------------------------


class ArmIO(ABC):
    """机械臂输入输出接口。"""

    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def disconnect(self) -> None: ...

    @abstractmethod
    def read_state(self) -> np.ndarray:
        """返回 (7,) float32 relative_rad。"""

    @abstractmethod
    def write_action(self, action: np.ndarray) -> None:
        """写入一步 (7,) 动作。"""

    @abstractmethod
    def read_images(self) -> dict[str, np.ndarray]:
        """返回三路 uint8 RGB。"""


class MockArmIO(ArmIO):
    """模拟串口 + 假图像，用于测延迟、不碰真机。"""

    def __init__(self, image_hw: tuple[int, int] = DEFAULT_IMAGE_HW, seed: int = 0) -> None:
        self._rng = np.random.default_rng(seed)
        self._h, self._w = image_hw
        self._state = np.zeros(STATE_DIM, dtype=np.float32)
        self._connected = False
        self.read_count = 0
        self.write_count = 0
        self.last_action: np.ndarray | None = None

    def connect(self) -> None:
        self._connected = True
        logger.info("[MockArm] 已连接（模拟串口） image=%sx%s", self._w, self._h)

    def disconnect(self) -> None:
        self._connected = False
        logger.info(
            "[MockArm] 断开  reads=%d writes=%d", self.read_count, self.write_count
        )

    def read_state(self) -> np.ndarray:
        if not self._connected:
            raise RuntimeError("MockArm 未连接")
        # 轻微扰动，模拟关节反馈
        self._state = self._state + self._rng.normal(0, 0.002, size=STATE_DIM).astype(
            np.float32
        )
        self.read_count += 1
        return self._state.copy()

    def write_action(self, action: np.ndarray) -> None:
        if not self._connected:
            raise RuntimeError("MockArm 未连接")
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] != STATE_DIM:
            raise ValueError(f"动作维必须 {STATE_DIM}，当前 {action.shape}")
        self.last_action = action.copy()
        self._state = action.copy()  # 开环假执行：目标即新状态
        self.write_count += 1

    def read_images(self) -> dict[str, np.ndarray]:
        imgs = {}
        for key in IMAGE_KEYS:
            imgs[key] = self._rng.integers(
                0, 256, size=(self._h, self._w, 3), dtype=np.uint8
            )
        return imgs


class SerialArmIO(ArmIO):
    """真实串口 + AVFoundation 相机（pot14 / ESP32 14 路电位器协议）。

    读（状态遥测，key=value，来自 USB 调试口等）:
      RAW_ADC,P0=..,P13=..  或  b=2032,P6=1521,P7=2171,...,P13=2583
      右臂用 P7..P13 → relative_rad（与训练 CSV 一致）。

    写（控制命令，必须与 arm14_pot_receiver 一致）:
      P + 14×4位ADC + * + CRC16四位十六进制 + \\n
      例: P1521...2583*EEE1\\n
      接收端校验后转发 UART7；>300ms 无合法帧会发 STOP。
      P0..P6 保持最近观测（左臂不动），P7..P13 由模型动作换算。

    连接后用前几帧标定右臂零点（请尽量与采集时一样先回零位）。
    mock_state=True 时不读串口状态。
    """

    def __init__(
        self,
        port: str,
        baudrate: int = 115200,
        camera_indices: tuple[int, int, int] = (0, 1, 2),
        image_hw: tuple[int, int] = DEFAULT_IMAGE_HW,
        *,
        mock_state: bool = False,
    ) -> None:
        self.port = port
        self.baudrate = baudrate
        self.camera_indices = camera_indices
        self._h, self._w = image_hw
        self.mock_state = mock_state
        self._ser = None
        self._caps: list[Any] = []
        self._last_state = np.zeros(STATE_DIM, dtype=np.float32)
        self._zero_adc: np.ndarray | None = None
        # 14 路 hold：写控制帧时左臂用观测值保持，右臂由动作更新
        self._hold_adc14 = np.full(_POT14_POT_COUNT, 2048, dtype=np.int32)
        self._write_count = 0

    def connect(self) -> None:
        try:
            import serial  # type: ignore  # noqa: F401
        except ImportError as exc:
            raise ImportError("真实串口需要 pyserial：pip install pyserial") from exc

        self._ser = open_pot14_serial(self.port, self.baudrate, timeout=1.0)
        self._caps = []
        for idx in self.camera_indices:
            try:
                cap = open_avf_capture(int(idx), width=640, height=480, fps=15)
                self._caps.append(cap)
                logger.info("[SerialArm] 相机 avf_index=%s OK", idx)
            except Exception as exc:  # noqa: BLE001
                logger.warning("相机 avf_index=%s 打不开: %s（黑图占位）", idx, exc)
                self._caps.append(None)

        logger.info("[SerialArm] 等待发送端 BT…")
        ready = wait_sender_ready(self._ser, timeout_s=15.0)
        logger.info("[SerialArm] sender ready=%s", ready)
        # 进入模型模式：完全屏蔽电位器遥操作，避免推理等待期间被手摇抢回
        serial_set_control_mode(self._ser, 2)
        time.sleep(0.2)
        ready2 = wait_sender_ready(self._ser, timeout_s=5.0)
        logger.info("[SerialArm] after MODE=2: %s", ready2)

        if not self.mock_state:
            self._calibrate_zero()

        logger.info(
            "[SerialArm] 已打开 %s @ %d  mock_state=%s zero=%s hold14=%s MODE=2",
            self.port,
            self.baudrate,
            self.mock_state,
            None if self._zero_adc is None else self._zero_adc.tolist(),
            self._hold_adc14.tolist(),
        )

    def _update_hold_from_fields(self, fields: dict[str, float]) -> None:
        for i, key in enumerate(_POT14_ALL_KEYS):
            if key in fields:
                self._hold_adc14[i] = int(
                    np.clip(round(fields[key]), 0, _POT14_ADC_MAX_VALUE)
                )

    def _calibrate_zero(self, samples: int = 5) -> None:
        """读取若干帧 ADC，取平均作为相对零点（对齐采集时 relative_counts=0）。"""
        assert self._ser is not None
        self._ser.reset_input_buffer()
        buf: list[np.ndarray] = []
        deadline = time.monotonic() + 5.0
        while len(buf) < samples and time.monotonic() < deadline:
            line = self._ser.readline().decode("utf-8", errors="ignore").strip()
            if not line:
                continue
            try:
                adc, fields = self._line_to_adc_and_fields(line)
            except ValueError:
                logger.debug("标定忽略行: %r", line[:80])
                continue
            self._update_hold_from_fields(fields)
            buf.append(adc)
        if not buf:
            raise TimeoutError(
                "串口标定失败：5 秒内未读到含 P7..P13 的状态行。"
                "请确认固件在发 RAW_ADC/P7=.../P13=..."
            )
        self._zero_adc = np.mean(np.stack(buf, axis=0), axis=0)
        # 右臂 hold 与零点对齐，避免首帧乱跳
        self._hold_adc14[7:14] = np.clip(
            np.rint(self._zero_adc), 0, _POT14_ADC_MAX_VALUE
        ).astype(np.int32)
        logger.info(
            "[SerialArm] 零点 ADC 标定完成（%d 帧）: %s",
            len(buf),
            self._zero_adc.round(1),
        )

    def _line_to_adc_and_fields(self, line: str) -> tuple[np.ndarray, dict[str, float]]:
        # 兼容 JSON（调试）与 pot14 固件 key=value
        if line.startswith("{"):
            data = json.loads(line)
            if "state" in data:
                raise ValueError("JSON state 请用 mock 或单独路径")
            if "adc" in data:
                adc = np.asarray(data["adc"], dtype=np.float64).reshape(-1)
                if adc.shape[0] != STATE_DIM:
                    raise ValueError(f"adc 维错误 {adc.shape}")
                fields = {k: float(v) for k, v in zip(_POT14_STATE_KEYS, adc.tolist())}
                return adc, fields
        fields = parse_pot14_serial_line(line)
        missing = [k for k in _POT14_STATE_KEYS if k not in fields]
        if missing:
            raise ValueError(f"串口行缺字段 {missing}: {line[:120]!r}")
        adc = np.array([fields[k] for k in _POT14_STATE_KEYS], dtype=np.float64)
        return adc, fields

    def disconnect(self) -> None:
        if self._ser is not None:
            try:
                serial_set_control_mode(self._ser, 1)
                time.sleep(0.15)
            except Exception:  # noqa: BLE001
                logger.warning("[SerialArm] 切回 MODE=1 失败（可手动串口发 1）")
        for cap in self._caps:
            try:
                if cap is not None:
                    cap.release()
            except Exception:  # noqa: BLE001
                pass
        if self._ser is not None:
            self._ser.close()
        logger.info("[SerialArm] 已断开 writes=%d（已请求恢复遥操作 MODE=1）", self._write_count)

    def _recv_state_frame(self) -> np.ndarray:
        assert self._ser is not None
        if self._zero_adc is None:
            raise RuntimeError("尚未标定零点 ADC")
        # 多读几行，丢掉半包
        last_err: Exception | None = None
        for _ in range(8):
            line = self._ser.readline().decode("utf-8", errors="ignore").strip()
            if not line:
                last_err = TimeoutError(
                    "串口 1 秒内无数据。确认固件持续发送 P7..P13 状态行。"
                )
                continue
            try:
                adc, fields = self._line_to_adc_and_fields(line)
                self._update_hold_from_fields(fields)
                return adc_to_relative_rad(adc, self._zero_adc)
            except ValueError as exc:
                last_err = exc
                continue
        raise TimeoutError(str(last_err) if last_err else "串口读状态失败")

    def _send_action_frame(self, action: np.ndarray) -> None:
        """下发 ESP32 RAW 控制帧（14 路 + CRC），非 key=value。"""
        assert self._ser is not None
        if self._zero_adc is None:
            payload = json.dumps({"action": [float(x) for x in action.tolist()]}) + "\n"
            self._ser.write(payload.encode("utf-8"))
            return
        right_adc = relative_rad_to_adc(action, self._zero_adc)
        self._hold_adc14[7:14] = right_adc
        # STM32 ARM_POT_DEAD_ADC=20：相对零点 |Δ|<20 的指令会被当成不动
        delta = right_adc.astype(np.float64) - self._zero_adc.astype(np.float64)
        delta = (delta + _POT14_ADC_COUNTS / 2) % _POT14_ADC_COUNTS - _POT14_ADC_COUNTS / 2
        max_abs = float(np.max(np.abs(delta)))
        payload = build_esp32_pot_raw_frame(self._hold_adc14)
        self._ser.write(payload.encode("ascii"))
        try:
            self._ser.flush()
        except Exception:  # noqa: BLE001
            pass
        self._write_count += 1
        if self._write_count <= 3 or self._write_count % 50 == 0:
            logger.info(
                "[SerialArm] 写控制帧 #%d len=%d max|Δadc|=%.1f payload=%r",
                self._write_count,
                len(payload.strip()),
                max_abs,
                payload.strip()[:70],
            )
            if max_abs < 20:
                logger.warning(
                    "[SerialArm] 指令在 STM32 死区(|Δ|<20)内，机械臂不会明显动作。"
                    "这是模型输出太靠近零位，不是帧格式错误。"
                )

    def write_raw_adc14(self, adc14: np.ndarray) -> None:
        """直接写 14 路目标 ADC（硬件点动自检用）。"""
        assert self._ser is not None
        self._hold_adc14 = np.clip(
            np.asarray(adc14, dtype=np.int32).reshape(_POT14_POT_COUNT),
            0,
            _POT14_ADC_MAX_VALUE,
        )
        payload = build_esp32_pot_raw_frame(self._hold_adc14)
        self._ser.write(payload.encode("ascii"))
        try:
            self._ser.flush()
        except Exception:  # noqa: BLE001
            pass
        self._write_count += 1

    def read_state(self) -> np.ndarray:
        if self.mock_state:
            return self._last_state.copy()
        self._last_state = self._recv_state_frame()
        return self._last_state.copy()

    def write_action(self, action: np.ndarray) -> None:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] != STATE_DIM:
            raise ValueError(f"动作维必须 {STATE_DIM}，当前 {action.shape}")
        self._send_action_frame(action)
        if self.mock_state:
            self._last_state = action.copy()

    def read_images(self) -> dict[str, np.ndarray]:
        import cv2  # type: ignore

        out: dict[str, np.ndarray] = {}
        for key, cap in zip(IMAGE_KEYS, self._caps):
            if cap is None or not cap.isOpened():
                out[key] = np.zeros((self._h, self._w, 3), dtype=np.uint8)
                continue
            ok, frame = cap.read()
            if not ok or frame is None:
                out[key] = np.zeros((self._h, self._w, 3), dtype=np.uint8)
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (self._w, self._h))
            out[key] = frame.astype(np.uint8)
        return out


def hardware_jog_test(
    *,
    port: str,
    baudrate: int = 115200,
    joint_index: int = 0,
    peak_delta_adc: int = 160,
    step_adc: int = 40,
    hold_seconds: float = 1.5,
    fps: float = 20.0,
) -> dict[str, Any]:
    """不走模型：对右臂某一轴做缓升/缓降点动，验证串口是否真能控臂。

    joint_index: 0..6 对应 P7..P13。
    步进默认 40（< STM32 MAX_STEP 64），峰值默认 160（> 死区 20）。
    """
    if joint_index < 0 or joint_index >= STATE_DIM:
        raise ValueError(f"joint_index 需在 0..{STATE_DIM - 1}")
    if step_adc <= 0 or step_adc > 64:
        raise ValueError("step_adc 建议 1..64（STM32 单帧最大步进 64）")

    arm = SerialArmIO(port=port, baudrate=baudrate, camera_indices=(0, 1, 2), mock_state=False)
    arm._ser = open_pot14_serial(port, baudrate, timeout=1.0)
    arm._caps = [None, None, None]

    logger.info("[jog] 等待发送端就绪（蓝牙重连可能需要数秒）…")
    ready = wait_sender_ready(arm._ser, timeout_s=15.0)
    logger.info("[jog] sender ready=%s bt=%s status=%r", ready.get("ready"), ready.get("bt"), ready.get("status"))
    if not ready.get("bt"):
        time.sleep(2.0)
        ready = wait_sender_ready(arm._ser, timeout_s=15.0)
        logger.info("[jog] 二次等待 bt=%s status=%r", ready.get("bt"), ready.get("status"))

    serial_set_control_mode(arm._ser, 2)
    time.sleep(0.25)
    ready = wait_sender_ready(arm._ser, timeout_s=5.0)
    logger.info("[jog] MODE=2 ready=%s", ready)

    arm._calibrate_zero()
    assert arm._zero_adc is not None
    base = arm._hold_adc14.copy()
    axis = 7 + joint_index
    period = 1.0 / max(fps, 1.0)
    sent = 0
    try:
        offset = 0
        while offset < peak_delta_adc:
            offset = min(offset + step_adc, peak_delta_adc)
            target = base.copy()
            target[axis] = int(
                np.clip(int(base[axis]) + offset, 0, _POT14_ADC_MAX_VALUE)
            )
            arm.write_raw_adc14(target)
            sent += 1
            time.sleep(period)
        t_end = time.monotonic() + hold_seconds
        while time.monotonic() < t_end:
            target = base.copy()
            target[axis] = int(
                np.clip(int(base[axis]) + peak_delta_adc, 0, _POT14_ADC_MAX_VALUE)
            )
            arm.write_raw_adc14(target)
            sent += 1
            time.sleep(period)
        offset = peak_delta_adc
        while offset > 0:
            offset = max(offset - step_adc, 0)
            target = base.copy()
            target[axis] = int(
                np.clip(int(base[axis]) + offset, 0, _POT14_ADC_MAX_VALUE)
            )
            arm.write_raw_adc14(target)
            sent += 1
            time.sleep(period)
        logger.info(
            "[jog] 完成 joint=P%d peakΔ=%d frames=%d bt_seen=%s",
            axis,
            peak_delta_adc,
            sent,
            ready.get("bt"),
        )
        return {
            "ok": True,
            "joint": f"P{axis}",
            "peak_delta_adc": peak_delta_adc,
            "frames": sent,
            "zero": arm._zero_adc.tolist(),
            "bt_ready": bool(ready.get("bt")),
            "mode": ready.get("mode"),
            "sender_status": ready.get("status", ""),
            "hint": "点动在 MODE=2 下执行。结束后会自动切回 MODE=1 遥操作。",
        }
    finally:
        try:
            if arm._ser is not None:
                serial_set_control_mode(arm._ser, 1)
                time.sleep(0.1)
                arm._ser.close()
        except Exception:  # noqa: BLE001
            pass
        logger.info("[jog] 串口已关闭（已请求 MODE=1）")


# ---------------------------------------------------------------------------
# 运行记录
# ---------------------------------------------------------------------------


@dataclass
class StepRecord:
    step: int
    client_rtt_ms: float
    server_infer_ms: float | None
    server_prev_total_ms: float | None
    action_shape: list[int]
    state: list[float]
    first_action: list[float]
    ok: bool
    error: str | None = None


class RunRecorder:
    def __init__(self, log_dir: Path, run_name: str) -> None:
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.run_id = f"{run_name}_{stamp}"
        self.jsonl_path = self.log_dir / f"{self.run_id}.jsonl"
        self.summary_path = self.log_dir / f"{self.run_id}_summary.json"
        self.records: list[StepRecord] = []
        meta = {
            "run_id": self.run_id,
            "started_at": stamp,
            "pid": os.getpid(),
        }
        with self.jsonl_path.open("w", encoding="utf-8") as f:
            f.write(json.dumps({"type": "meta", **meta}, ensure_ascii=False) + "\n")
        logger.info("运行记录: %s", self.jsonl_path)

    def add(self, record: StepRecord) -> None:
        self.records.append(record)
        with self.jsonl_path.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps({"type": "step", **asdict(record)}, ensure_ascii=False) + "\n"
            )

    def finalize(self, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        rtts = [r.client_rtt_ms for r in self.records if r.ok]
        infers = [
            r.server_infer_ms
            for r in self.records
            if r.ok and r.server_infer_ms is not None
        ]
        summary: dict[str, Any] = {
            "run_id": self.run_id,
            "num_steps": len(self.records),
            "num_ok": sum(1 for r in self.records if r.ok),
            "client_rtt_ms": _stats(rtts),
            "server_infer_ms": _stats(infers),
            "jsonl": str(self.jsonl_path),
        }
        if extra:
            summary.update(extra)
        self.summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info("汇总已写: %s", self.summary_path)
        return summary


def _stats(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": float(len(arr)),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
    }


def print_step_line(step: int, total: int, record: StepRecord) -> None:
    infer = (
        f"{record.server_infer_ms:.1f}ms"
        if record.server_infer_ms is not None
        else "n/a"
    )
    status = "OK" if record.ok else f"FAIL:{record.error}"
    print(
        f"[{step + 1}/{total}] rtt={record.client_rtt_ms:.1f}ms  "
        f"server_infer={infer}  action={record.action_shape}  {status}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def run_loop(
    *,
    arm: ArmIO,
    client: WebsocketClientPolicy,
    recorder: RunRecorder,
    task: str,
    steps: int,
    execute_chunk: bool,
    fps: float,
    do_reset: bool,
    max_joint_step: float = DEFAULT_MAX_JOINT_STEP,
) -> dict[str, Any]:
    arm.connect()
    try:
        if do_reset:
            reset_resp = client.reset(DEFAULT_ROBO_NAME)
            logger.info("reset 响应 keys=%s", list(reset_resp.keys()))

        last_cmd: np.ndarray | None = None
        for step in range(steps):
            try:
                state = arm.read_state()
                images = arm.read_images()
                obs = build_pot14_obs(images=images, state=state, task=task)
                t_wall0 = time.perf_counter()
                action_chunk = client.infer(obs)
                rtt_ms = float(action_chunk.get("_client_rtt_ms", (time.perf_counter() - t_wall0) * 1000))
                timing = action_chunk.get("server_timing") or {}
                actions = flatten_action_chunk(action_chunk)
                if execute_chunk:
                    for action in actions:
                        limited = limit_action_step(action, last_cmd, max_joint_step)
                        arm.write_action(limited)
                        last_cmd = limited
                        if fps > 0:
                            time.sleep(1.0 / fps)
                else:
                    limited = limit_action_step(actions[0], last_cmd, max_joint_step)
                    arm.write_action(limited)
                    last_cmd = limited

                record = StepRecord(
                    step=step,
                    client_rtt_ms=rtt_ms,
                    server_infer_ms=_as_float_or_none(timing.get("infer_ms")),
                    server_prev_total_ms=_as_float_or_none(timing.get("prev_total_ms")),
                    action_shape=list(actions.shape),
                    state=[float(x) for x in state.tolist()],
                    first_action=[float(x) for x in actions[0].tolist()],
                    ok=True,
                )
            except Exception as exc:  # noqa: BLE001 — 单步失败记日志后继续/或中断
                logger.exception("step %s 失败", step)
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
            print_step_line(step, steps, record)
            if not record.ok:
                break
    finally:
        arm.disconnect()
        client.close()

    summary = recorder.finalize(
        extra={
            "task": task,
            "ws_uri": client.uri,
            "arm": type(arm).__name__,
            "action_fps": fps,
            "max_joint_step": max_joint_step,
        }
    )
    print("\n=== 延迟汇总 ===", flush=True)
    print(json.dumps(summary.get("client_rtt_ms"), ensure_ascii=False, indent=2), flush=True)
    print("server_infer_ms:", json.dumps(summary.get("server_infer_ms"), ensure_ascii=False, indent=2), flush=True)
    print(f"记录文件: {recorder.jsonl_path}", flush=True)
    return summary


def _as_float_or_none(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def preview_cameras(
    indices: tuple[int, int, int], image_hw: tuple[int, int] = DEFAULT_IMAGE_HW
) -> int:
    """弹出三路预览（AVFoundation 序号），按 q 退出。"""
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise ImportError("需要 opencv-python：pip install opencv-python") from exc

    h, w = image_hw
    caps = []
    titles = ["top", "left", "right"]
    print(f"预览 avf_indices={indices}，按 q 退出", flush=True)
    for idx, title in zip(indices, titles):
        try:
            cap = open_avf_capture(int(idx), width=640, height=480)
            print(f"  [OK] avf={idx} -> {title}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  [!] avf={idx} ({title}) 打不开: {exc}", flush=True)
            cap = None
        caps.append((title, cap))

    try:
        while True:
            for title, cap in caps:
                if cap is None or not cap.isOpened():
                    frame = np.zeros((h, w, 3), dtype=np.uint8)
                else:
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        frame = np.zeros((h, w, 3), dtype=np.uint8)
                    else:
                        frame = cv2.resize(frame, (w, h))
                cv2.imshow(f"pot14:{title}", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    finally:
        for _, cap in caps:
            if cap is not None:
                cap.release()
        cv2.destroyAllWindows()
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="pot14 本地串口/模拟 + WebSocket 推理客户端")
    p.add_argument("--list-ports", action="store_true", help="列出本机串口后退出")
    p.add_argument(
        "--list-cameras",
        action="store_true",
        help="列出本机 USB 摄像头 OpenCV index 后退出",
    )
    p.add_argument(
        "--preview-cameras",
        action="store_true",
        help="按 --camera-indices 打开三路预览窗口（按 q 退出），用于确认 top/left/right 顺序",
    )
    p.add_argument(
        "--mode",
        choices=["menu", "mock-latency", "serial"],
        default="menu",
        help="menu=交互菜单（默认）；mock-latency=测试；serial=实机",
    )
    p.add_argument(
        "--ws-url",
        default=os.environ.get("VLA_WS_URL", DEFAULT_WS_URL),
        help="AutoDL 公网地址（https/wss 均可）",
    )
    p.add_argument("--serial-port", default=os.environ.get("ROBOT_SERIAL_PORT", ""))
    p.add_argument("--baudrate", type=int, default=115200)
    p.add_argument("--camera-indices", type=int, nargs=3, default=[0, 1, 2])
    p.add_argument("--task", default=DEFAULT_TASK)
    p.add_argument("--steps", type=int, default=20, help="推理轮数")
    p.add_argument("--fps", type=float, default=DEFAULT_ACTION_FPS, help="动作下发频率 Hz（默认 5，过快易过流）")
    p.add_argument(
        "--max-joint-step",
        type=float,
        default=DEFAULT_MAX_JOINT_STEP,
        help="相邻指令每轴最大变化（弧度），0=不限制",
    )
    p.add_argument(
        "--execute-chunk",
        action="store_true",
        help="把整段 action chunk 逐步写出（默认延迟测试只写第 1 步）",
    )
    p.add_argument("--no-reset", action="store_true", help="跳过首次 reset")
    p.add_argument(
        "--log-dir",
        default=str(Path(__file__).resolve().parent.parent / "logs" / "pot14_runs"),
        help="运行记录目录",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--connect-timeout", type=float, default=60.0)
    p.add_argument(
        "--insecure-ssl",
        action="store_true",
        default=True,
        help="跳过 SSL 证书校验（AutoDL 自签证书默认开启）",
    )
    p.add_argument(
        "--secure-ssl",
        action="store_true",
        help="强制校验证书（覆盖 --insecure-ssl）",
    )
    return p.parse_args()


def _prompt(text: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    raw = input(f"{text}{suffix}: ").strip()
    return raw if raw else default


def _prompt_int(text: str, default: int) -> int:
    raw = _prompt(text, str(default))
    try:
        return int(raw)
    except ValueError:
        print(f"  输入无效，使用默认 {default}")
        return default


def print_ports() -> None:
    ports = list_serial_ports()
    print("本机串口设备：")
    if not ports:
        print("  （未找到）")
    for port in ports:
        print(f"  {port}")
    print("Mac 请优先使用 /dev/cu.*")


def print_cameras() -> None:
    cams = list_usb_cameras(exclude_builtin=True)
    print("本机摄像头（AVFoundation，已排除 FaceTime）：")
    if not cams:
        print("  （未找到；需 pyobjc-framework-AVFoundation，并允许终端访问相机）")
    for cam in cams:
        name = cam.get("name") or "(无名)"
        uid = str(cam.get("unique_id") or "")[-8:]
        hint = cam.get("role_hint") or ""
        print(
            f"  avf={cam['index']}  {name}  uid…{uid}"
            + (f"  ← 默认建议 {hint}" if hint else "")
        )
    suggested = suggest_camera_indices(cams)
    print(f"建议 --camera-indices {' '.join(str(i) for i in suggested)}")
    print("前端可自行把 top/left/right 改成任意 avf 组合")


def run_once(args: argparse.Namespace, mode: str) -> int:
    """执行一次测试或实机推理。"""
    if mode == "serial" and not args.serial_port:
        print("实机模式必须指定串口（菜单里会询问，或 --serial-port）", file=sys.stderr)
        return 2

    print("\n=== pot14 本地客户端 ===", flush=True)
    print(f"mode     : {mode}", flush=True)
    print(f"ws-url   : {normalize_ws_url(args.ws_url)}", flush=True)
    print(f"task     : {args.task}", flush=True)
    print(f"steps    : {args.steps}", flush=True)
    print(f"action_fps: {args.fps}  max_joint_step: {args.max_joint_step}", flush=True)
    if mode == "serial":
        print(f"serial   : {args.serial_port}", flush=True)
        print(f"cameras  : {tuple(args.camera_indices)} (top,left,right)", flush=True)

    if mode == "mock-latency":
        arm: ArmIO = MockArmIO(seed=args.seed)
        run_name = "mock_latency"
        execute_chunk = bool(args.execute_chunk)
    else:
        arm = SerialArmIO(
            port=args.serial_port,
            baudrate=args.baudrate,
            camera_indices=tuple(args.camera_indices),
        )
        run_name = "serial"
        execute_chunk = True

    client = WebsocketClientPolicy(
        uri=args.ws_url,
        connect_timeout=args.connect_timeout,
        insecure_ssl=not args.secure_ssl,
    )
    recorder = RunRecorder(Path(args.log_dir), run_name=run_name)
    try:
        run_loop(
            arm=arm,
            client=client,
            recorder=recorder,
            task=args.task,
            steps=args.steps,
            execute_chunk=execute_chunk,
            fps=args.fps,
            do_reset=not args.no_reset,
            max_joint_step=args.max_joint_step,
        )
    except KeyboardInterrupt:
        print("\n已中断当前运行（Ctrl+C）", flush=True)
        try:
            client.close()
        except Exception:  # noqa: BLE001
            pass
        return 130
    return 0


def interactive_menu(args: argparse.Namespace) -> int:
    """交互菜单：可选测试/实机，跑完可返回重选。"""
    print(
        "\n"
        "========================================\n"
        "  pot14 本地客户端（交互菜单）\n"
        "  AutoDL 上需已运行 deploy.py --port 6006\n"
        "========================================",
        flush=True,
    )
    while True:
        print(
            "\n请选择：\n"
            "  1) 测试模式  — 模拟串口/相机，测服务器推理延迟\n"
            "  2) 实机模式  — 真实串口 + USB 三路相机\n"
            "  3) 列出串口\n"
            "  4) 列出摄像头\n"
            "  5) 预览摄像头（确认 top/left/right）\n"
            "  6) 修改 WebSocket 地址 / 任务 / 步数\n"
            "  0) 退出\n",
            flush=True,
        )
        choice = input("输入序号: ").strip()

        if choice == "0":
            print("退出。")
            return 0
        if choice == "3":
            print_ports()
            continue
        if choice == "4":
            print_cameras()
            continue
        if choice == "5":
            raw = _prompt(
                "camera-indices top left right（空格分隔）",
                " ".join(str(i) for i in args.camera_indices),
            )
            try:
                parts = [int(x) for x in raw.split()]
                if len(parts) != 3:
                    raise ValueError
                args.camera_indices = parts
            except ValueError:
                print("  格式错误，保持原值", flush=True)
            preview_cameras(tuple(args.camera_indices), DEFAULT_IMAGE_HW)
            continue
        if choice == "6":
            args.ws_url = _prompt("WebSocket 地址", args.ws_url)
            args.task = _prompt("任务文本", args.task)
            args.steps = _prompt_int("推理轮数 steps", args.steps)
            print(
                f"  已更新: ws={normalize_ws_url(args.ws_url)}  "
                f"task={args.task}  steps={args.steps}",
                flush=True,
            )
            continue
        if choice == "1":
            args.steps = _prompt_int("本次测试步数", args.steps)
            code = run_once(args, "mock-latency")
            print(f"\n测试结束（exit={code}）。记录在 {args.log_dir}", flush=True)
            again = _prompt("返回菜单？(Y/n)", "Y").lower()
            if again in ("n", "no", "q"):
                return code
            continue
        if choice == "2":
            print_ports()
            args.serial_port = _prompt("串口路径", args.serial_port)
            if not args.serial_port:
                print("  未填写串口，取消实机运行", flush=True)
                continue
            print_cameras()
            raw = _prompt(
                "camera-indices top left right",
                " ".join(str(i) for i in args.camera_indices),
            )
            try:
                parts = [int(x) for x in raw.split()]
                if len(parts) != 3:
                    raise ValueError
                args.camera_indices = parts
            except ValueError:
                print("  格式错误，使用原 camera-indices", flush=True)
            args.task = _prompt("任务文本", args.task)
            args.steps = _prompt_int("推理轮数", args.steps)
            code = run_once(args, "serial")
            print(f"\n实机运行结束（exit={code}）。记录在 {args.log_dir}", flush=True)
            again = _prompt("返回菜单？(Y/n)", "Y").lower()
            if again in ("n", "no", "q"):
                return code
            continue

        print("  无效选项，请输入 0–6", flush=True)


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.list_ports:
        print_ports()
        return 0

    if args.list_cameras:
        print_cameras()
        print(
            "\n推理需要三路，顺序：--camera-indices <top> <left> <right>\n"
            "例如：--camera-indices 0 1 2"
        )
        return 0

    if args.preview_cameras:
        return preview_cameras(tuple(args.camera_indices), DEFAULT_IMAGE_HW)

    # 默认进入菜单；命令行显式 --mode mock-latency/serial 则单次运行
    if args.mode == "menu":
        return interactive_menu(args)
    return run_once(args, args.mode)


if __name__ == "__main__":
    raise SystemExit(main())
