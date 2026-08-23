import logging
import time
from typing import Dict, Optional, Tuple

from typing_extensions import override
import websockets.sync.client

import functools
import msgpack
import numpy as np

from typing import Optional
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig


def pack_array(obj):
    if (isinstance(obj, (np.ndarray, np.generic))) and obj.dtype.kind in ("V", "O", "c"):
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


def unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])

    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])

    return obj


Packer = functools.partial(msgpack.Packer, default=pack_array)
packb = functools.partial(msgpack.packb, default=pack_array)

Unpacker = functools.partial(msgpack.Unpacker, object_hook=unpack_array)
unpackb = functools.partial(msgpack.unpackb, object_hook=unpack_array)


from PIL import Image


def convert_to_uint8(img: np.ndarray) -> np.ndarray:
    """Converts an image to uint8 if it is a float image.

    This is important for reducing the size of the image when sending it over the network.
    """
    if np.issubdtype(img.dtype, np.floating):
        img = (255 * img).astype(np.uint8)
    return img


def resize_with_pad(images: np.ndarray, height: int, width: int, method=Image.BILINEAR) -> np.ndarray:
    """Replicates tf.image.resize_with_pad for multiple images using PIL. Resizes a batch of images to a target height.

    Args:
        images: A batch of images in [..., height, width, channel] format.
        height: The target height of the image.
        width: The target width of the image.
        method: The interpolation method to use. Default is bilinear.

    Returns:
        The resized images in [..., height, width, channel].
    """
    # If the images are already the correct size, return them as is.
    if images.shape[-3:-1] == (height, width):
        return images

    original_shape = images.shape

    images = images.reshape(-1, *original_shape[-3:])
    resized = np.stack([_resize_with_pad_pil(Image.fromarray(im), height, width, method=method) for im in images])
    return resized.reshape(*original_shape[:-3], *resized.shape[-3:])


def _resize_with_pad_pil(image: Image.Image, height: int, width: int, method: int) -> Image.Image:
    """Replicates tf.image.resize_with_pad for one image using PIL. Resizes an image to a target height and
    width without distortion by padding with zeros.

    Unlike the jax version, note that PIL uses [width, height, channel] ordering instead of [batch, h, w, c].
    """
    cur_width, cur_height = image.size
    if cur_width == width and cur_height == height:
        return image  # No need to resize if the image is already the correct size.

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_image = image.resize((resized_width, resized_height), resample=method)

    zero_image = Image.new(resized_image.mode, (width, height), 0)
    pad_height = max(0, int((height - resized_height) / 2))
    pad_width = max(0, int((width - resized_width) / 2))
    zero_image.paste(resized_image, (pad_width, pad_height))
    assert zero_image.size == (width, height)
    return zero_image


FPS = 30

TASK = "Grab the black cube"

ACTION_KEY_MAP = [
    'shoulder_pan.pos',
    'shoulder_lift.pos',
    'elbow_flex.pos',
    'wrist_flex.pos',
    'wrist_roll.pos',
    'gripper.pos'
]


def so100_to_lingbot(obs: dict, task: Optional[str] = None) -> dict:
    vla_input = {}

    # ── 1. 图像：重命名为 yaml 中定义的完整路径 ──────────────────
    image_map = {
        "camera_top":         "observation.images.camera_top",
        "camera_wrist_left":  "observation.images.camera_wrist_left",
        "camera_wrist_right": "observation.images.camera_wrist_right",
    }
    for raw_key, vla_key in image_map.items():
        if raw_key in obs:
            img = obs[raw_key]
            assert img.dtype == np.uint8, f"{raw_key}: 必须是 uint8，当前 {img.dtype}"
            assert img.shape[-1] == 3,  f"{raw_key}: 必须是 RGB，当前 shape {img.shape}"
            vla_input[vla_key] = img

    # ── 2. 状态：构建 6 维向量（5 臂关节 + 1 夹爪）───────────────
    # 顺序必须与 yaml 中的 start:end 严格一致
    motor_keys = [
        'shoulder_pan.pos',   # index 0  → arm.position[0]
        'shoulder_lift.pos',  # index 1  → arm.position[1]
        'elbow_flex.pos',     # index 2  → arm.position[2]
        'wrist_flex.pos',     # index 3  → arm.position[3]
        'wrist_roll.pos',     # index 4  → arm.position[4]
        'gripper.pos',        # index 5  → effector.position[0]
    ]

    state_vec = np.array([obs[k] for k in motor_keys], dtype=np.float32)

    vla_input["observation.state"] = state_vec

    # ── 3. 语言指令 ──────────────────────────────────────────────
    if task is not None:
        vla_input["task"] = task

    return vla_input


def send_vla_action(robot, action_vec: np.ndarray):
    """
    将 LingBotVLA 输出的 6 维绝对位置发送给 SO-100。

    Args:
        robot: SO100Follower 实例
        action_vec: (6,) float32 或 float64，来自 action_chunk['action'][0]
    """
    assert action_vec.shape == (6,), f"动作维度必须是 (6,)，当前 {action_vec.shape}"

    print(f"action_vec = {action_vec}")

    # 1. 向量 → 字典（SO100Follower 的 send_action 期望字典格式）
    action_dict = {
        name: float(action_vec[i])  # 确保是 Python float，非 numpy scalar
        for i, name in enumerate(ACTION_KEY_MAP)
    }

    # 2. 发送（SO100Follower 内部会处理为 Dynamixel 协议）
    robot.send_action(action_dict)


class WebsocketClientPolicy:
    """Implements the Policy interface by communicating with a server over websocket.

    See WebsocketPolicyServer for a corresponding server implementation.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: Optional[int] = None,
        api_key: Optional[str] = None,
        *,
        uri: Optional[str] = None,
    ) -> None:
        # 优先用完整 URI（AutoDL 公网常用 wss://host:8443）
        if uri:
            self._uri = uri.rstrip("/")
            if self._uri.startswith("https://"):
                self._uri = "wss://" + self._uri[len("https://") :]
            elif self._uri.startswith("http://"):
                self._uri = "ws://" + self._uri[len("http://") :]
            elif not (self._uri.startswith("ws://") or self._uri.startswith("wss://")):
                self._uri = "wss://" + self._uri
        else:
            self._uri = f"ws://{host}"
            if port is not None:
                self._uri += f":{port}"
        self._packer = Packer()
        self._api_key = api_key
        self._ws, self._server_metadata = self._wait_for_server()

    def get_server_metadata(self) -> Dict:
        return self._server_metadata

    def _wait_for_server(self) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
        logging.info(f"Waiting for server at {self._uri}...")
        while True:
            try:
                headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
                conn = websockets.sync.client.connect(
                    self._uri, compression=None, max_size=None, additional_headers=headers
                )
                metadata = unpackb(conn.recv())
                return conn, metadata
            except ConnectionRefusedError:
                logging.info("Still waiting for server...")
                time.sleep(5)
            except OSError as exc:
                logging.info("Still waiting for server... (%s)", exc)
                time.sleep(5)

    @override
    def infer(self, obs: Dict) -> Dict:  # noqa: UP006
        data = self._packer.pack(obs)
        self._ws.send(data)
        response = self._ws.recv()
        if isinstance(response, str):
            # we're expecting bytes; if the server sends a string, it's an error.
            raise RuntimeError(f"Error in inference server:\n{response}")
        return unpackb(response)

    @override
    def reset(self, robo_name: str) -> None:
        self.infer(dict(reset=True, robo_name=robo_name))


def list_mac_serial_ports() -> list[str]:
    """列出本机串口（Mac 蓝牙/USB 常用 /dev/cu.*）。"""
    import glob

    ports = sorted(set(glob.glob("/dev/cu.*") + glob.glob("/dev/tty.*")))
    return ports


if __name__ == "__main__":
    import argparse
    import os

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="本地串口机械臂 + 云端 WebSocket 推理（SO-100 示例）")
    parser.add_argument(
        "--list-ports",
        action="store_true",
        help="只列出本机串口后退出（Mac 查蓝牙串口用这个）",
    )
    parser.add_argument(
        "--serial-port",
        default=os.environ.get("ROBOT_SERIAL_PORT", "/dev/ttyACM1"),
        help="机械臂串口路径，Mac 蓝牙多为 /dev/cu.XXXX",
    )
    parser.add_argument(
        "--ws-url",
        default=os.environ.get(
            "VLA_WS_URL",
            "wss://u1087324-85uh-a5fac6ab.weste.seetacloud.com:8443",
        ),
        help="AutoDL 公网 WebSocket 地址（可用 https://...，会自动改成 wss://）",
    )
    parser.add_argument("--task", default=TASK, help="任务文本")
    parser.add_argument("--steps", type=int, default=300)
    args = parser.parse_args()

    if args.list_ports:
        ports = list_mac_serial_ports()
        print("本机串口设备：")
        if not ports:
            print("  （未找到 /dev/cu.* 或 /dev/tty.*）")
        for p in ports:
            print(f"  {p}")
        print("\nMac 提示：优先选 /dev/cu.开头；蓝牙 SPP 常见名含 Bluetooth / SPP / 设备名。")
        raise SystemExit(0)

    # Create robot configuration
    robot_config = SO100FollowerConfig(
        id="arm_follower",
        cameras={
            "camera_top": OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=FPS),
            "camera_wrist_left": OpenCVCameraConfig(index_or_path=2, width=640, height=480, fps=FPS),
            "camera_wrist_right": OpenCVCameraConfig(index_or_path=4, width=640, height=480, fps=FPS),
        },
        port=args.serial_port,
    )

    robot = SO100Follower(robot_config)

    robot.connect()

    print(f"串口: {args.serial_port}")
    print(f"WebSocket: {args.ws_url}")
    policy_on_device = WebsocketClientPolicy(uri=args.ws_url)

    # policy_on_device.reset("so100")

    try:
        for step in range(args.steps):
            raw_obs = robot.get_observation()

            obs = so100_to_lingbot(raw_obs, task=args.task)

            action_chunk = policy_on_device.infer(obs)

            print(f"action_chunk server_timing = {action_chunk['server_timing']}")

            for action in action_chunk["action"]:
                send_vla_action(robot, action)
                time.sleep(1.0 / FPS)  # 30Hz 执行，但每 16 步才更新一次观测

    except KeyboardInterrupt:
        print("检测到 Ctrl+C，正在终止...")
    finally:
        robot.disconnect()
