# pot14 ESP32 固件

## arm14_pot_sender（遥操作发送端）

- 版本：`2026-08-23-mode12-v3`
- Arduino 板型：`ESP32 Dev Module`
- USB 串口命令：
  - `1`：遥操作（电位器 → 蓝牙）
  - `2`：模型模式（屏蔽电位器，仅 USB 的 `P...*CRC` 帧 → 蓝牙）
- Mac 客户端连接时会自动发 `2`，断开时发 `1`

烧录：打开 `arm14_pot_sender/arm14_pot_sender.ino` 上传。

## arm14_pot_receiver（接收端）

校验 CRC 后转发到 STM32 UART7。保持与发送端相同的 RAW 帧格式。
