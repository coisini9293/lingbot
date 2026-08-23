#include <Arduino.h>
#include <string.h>
#include "BluetoothSerial.h"
#include "esp_bt_device.h"

BluetoothSerial SerialBT;

static constexpr char TX_BT_DEVICE_NAME[] = "ESP32_ARM14_POT_TX";
/*
 * 模式切换（USB 串口发一行后回车）：
 *   1  = 遥操作（电位器 → 蓝牙）
 *   2  = 模型/PC（完全屏蔽电位器；仅 USB 的 P 帧 → 蓝牙）
 */
static constexpr char FW_VERSION[] = "2026-08-23-mode12-v3";
static const uint8_t KNOWN_BOARD_BT_ADDRESSES[2][6] = {
  {0x68, 0x09, 0x47, 0x52, 0xED, 0x92},
  {0x68, 0x09, 0x47, 0x52, 0xCF, 0x7E}
};
static uint8_t receiverBtAddress[6] = {0};
static bool receiverAddressReady = false;

static constexpr uint8_t POT_COUNT = 14;
static constexpr uint16_t ADC_MAX_VALUE = 4095;
static constexpr uint8_t ADC_DIGITS = 4;
static constexpr uint8_t ADC_MEDIAN_SAMPLE_COUNT = 5;
static constexpr uint32_t SEND_PERIOD_MS = 20;
static constexpr uint32_t CONNECT_RETRY_MS = 1000;
static constexpr uint32_t TELEMETRY_PERIOD_MS = 1000;
static constexpr float ADC_SUPPLY_VOLTAGE = 3.3f;
static constexpr size_t FRAME_DATA_LEN = 1 + POT_COUNT * ADC_DIGITS;
static constexpr size_t FRAME_LEN = FRAME_DATA_LEN + 1 + 4;
static constexpr size_t FRAME_BUFFER_SIZE = FRAME_LEN + 2;
static constexpr size_t USB_LINE_BUFFER_SIZE = 80;

enum ControlMode : uint8_t
{
  MODE_TELEOP = 1,
  MODE_MODEL = 2
};

static const uint8_t POT_PINS[POT_COUNT] = {
  34, 39, 32, 33, 25, 26, 27,
  14, 12, 13, 4, 36, 2, 15
};

static uint16_t latestAdc[POT_COUNT] = {0};
static uint32_t lastSendMs = 0;
static uint32_t lastTelemetryMs = 0;
static uint32_t sentFrameCount = 0;
static uint32_t pcForwardCount = 0;
static uint32_t pcDropNoBtCount = 0;
static char lastPcFrame[FRAME_BUFFER_SIZE];
static size_t lastPcFrameLen = 0;
static ControlMode controlMode = MODE_TELEOP;
static volatile uint32_t btConnectAttemptCount = 0;
static volatile uint32_t btConnectFailureCount = 0;
static volatile bool btConnectInProgress = false;
static bool btConnectTaskStarted = false;

static char usbLine[USB_LINE_BUFFER_SIZE];
static size_t usbLineLength = 0;

static void printMacBytes(const uint8_t *mac);
static bool macEquals(const uint8_t *left, const uint8_t *right);
static bool selectReceiverAddress();
static uint16_t crc16Ccitt(const uint8_t *data, size_t length);
static bool hexNibble(char value, uint8_t &nibble);
static bool decodeRawFrame(const char *data, size_t length, uint16_t *values);
static void configureAdcPins();
static uint16_t readRawAdc(uint8_t pin);
static size_t buildRawFrameFromLatest(char *frame, size_t frameSize);
static size_t samplePotsToLatest();
static void bluetoothConnectTask(void *parameter);
static void sampleAndSendTeleop();
static void sendHeldOrPcFrame();
static void forwardPcFrame(const char *data, size_t length);
static void enterMode(ControlMode mode);
static void handleUsbLine(const char *line, size_t length);
static void pushUsbByte(char ch);
static void pollUsbCommands();
static void printRawAdcTelemetry();
static void printTelemetry();
static void printBootInfo();

static void printMacBytes(const uint8_t *mac)
{
  if (mac == nullptr)
  {
    Serial.println("unknown");
    return;
  }
  for (uint8_t index = 0; index < 6; index++)
  {
    if (index > 0U)
    {
      Serial.print(':');
    }
    if (mac[index] < 0x10U)
    {
      Serial.print('0');
    }
    Serial.print(mac[index], HEX);
  }
  Serial.println();
}

static bool macEquals(const uint8_t *left, const uint8_t *right)
{
  if ((left == nullptr) || (right == nullptr))
  {
    return false;
  }
  for (uint8_t index = 0; index < 6; index++)
  {
    if (left[index] != right[index])
    {
      return false;
    }
  }
  return true;
}

static bool selectReceiverAddress()
{
  const uint8_t *local = esp_bt_dev_get_address();
  const uint8_t *remote = nullptr;

  if (macEquals(local, KNOWN_BOARD_BT_ADDRESSES[0]))
  {
    remote = KNOWN_BOARD_BT_ADDRESSES[1];
  }
  else if (macEquals(local, KNOWN_BOARD_BT_ADDRESSES[1]))
  {
    remote = KNOWN_BOARD_BT_ADDRESSES[0];
  }
  if (remote == nullptr)
  {
    return false;
  }
  memcpy(receiverBtAddress, remote, sizeof(receiverBtAddress));
  return true;
}

static uint16_t crc16Ccitt(const uint8_t *data, size_t length)
{
  uint16_t crc = 0xFFFFU;
  for (size_t index = 0; index < length; index++)
  {
    crc ^= (uint16_t)data[index] << 8;
    for (uint8_t bit = 0; bit < 8; bit++)
    {
      crc = (crc & 0x8000U) ?
        (uint16_t)((crc << 1) ^ 0x1021U) : (uint16_t)(crc << 1);
    }
  }
  return crc;
}

static bool hexNibble(char value, uint8_t &nibble)
{
  if ((value >= '0') && (value <= '9'))
  {
    nibble = (uint8_t)(value - '0');
    return true;
  }
  if ((value >= 'A') && (value <= 'F'))
  {
    nibble = (uint8_t)(value - 'A' + 10);
    return true;
  }
  if ((value >= 'a') && (value <= 'f'))
  {
    nibble = (uint8_t)(value - 'a' + 10);
    return true;
  }
  return false;
}

static bool decodeRawFrame(const char *data, size_t length, uint16_t *values)
{
  uint16_t receivedCrc = 0U;
  size_t position = 1U;

  if ((data == nullptr) || (length != FRAME_LEN) ||
      (data[0] != 'P') || (data[FRAME_DATA_LEN] != '*'))
  {
    return false;
  }
  for (uint8_t pot = 0; pot < POT_COUNT; pot++)
  {
    uint16_t value = 0U;
    for (uint8_t digit = 0; digit < ADC_DIGITS; digit++)
    {
      const char ch = data[position++];
      if ((ch < '0') || (ch > '9'))
      {
        return false;
      }
      value = (uint16_t)(value * 10U + (uint16_t)(ch - '0'));
    }
    if (value > ADC_MAX_VALUE)
    {
      return false;
    }
    if (values != nullptr)
    {
      values[pot] = value;
    }
  }
  for (uint8_t digit = 0; digit < 4U; digit++)
  {
    uint8_t nibble;
    if (!hexNibble(data[FRAME_DATA_LEN + 1U + digit], nibble))
    {
      return false;
    }
    receivedCrc = (uint16_t)((receivedCrc << 4) | nibble);
  }
  return receivedCrc == crc16Ccitt((const uint8_t *)data, FRAME_DATA_LEN);
}

static void configureAdcPins()
{
  analogReadResolution(12);
  for (uint8_t index = 0; index < POT_COUNT; index++)
  {
    pinMode(POT_PINS[index], INPUT);
    analogSetPinAttenuation(POT_PINS[index], ADC_11db);
  }
}

static uint16_t readRawAdc(uint8_t pin)
{
  uint16_t samples[ADC_MEDIAN_SAMPLE_COUNT];

  (void)analogRead(pin);
  delayMicroseconds(80);

  for (uint8_t sample = 0; sample < ADC_MEDIAN_SAMPLE_COUNT; sample++)
  {
    int value = analogRead(pin);
    if (value < 0)
    {
      value = 0;
    }
    if (value > ADC_MAX_VALUE)
    {
      value = ADC_MAX_VALUE;
    }
    samples[sample] = (uint16_t)value;
    delayMicroseconds(20);
  }

  for (uint8_t index = 1; index < ADC_MEDIAN_SAMPLE_COUNT; index++)
  {
    const uint16_t value = samples[index];
    uint8_t insert = index;
    while ((insert > 0U) && (samples[insert - 1U] > value))
    {
      samples[insert] = samples[insert - 1U];
      insert--;
    }
    samples[insert] = value;
  }

  return samples[ADC_MEDIAN_SAMPLE_COUNT / 2U];
}

static size_t buildRawFrameFromLatest(char *frame, size_t frameSize)
{
  static const char hex[] = "0123456789ABCDEF";
  size_t position = 0;
  uint16_t crc;

  if ((frame == nullptr) || (frameSize < FRAME_BUFFER_SIZE))
  {
    return 0;
  }
  frame[position++] = 'P';
  for (uint8_t index = 0; index < POT_COUNT; index++)
  {
    const uint16_t value = latestAdc[index];
    frame[position++] = (char)('0' + ((value / 1000U) % 10U));
    frame[position++] = (char)('0' + ((value / 100U) % 10U));
    frame[position++] = (char)('0' + ((value / 10U) % 10U));
    frame[position++] = (char)('0' + (value % 10U));
  }
  crc = crc16Ccitt((const uint8_t *)frame, FRAME_DATA_LEN);
  frame[position++] = '*';
  frame[position++] = hex[(crc >> 12) & 0x0FU];
  frame[position++] = hex[(crc >> 8) & 0x0FU];
  frame[position++] = hex[(crc >> 4) & 0x0FU];
  frame[position++] = hex[crc & 0x0FU];
  frame[position++] = '\n';
  frame[position] = '\0';
  return position;
}

static size_t samplePotsToLatest()
{
  for (uint8_t index = 0; index < POT_COUNT; index++)
  {
    latestAdc[index] = readRawAdc(POT_PINS[index]);
  }
  return buildRawFrameFromLatest(lastPcFrame, sizeof(lastPcFrame));
}

static void bluetoothConnectTask(void *parameter)
{
  (void)parameter;
  while (true)
  {
    if (!receiverAddressReady)
    {
      vTaskDelay(pdMS_TO_TICKS(1000));
      continue;
    }

    if (!SerialBT.connected())
    {
      btConnectInProgress = true;
      btConnectAttemptCount++;
      if (!SerialBT.connect(receiverBtAddress))
      {
        btConnectFailureCount++;
      }
      btConnectInProgress = false;
      vTaskDelay(pdMS_TO_TICKS(CONNECT_RETRY_MS));
      continue;
    }

    vTaskDelay(pdMS_TO_TICKS(200));
  }
}

static void sampleAndSendTeleop()
{
  char frame[FRAME_BUFFER_SIZE];
  for (uint8_t index = 0; index < POT_COUNT; index++)
  {
    latestAdc[index] = readRawAdc(POT_PINS[index]);
  }
  const size_t length = buildRawFrameFromLatest(frame, sizeof(frame));
  if ((length > 0U) && SerialBT.connected())
  {
    SerialBT.write((const uint8_t *)frame, length);
    sentFrameCount++;
  }
}

static void sendHeldOrPcFrame()
{
  if (!SerialBT.connected())
  {
    return;
  }
  if (lastPcFrameLen == FRAME_LEN)
  {
    SerialBT.write((const uint8_t *)lastPcFrame, lastPcFrameLen);
    SerialBT.write('\n');
    sentFrameCount++;
  }
}

static void forwardPcFrame(const char *data, size_t length)
{
  uint16_t values[POT_COUNT];
  if (!decodeRawFrame(data, length, values))
  {
    return;
  }
  if (controlMode != MODE_MODEL)
  {
    /* 模式1仍允许 USB 点动调试，但不改模式。 */
  }
  for (uint8_t index = 0; index < POT_COUNT; index++)
  {
    latestAdc[index] = values[index];
  }
  if (length < FRAME_BUFFER_SIZE)
  {
    memcpy(lastPcFrame, data, length);
    lastPcFrameLen = length;
  }
  if (!SerialBT.connected())
  {
    pcDropNoBtCount++;
    if ((pcDropNoBtCount % 20U) == 1U)
    {
      Serial.println("[USB] got P frame but BT not connected");
    }
    return;
  }
  SerialBT.write((const uint8_t *)data, length);
  SerialBT.write('\n');
  sentFrameCount++;
  pcForwardCount++;
  if ((pcForwardCount % 50U) == 1U)
  {
    Serial.print("[USB] forwarded PC frames: ");
    Serial.println(pcForwardCount);
  }
}

static void enterMode(ControlMode mode)
{
  if (mode == controlMode)
  {
    Serial.print("[MODE] already ");
    Serial.println((int)mode);
    return;
  }
  controlMode = mode;
  usbLineLength = 0U;
  if (mode == MODE_TELEOP)
  {
    lastPcFrameLen = 0U;
    Serial.println("[MODE] 1 TELEOP — pots -> BT, model USB ignored for stream");
  }
  else
  {
    /* 进入模型模式：冻结当前电位器姿态作初始 hold，之后不再读电位器。 */
    const size_t length = samplePotsToLatest();
    if (length >= FRAME_LEN)
    {
      lastPcFrameLen = FRAME_LEN;
    }
    pcForwardCount = 0U;
    Serial.println("[MODE] 2 MODEL — pots FULLY blocked; only USB P frames -> BT");
    Serial.println("[MODE] holding frozen pose until first PC frame");
  }
  Serial.print("STATUS,FW=");
  Serial.print(FW_VERSION);
  Serial.print(",MODE=");
  Serial.println((int)controlMode);
}

static void handleUsbLine(const char *line, size_t length)
{
  if ((line == nullptr) || (length == 0U))
  {
    return;
  }
  /* 模式命令：单独一行 "1" 或 "2" */
  if ((length == 1U) && (line[0] == '1'))
  {
    enterMode(MODE_TELEOP);
    return;
  }
  if ((length == 1U) && (line[0] == '2'))
  {
    enterMode(MODE_MODEL);
    return;
  }
  if ((length >= 1U) && (line[0] == 'P'))
  {
    if (controlMode == MODE_TELEOP)
    {
      /* 模式1：忽略模型流，避免误控；可用点动时请先发 2 */
      static uint32_t ignoreCount = 0;
      ignoreCount++;
      if ((ignoreCount % 50U) == 1U)
      {
        Serial.println("[USB] ignore P frame in MODE 1; send 2 for model mode");
      }
      return;
    }
    forwardPcFrame(line, length);
  }
}

static void pushUsbByte(char ch)
{
  if (ch == '\r')
  {
    return;
  }
  if (ch == '\n')
  {
    if (usbLineLength > 0U)
    {
      usbLine[usbLineLength] = '\0';
      handleUsbLine(usbLine, usbLineLength);
    }
    usbLineLength = 0U;
    return;
  }
  if ((ch < 0x21) || (ch > 0x7E))
  {
    usbLineLength = 0U;
    return;
  }
  /* 模式数字 1/2：允许作为行首 */
  if ((usbLineLength == 0U) && (ch != 'P') && (ch != '1') && (ch != '2'))
  {
    return;
  }
  if ((ch == 'P') && (usbLineLength > 0U))
  {
    usbLineLength = 0U;
  }
  if (usbLineLength >= (USB_LINE_BUFFER_SIZE - 1U))
  {
    usbLineLength = 0U;
    return;
  }
  usbLine[usbLineLength++] = ch;
  if ((usbLine[0] == 'P') && (usbLineLength == FRAME_LEN))
  {
    handleUsbLine(usbLine, usbLineLength);
    usbLineLength = 0U;
  }
}

static void pollUsbCommands()
{
  while (Serial.available() > 0)
  {
    pushUsbByte((char)Serial.read());
  }
}

static void printRawAdcTelemetry()
{
  /* 模式2也打印同一格式，方便 Mac 解析；数值来自 hold/PC，不是实时电位器。 */
  Serial.print("RAW_ADC");
  for (uint8_t index = 0; index < POT_COUNT; index++)
  {
    Serial.print(",P");
    Serial.print(index);
    Serial.print('=');
    Serial.print(latestAdc[index]);
  }
  Serial.println();
}

static void printTelemetry()
{
  Serial.print("STATUS,FW=");
  Serial.print(FW_VERSION);
  Serial.print(",MODE=");
  Serial.print((int)controlMode);
  Serial.print(",PC_FWD=");
  Serial.print(pcForwardCount);
  Serial.print(",BT_SENT=");
  Serial.print(sentFrameCount);
  Serial.print(",BT=");
  Serial.print(SerialBT.connected() ? "ON" : "OFF");
  Serial.print(",CONNECTING=");
  Serial.print(btConnectInProgress ? 1 : 0);
  Serial.println();
}

static void printBootInfo()
{
  Serial.println("=== ESP32 14-Pot RAW Sender ===");
  Serial.print("Firmware: ");
  Serial.println(FW_VERSION);
  Serial.print("Sender actual BT MAC: ");
  printMacBytes(esp_bt_dev_get_address());
  Serial.print("Receiver selected BT MAC: ");
  printMacBytes(receiverAddressReady ? receiverBtAddress : nullptr);
  Serial.println("USB commands:");
  Serial.println("  1 = TELEOP (pots -> BT)");
  Serial.println("  2 = MODEL  (pots blocked; USB P frames only -> BT)");
  Serial.println("Default MODE=1");
}

void setup()
{
  Serial.begin(115200);
  delay(300);
  configureAdcPins();
  SerialBT.begin(TX_BT_DEVICE_NAME, true);
  receiverAddressReady = selectReceiverAddress();
  printBootInfo();
  controlMode = MODE_TELEOP;
  btConnectTaskStarted =
      (xTaskCreatePinnedToCore(bluetoothConnectTask,
                               "bt_connect",
                               4096,
                               nullptr,
                               1,
                               nullptr,
                               0) == pdPASS);
}

void loop()
{
  const uint32_t now = millis();
  pollUsbCommands();

  if ((now - lastSendMs) >= SEND_PERIOD_MS)
  {
    lastSendMs = now;
    if (controlMode == MODE_TELEOP)
    {
      sampleAndSendTeleop();
    }
    else
    {
      /* 模式2：绝不读电位器；只重发 hold / 最近 PC 帧，避免接收端 STOP。 */
      sendHeldOrPcFrame();
    }
    printRawAdcTelemetry();
  }
  if ((now - lastTelemetryMs) >= TELEMETRY_PERIOD_MS)
  {
    lastTelemetryMs = now;
    printTelemetry();
  }
  delay(1);
}
