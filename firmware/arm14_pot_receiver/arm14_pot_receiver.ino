#include <Arduino.h>
#include "BluetoothSerial.h"
#include "esp_bt_device.h"

BluetoothSerial SerialBT;
HardwareSerial STM32Serial(2);

static constexpr char RX_BT_DEVICE_NAME[] = "ESP32_ARM14_POT_RX";
static constexpr char FW_VERSION[] = "2026-07-22-raw-forward-v3";
static constexpr uint32_t STM32_UART_BAUD = 115200;
static constexpr int STM32_UART_RX_PIN = 16;
static constexpr int STM32_UART_TX_PIN = 17;
static constexpr uint8_t POT_COUNT = 14;
static constexpr uint8_t ADC_DIGITS = 4;
static constexpr uint16_t ADC_MAX_VALUE = 4095;
static constexpr size_t FRAME_DATA_LEN = 1 + POT_COUNT * ADC_DIGITS;
static constexpr size_t FRAME_LEN = FRAME_DATA_LEN + 1 + 4;
static constexpr size_t LINE_BUFFER_SIZE = 80;
static constexpr uint32_t INPUT_TIMEOUT_MS = 300;
static constexpr uint32_t MAC_REPORT_PERIOD_MS = 1000;
static constexpr uint32_t DATA_REPORT_PERIOD_MS = 1000;

static char btLine[LINE_BUFFER_SIZE];
static size_t btLineLength = 0;
static bool btWasConnected = false;
static bool commandStreamActive = false;
static uint32_t lastValidFrameMs = 0;
static uint32_t forwardedFrameCount = 0;
static uint32_t lastMacReportMs = 0;
static uint32_t lastDataReportMs = 0;

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

static void printReceiverMac()
{
  Serial.print("Receiver local MAC: ");
  printMacBytes(esp_bt_dev_get_address());
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

static bool decodeRawFrame(const char *data, size_t length,
                           uint16_t *values)
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

static void forwardStop(const char *reason)
{
  STM32Serial.print("STOP\n");
  commandStreamActive = false;
  Serial.print("[UART7] STOP: ");
  Serial.println(reason);
}

static void printReceivedRaw(const char *data, size_t length)
{
  uint16_t values[POT_COUNT];
  if (!decodeRawFrame(data, length, values))
  {
    return;
  }
  Serial.print("BT_RX_RAW,seq=");
  Serial.print(forwardedFrameCount);
  for (uint8_t index = 0; index < POT_COUNT; index++)
  {
    Serial.print(",P");
    Serial.print(index);
    Serial.print('=');
    Serial.print(values[index]);
  }
  Serial.println();
}

static void handleRawFrame(const char *data, size_t length)
{
  if (!decodeRawFrame(data, length, nullptr))
  {
    return;
  }
  STM32Serial.write((const uint8_t *)data, length);
  STM32Serial.write('\n');
  lastValidFrameMs = millis();
  commandStreamActive = true;
  forwardedFrameCount++;

  if ((millis() - lastDataReportMs) >= DATA_REPORT_PERIOD_MS)
  {
    lastDataReportMs = millis();
    printReceivedRaw(data, length);
  }
  if ((forwardedFrameCount % 100U) == 1U)
  {
    Serial.print("[UART7] valid raw frames forwarded: ");
    Serial.println(forwardedFrameCount);
  }
}

static void pushBluetoothByte(char ch)
{
  if (ch == '\r')
  {
    return;
  }
  if (ch == '\n')
  {
    if (btLineLength > 0U)
    {
      handleRawFrame(btLine, btLineLength);
    }
    btLineLength = 0U;
    return;
  }
  if ((ch < 0x21) || (ch > 0x7E))
  {
    btLineLength = 0U;
    return;
  }
  if ((ch == 'P') && (btLineLength > 0U))
  {
    btLineLength = 0U;
  }
  if ((btLineLength == 0U) && (ch != 'P'))
  {
    return;
  }
  if (btLineLength >= LINE_BUFFER_SIZE)
  {
    btLineLength = 0U;
    return;
  }
  btLine[btLineLength++] = ch;
  if (btLineLength == FRAME_LEN)
  {
    handleRawFrame(btLine, btLineLength);
    btLineLength = 0U;
  }
}

static void printBootInfo()
{
  Serial.println("=== ESP32 14-Pot RAW Receiver ===");
  Serial.print("Firmware: ");
  Serial.println(FW_VERSION);
  printReceiverMac();
  Serial.println("Bluetooth input: raw ADC P frames only");
  Serial.println("UART2 GPIO17 TX -> STM32 PE7 UART7 RX");
  Serial.println("Receiver validates CRC and forwards values without conversion");
}

void setup()
{
  Serial.begin(115200);
  delay(300);
  STM32Serial.begin(STM32_UART_BAUD, SERIAL_8N1,
                    STM32_UART_RX_PIN, STM32_UART_TX_PIN);
  SerialBT.begin(RX_BT_DEVICE_NAME);
  printBootInfo();
  lastMacReportMs = millis();
  forwardStop("receiver startup");
  Serial.println("[BT] Waiting for sender");
}

void loop()
{
  const bool connected = SerialBT.hasClient();
  const uint32_t now = millis();
  if (connected && !btWasConnected)
  {
    Serial.println("[BT] Sender connected");
  }
  if (!connected && btWasConnected)
  {
    btLineLength = 0U;
    forwardStop("Bluetooth disconnected");
    Serial.println("[BT] Sender disconnected");
  }
  btWasConnected = connected;

  if (!connected && ((now - lastMacReportMs) >= MAC_REPORT_PERIOD_MS))
  {
    lastMacReportMs = now;
    printReceiverMac();
  }
  while (SerialBT.available() > 0)
  {
    pushBluetoothByte((char)SerialBT.read());
  }
  if (commandStreamActive && ((now - lastValidFrameMs) > INPUT_TIMEOUT_MS))
  {
    forwardStop("raw frame timeout");
  }
  delay(2);
}
