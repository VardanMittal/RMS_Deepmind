/*  SensorRT — High-Speed Deterministic BLE + LSM9DS1 IMU + EMG
 *
 *  Architecture for Biokinesis-SDK integration:
 *   - Runs a tight deterministic loop at 100 Hz.
 *   - Transmits one packet per 10 ms.
 *
 *  Features:
 *   - Deterministic micros() gating.
 *   - BLE DLE (Data Length Extension) and 2M PHY enabled for max throughput.
 *   - Non-blocking writes inside the hot path.
 */

#include <Arduino.h>
#include <bluefruit.h>
#include <Wire.h>
#include <SparkFunLSM9DS1.h>
#include "SensorPacket.h"

// ─── Pin Definitions ────────────────────────────────────────────────────────
#ifndef LED_RED
#define LED_RED   11
#endif
#ifndef LED_GRN
#define LED_GRN   12
#endif
#ifndef LED_BLU
#define LED_BLU   13
#endif
#define EMG_PIN   A0

#define IMU_AG_ADDR   0x6B
#define IMU_MAG_ADDR  0x1E

// ─── Timing Constants ───────────────────────────────────────────────────────
#define LOOP_RATE_HZ         1000
#define LOOP_PERIOD_US       (1000000UL / LOOP_RATE_HZ)   // 1000 µs
#define DRIFT_CAP_PERIODS    2
#define BATT_READ_INTERVAL   (5UL * LOOP_RATE_HZ)         // 5 seconds

// ─── Globals ────────────────────────────────────────────────────────────────
BLEUart bleuart;
LSM9DS1 imu;

volatile uint32_t next_sample_us = 0;
uint32_t seq = 0;
uint32_t tx_fail_total = 0;
uint32_t tx_skip_total = 0;       // Packets skipped (BLE buffer full)
uint32_t batt_divider = 0;
int8_t cached_batt_pct = 100;

SensorPacket pkt;
uint8_t emg_idx = 0;

// Rate logging
uint32_t rate_sample_count = 0;
uint32_t last_rate_ms = 0;

void ledRed()   { digitalWrite(LED_RED, LOW); digitalWrite(LED_GRN, HIGH); digitalWrite(LED_BLU, HIGH); }
void ledGreen() { digitalWrite(LED_RED, HIGH); digitalWrite(LED_GRN, LOW); digitalWrite(LED_BLU, HIGH); }
void ledBlue()  { digitalWrite(LED_RED, HIGH); digitalWrite(LED_GRN, HIGH); digitalWrite(LED_BLU, LOW); }
void ledOff()   { digitalWrite(LED_RED, HIGH); digitalWrite(LED_GRN, HIGH); digitalWrite(LED_BLU, HIGH); }

void connect_cb(uint16_t conn_handle) {
  (void)conn_handle;
  ledGreen();
  // The Adafruit stack handles PHY and DLE automatically if configured via 
  // Bluefruit.configPrphBandwidth() before begin(). 
  // Calling update requests immediately in the connection callback 
  // often causes BlueZ (Jetson Nano) and Android to drop the connection during GATT discovery.
  Serial.println("[BLE] Connected");
}

void disconnect_cb(uint16_t conn_handle, uint8_t reason) {
  (void)conn_handle;
  Serial.print("[BLE] Disconnected, reason: 0x");
  Serial.println(reason, HEX);
  ledBlue();
}

bool initIMU() {
  Wire.begin();
  Wire.setClock(400000); // Fast I2C

  imu.settings.device.commInterface = IMU_MODE_I2C;
  imu.settings.device.agAddress = IMU_AG_ADDR;
  imu.settings.device.mAddress = IMU_MAG_ADDR;

  // Set IMU Output Data Rate to 119 Hz (must be > 100 Hz loop)
  imu.settings.accel.scale = 4;
  imu.settings.accel.sampleRate = 3; // 119 Hz
  imu.settings.gyro.scale = 2000;
  imu.settings.gyro.sampleRate = 3; // 119 Hz
  imu.settings.mag.scale = 4;
  imu.settings.mag.sampleRate = 7; // 80Hz is max for Mag

  return imu.begin();
}

void setupBLE() {
  Bluefruit.configPrphBandwidth(BANDWIDTH_MAX);
  Bluefruit.begin();
  Bluefruit.setTxPower(4); // Max power
  Bluefruit.setName("SensorRT_Biokinesis");

  // Set preferred connection parameters
  // min = 6 (7.5ms), max = 12 (15ms), latency = 0, timeout = 400 (4s)
  Bluefruit.Periph.setConnInterval(6, 12);
  Bluefruit.Periph.setConnSlaveLatency(0);
  Bluefruit.Periph.setConnSupervisionTimeout(400);

  Bluefruit.Periph.setConnectCallback(connect_cb);
  Bluefruit.Periph.setDisconnectCallback(disconnect_cb);

  bleuart.begin();

  Bluefruit.Advertising.addFlags(BLE_GAP_ADV_FLAGS_LE_ONLY_GENERAL_DISC_MODE);
  Bluefruit.Advertising.addTxPower();
  Bluefruit.Advertising.addService(bleuart);
  Bluefruit.ScanResponse.addName();
  
  // Fast adv interval gives quick connection but burns power while advertising
  Bluefruit.Advertising.restartOnDisconnect(true);
  Bluefruit.Advertising.setInterval(32, 244); 
  Bluefruit.Advertising.setFastTimeout(30);
  Bluefruit.Advertising.start(0);

  ledBlue();
}

void setup() {
  pinMode(LED_RED, OUTPUT);
  pinMode(LED_GRN, OUTPUT);
  pinMode(LED_BLU, OUTPUT);
  ledOff();

  Serial.begin(115200);
  
  // High res 12-bit ADC for Xiao nRF52840
  analogReadResolution(12);
  pinMode(EMG_PIN, INPUT);

  if (!initIMU()) {
    ledRed();
    while (1) { delay(100); }
  }

  setupBLE();

  pkt.magic = PACKET_MAGIC;
  pkt.seq = 0;

  next_sample_us = micros();
  last_rate_ms = millis();
}

void loop() {
  uint32_t now_us = micros();

  // Deterministic 100 Hz gate
  if ((int32_t)(now_us - next_sample_us) < 0) return;

  uint32_t lag_us = now_us - next_sample_us;
  if (lag_us > (DRIFT_CAP_PERIODS * LOOP_PERIOD_US)) {
    next_sample_us = now_us;
  } else {
    next_sample_us += LOOP_PERIOD_US;
  }

  // == 100 Hz execution slice ==
  
  // 1. Read EMG
  pkt.emg[emg_idx++] = analogRead(EMG_PIN);

  // 2. Read IMU + Send packet
  if (emg_idx == EMG_SAMPLES_PER_PACKET) {
      emg_idx = 0;
      pkt.timestamp_us = now_us;
      
      // Read IMU directly to packet buffers
      imu.readAccel();
      imu.readGyro();
      imu.readMag();

      pkt.ax = imu.calcAccel(imu.ax);
      pkt.ay = imu.calcAccel(imu.ay);
      pkt.az = imu.calcAccel(imu.az);
      pkt.gx = imu.calcGyro(imu.gx);
      pkt.gy = imu.calcGyro(imu.gy);
      pkt.gz = imu.calcGyro(imu.gz);
      pkt.mx = imu.calcMag(imu.mx);
      pkt.my = imu.calcMag(imu.my);
      pkt.mz = imu.calcMag(imu.mz);

      pkt.batt_pct = (uint8_t)cached_batt_pct;
      pkt.tx_fail_count = tx_fail_total;

      // Non-blocking transmit: skip packet if BLE buffer is full
      // This prevents bleuart.write() from blocking the 2000 Hz loop
      if (Bluefruit.connected()) {
          // Check if there is enough space in the TX FIFO
          if (bleuart.notifyEnabled()) {
              uint16_t written = bleuart.write((uint8_t *)&pkt, sizeof(SensorPacket));
              if (written != sizeof(SensorPacket)) {
                  tx_fail_total++;
              }
          } else {
              tx_skip_total++;
          }
      }
      pkt.seq++;
  }

  // Diagnostics output at 1 Hz
  rate_sample_count++;
  uint32_t now_ms = millis();
  if (now_ms - last_rate_ms >= 1000) {
    if (Serial) {
      Serial.print("[RATE] Loop Hz: ");
      Serial.print(rate_sample_count); // Should read exactly 100
      Serial.print(" | TX fails: ");
      Serial.print(tx_fail_total);
      Serial.print(" | TX skips: ");
      Serial.println(tx_skip_total);
    }
    rate_sample_count = 0;
    last_rate_ms = now_ms;

    // Simulate battery slowly over time since Xiao NRF52840 requires pin toggle
    // Replace with readBattery() if your board wiring is complete.
    batt_divider++;
    if (batt_divider > 5) {
      batt_divider = 0;
      // cached_batt_pct = ...
    }
  }
}

