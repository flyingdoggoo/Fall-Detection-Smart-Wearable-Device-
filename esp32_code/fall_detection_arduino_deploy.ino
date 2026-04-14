#include <WiFi.h>
#include <WiFiUdp.h>
#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <math.h>

// ================================================================
// 1. CẤU HÌNH WIFI & SERVER
// ================================================================
// const char* ssid       = "ITF - Da Nang";
// const char* password   = "888888888";
// const char* serverHost = "172.20.10.3";

// const char* ssid       = "Giat Ui Tigon 1"; 
// const char* password   = "789789789";
// const char* serverHost = "192.168.1.7";

const char* ssid       = "Veitel";
const char* password   = "12345667";
const char* serverHost = "10.30.239.150";
const uint16_t serverPort = 5683;


#define I2C_SDA         4
#define I2C_SCL         5
#define BUTTON_PIN      9
#define BUZZER_PIN      18      // Chân buzzer cảnh báo ngã (OFFLINE)
#define BUZZER_DUTY_ON  255      // 24/255 ~ 9.4% duty -> giảm dòng đột biến
#define SAMPLE_INTERVAL 20      // 50 Hz = 20 ms
#define WINDOW_SIZE     100     // 100 mẫu = 2 giây / window
#define COAP_LOCAL_PORT 56830
#define COAP_ACK_TIMEOUT_MS 250
#define COAP_MAX_RETRIES 2
#define COAP_CHUNK_SAMPLES 25

#define SENSOR_PACKET_VERSION_LEGACY 1
#define ACCEL_SCALE_LEGACY 400.0f
#define GYRO_SCALE_LEGACY 1000.0f

// ================================================================
// MAX30102 – CHỈ BẬT LED + ĐỌC IR ĐỂ DETECT NGÓN TAY
// ================================================================
#define MAX30102_ADDR       0x57
#define MAX30102_INT_STAT1  0x00
#define MAX30102_FIFO_WR    0x04
#define MAX30102_FIFO_RD    0x06
#define MAX30102_FIFO_DATA  0x07
#define MAX30102_FIFO_CFG   0x08
#define MAX30102_MODE_CFG   0x09
#define MAX30102_SPO2_CFG   0x0A
#define MAX30102_LED1_PA    0x0C  // RED
#define MAX30102_LED2_PA    0x0D  // IR
#define MAX30102_PART_ID    0xFF

#define IR_FINGER_THRESH    50000  // IR > 50000 = có ngón tay

// ================================================================
// FSM – NGƯỠNG PHÁT HIỆN NGÃ OFFLINE
// ================================================================
#define FREE_FALL_THRESH      5.5f   
#define IMPACT_THRESH         50.0f  
#define MIN_FREEFALL_SAMPLES    2    
#define IMPACT_WINDOW_MS      800    
#define POST_FALL_VARIANCE   0.5f   
#define POST_FALL_CHECK_MS   5000    
#define ALERT_DURATION_MS    2000    
#define FSM_TIMEOUT_MS       7000  

// ================================================================
// 2. CẤU TRÚC DỮ LIỆU
// ================================================================
struct SensorSample {
  float timestamp;
  float ax, ay, az;
  float gx, gy, gz;
};

struct WindowPackage {
  SensorSample samples[WINDOW_SIZE];
  int   bpm;
  long  ir;
  float mag_avg;     
  float sma;         
  float max_a;       
  float max_g;       
  float std_accel;   
  float jerk_peak;   
  int   fsm_state;
  bool  fall_detected;
  bool  ready = false;
} currentWindow;

// ================================================================
// FSM STATES
// ================================================================
enum FallState {
  FSM_MONITORING     = 0,
  FSM_FREEFALL       = 1,
  FSM_IMPACT         = 2,
  FSM_FALL_CONFIRMED = 3
};

volatile FallState fallState       = FSM_MONITORING;
unsigned long freefallStartMs      = 0;
int           freefallSampleCount  = 0;
unsigned long impactDetectedMs     = 0;
unsigned long alertStartMs         = 0;
float         postFallMagBuf[200];
int           postFallBufIdx       = 0;
bool          buzzerActive         = false;
bool          fallDetectedFlag     = false;

Adafruit_MPU6050  mpu;

int   beatAvg        = 0;
long  irValue        = 0;
unsigned long lastHrCheck = 0;   

unsigned long loopCount       = 0;
unsigned long lastLoopReport  = 0;

bool          systemState    = false;
unsigned long lastSampleTime = 0;
unsigned long startTime      = 0;
int           bufferIndex    = 0;

const unsigned long longPressMs = 1000;
const unsigned long debounceMs  = 40;

bool          beepRunning      = false;
bool          beepPinHigh      = false;
int           beepRemainingOn  = 0;
unsigned long beepNextToggleAt = 0;
int           beepOnMs         = 0;
int           beepOffMs        = 0;

const char* batchPath       = "api/sensor-batch";

WiFiUDP   coapUdp;
IPAddress serverIP;
bool      serverIPResolved   = false;
uint16_t  nextCoapMessageId  = 0;
uint16_t  nextCoapToken      = 0;

uint8_t sensorPayloadBuf[1300];
uint8_t coapPacketBuf[1500];
uint8_t coapReplyBuf[96];

// ================================================================
// 3. NaN / Inf GUARD
// ================================================================
float safeFloat(float v, float fallback = 0.0f) {
  if (isnan(v) || isinf(v)) return fallback;
  return constrain(v, -160.0f, 160.0f);   
}

float safeGyro(float v, float fallback = 0.0f) {
  if (isnan(v) || isinf(v)) return fallback;
  return constrain(v, -35.0f, 35.0f);     
}

// ================================================================
// MAX30102 – I2C HELPERS (KHÔNG CẦN THƯ VIỆN)
// ================================================================
void max30102_write(uint8_t reg, uint8_t val) {
  Wire.beginTransmission(MAX30102_ADDR);
  Wire.write(reg);
  Wire.write(val);
  Wire.endTransmission();
}

uint8_t max30102_read8(uint8_t reg) {
  Wire.beginTransmission(MAX30102_ADDR);
  Wire.write(reg);
  Wire.endTransmission(false);
  Wire.requestFrom((uint8_t)MAX30102_ADDR, (uint8_t)1);
  return Wire.available() ? Wire.read() : 0;
}

bool max30102_detected = false;

bool max30102_init() {
  // Kiểm tra part ID
  uint8_t partId = max30102_read8(MAX30102_PART_ID);
  if (partId != 0x15) {
    Serial.printf("✗ MAX30102 not found (ID=0x%02X)\n", partId);
    return false;
  }

  // Reset
  max30102_write(MAX30102_MODE_CFG, 0x40);
  delay(100);

  // Mode: SpO2 (RED + IR LEDs on)
  max30102_write(MAX30102_MODE_CFG, 0x03);

  // SPO2 config: ADC range 4096, 100 samples/s, 18-bit resolution
  max30102_write(MAX30102_SPO2_CFG, 0x27);

  // LED cường độ thấp → tiết kiệm điện, đủ detect ngón tay
  max30102_write(MAX30102_LED1_PA, 0x24);  // RED ~7mA
  max30102_write(MAX30102_LED2_PA, 0x24);  // IR  ~7mA

  // FIFO config: sample averaging = 4, rollover enabled
  max30102_write(MAX30102_FIFO_CFG, 0x50);

  // Clear FIFO pointers
  max30102_write(MAX30102_FIFO_WR, 0x00);
  max30102_write(MAX30102_FIFO_RD, 0x00);

  Serial.println("✓ MAX30102 LED ON – detect finger only");
  return true;
}

// Đọc 1 sample IR từ FIFO (3 byte RED + 3 byte IR)
uint32_t max30102_readIR() {
  Wire.beginTransmission(MAX30102_ADDR);
  Wire.write(MAX30102_FIFO_DATA);
  Wire.endTransmission(false);
  Wire.requestFrom((uint8_t)MAX30102_ADDR, (uint8_t)6);

  // Bỏ qua 3 byte RED
  Wire.read(); Wire.read(); Wire.read();

  // Đọc 3 byte IR (18-bit)
  uint32_t ir = 0;
  ir  = (uint32_t)Wire.read() << 16;
  ir |= (uint32_t)Wire.read() << 8;
  ir |= (uint32_t)Wire.read();
  ir &= 0x3FFFF;  // mask 18-bit

  return ir;
}

// ================================================================
// 4. BỘ LỌC KALMAN
// ================================================================
class SimpleKalmanFilter {
private:
  float _err_measure, _err_estimate, _q;
  float _current_estimate, _last_estimate;
  bool  _initialised;
public:
  SimpleKalmanFilter(float mea_e, float est_e, float q)
    : _err_measure(mea_e), _err_estimate(est_e), _q(q),
      _current_estimate(0), _last_estimate(0), _initialised(false) {}

  float updateEstimate(float mea) {
    if (isnan(mea) || isinf(mea)) return _last_estimate;

    if (!_initialised) {
      _current_estimate = _last_estimate = mea;
      _initialised = true;
      return mea;
    }

    float gain = _err_estimate / (_err_estimate + _err_measure);
    gain = constrain(gain, 0.01f, 0.99f);

    _current_estimate = _last_estimate + gain * (mea - _last_estimate);

    float innovation = fabs(mea - _last_estimate);
    float adaptiveQ  = _q + innovation * 0.001f;
    _err_estimate = (1.0f - gain) * _err_estimate
                    + fabs(_last_estimate - _current_estimate) * adaptiveQ;
    _err_estimate = max(_err_estimate, 0.001f);

    _last_estimate = _current_estimate;
    return _current_estimate;
  }

  void reset() {
    _current_estimate = _last_estimate = 0;
    _err_estimate = _err_measure;
    _initialised  = false;
  }
};

SimpleKalmanFilter kf_ax(2, 2, 0.02), kf_ay(2, 2, 0.02), kf_az(2, 2, 0.02);
SimpleKalmanFilter kf_gx(0.5, 0.5, 0.015), kf_gy(0.5, 0.5, 0.015), kf_gz(0.5, 0.5, 0.015);

// ================================================================
// 5. BUZZER CONTROL (PWM THROTTLE) - GIẢM DÒNG ĐIỆN ĐỈNH
// ================================================================
void setBuzzer(bool state) {
  if (state) {
    #if defined(ESP_ARDUINO_VERSION_MAJOR) && ESP_ARDUINO_VERSION_MAJOR >= 3
      ledcWrite(BUZZER_PIN, BUZZER_DUTY_ON); // ESP32 Core 3.x
    #else
      ledcWrite(0, BUZZER_DUTY_ON);          // ESP32 Core 2.x 
    #endif
  } else {
    #if defined(ESP_ARDUINO_VERSION_MAJOR) && ESP_ARDUINO_VERSION_MAJOR >= 3
      ledcWrite(BUZZER_PIN, 0);
    #else
      ledcWrite(0, 0);
    #endif
  }
}

void buzzerOn()  {
  beepRunning = false;
  beepPinHigh = false;
  setBuzzer(true);
  buzzerActive = true;
}

void buzzerOff() {
  setBuzzer(false);
  buzzerActive = false;
}

void startBeepPattern(int count, int onMs, int offMs) {
  if (count <= 0 || onMs <= 0) return;
  if (buzzerActive) return; 

  beepRunning      = true;
  beepPinHigh      = true;
  beepRemainingOn  = count;
  beepOnMs         = onMs;
  beepOffMs        = max(offMs, 0);
  beepNextToggleAt = millis() + (unsigned long)beepOnMs;
  setBuzzer(true);
}

void updateBeepPattern() {
  if (buzzerActive || !beepRunning) return;

  unsigned long now = millis();
  if (now < beepNextToggleAt) return;

  if (beepPinHigh) {
    setBuzzer(false);
    beepPinHigh = false;
    beepRemainingOn--;

    if (beepRemainingOn <= 0) {
      beepRunning = false;
      return;
    }
    beepNextToggleAt = now + (unsigned long)beepOffMs;
  } else {
    setBuzzer(true);
    beepPinHigh = true;
    beepNextToggleAt = now + (unsigned long)beepOnMs;
  }
}

void beepSilenced()   { startBeepPattern(2, 50, 80);  }  
void beepStopped()    { startBeepPattern(1, 400, 0);  }  
void beepStarted()    { startBeepPattern(2, 120, 100); } 

// ================================================================
// 6. FSM
// ================================================================
void updateFSM(float magnitude) {
  unsigned long now = millis();

  switch (fallState) {
    case FSM_MONITORING:
      if (magnitude < FREE_FALL_THRESH) {
        freefallSampleCount++;
        if (freefallSampleCount >= MIN_FREEFALL_SAMPLES) {
          fallState       = FSM_FREEFALL;
          freefallStartMs = now;
          Serial.println("⚡ FSM → FREE-FALL");
        }
      } else {
        freefallSampleCount = 0;
      }
      break;

    case FSM_FREEFALL:
      if (magnitude > IMPACT_THRESH) {
        fallState        = FSM_IMPACT;
        impactDetectedMs = now;
        postFallBufIdx   = 0;
        Serial.printf("💥 FSM → IMPACT (%.1f m/s²)\n", magnitude);
      } else if (now - freefallStartMs > IMPACT_WINDOW_MS) {
        fallState           = FSM_MONITORING;
        freefallSampleCount = 0;
      }
      break;

    case FSM_IMPACT:
      if (postFallBufIdx < 200) {
        postFallMagBuf[postFallBufIdx++] = magnitude;
      }
      
      if (now - impactDetectedMs >= POST_FALL_CHECK_MS) {
        int n = min(postFallBufIdx, 200);
        int startIdx = 25; 
        int sampleCount = n - startIdx;

        if (sampleCount > 10) {
          float sum = 0, sumSq = 0;
          for (int i = startIdx; i < n; i++) {
            sum += postFallMagBuf[i];
            sumSq += postFallMagBuf[i] * postFallMagBuf[i];
          }
          float mean = sum / sampleCount;
          float var = (sumSq / sampleCount) - (mean * mean);

          if (var < POST_FALL_VARIANCE) {
            fallState = FSM_FALL_CONFIRMED;
            alertStartMs = now;
            fallDetectedFlag = true;
            buzzerOn();
            Serial.printf("🚨 CHỐT: NGÃ THẬT (Variance: %.2f)\n", var);
          } else {
            fallState = FSM_MONITORING;
            freefallSampleCount = 0;
            Serial.printf("↩ RESET: Có cử động (Variance: %.2f)\n", var);
          }
        }
      }

      if (now - impactDetectedMs > FSM_TIMEOUT_MS) {
        fallState           = FSM_MONITORING;
        freefallSampleCount = 0;
      }
      break;

    case FSM_FALL_CONFIRMED:
      if (now - alertStartMs >= ALERT_DURATION_MS) {
        buzzerOff();
        fallState           = FSM_MONITORING;
        freefallSampleCount = 0;
        fallDetectedFlag    = false;
        Serial.println("🔕 FSM → Alert ended");
      }
      break;
  }
}

// ================================================================
// 7. CoAP HELPERS
// ================================================================
bool resolveServerIP() {
  if (serverIPResolved) return true;
  if (WiFi.status() != WL_CONNECTED) return false;
  if (serverIP.fromString(serverHost)) {
    serverIPResolved = true;
    return true;
  }
  serverIPResolved = WiFi.hostByName(serverHost, serverIP);
  return serverIPResolved;
}

bool appendBytes(uint8_t* buffer, size_t capacity, size_t& offset, const void* data, size_t len) {
  if (offset + len > capacity) return false;
  memcpy(buffer + offset, data, len);
  offset += len;
  return true;
}

bool appendU16LE(uint8_t* buffer, size_t capacity, size_t& offset, uint16_t value) {
  uint8_t raw[2] = { (uint8_t)(value & 0xFF), (uint8_t)((value >> 8) & 0xFF) };
  return appendBytes(buffer, capacity, offset, raw, sizeof(raw));
}

bool appendU32LE(uint8_t* buffer, size_t capacity, size_t& offset, uint32_t value) {
  uint8_t raw[4] = {
    (uint8_t)(value & 0xFF),
    (uint8_t)((value >> 8) & 0xFF),
    (uint8_t)((value >> 16) & 0xFF),
    (uint8_t)((value >> 24) & 0xFF)
  };
  return appendBytes(buffer, capacity, offset, raw, sizeof(raw));
}

bool appendI16LE(uint8_t* buffer, size_t capacity, size_t& offset, int16_t value) {
  uint8_t raw[2] = { (uint8_t)(value & 0xFF), (uint8_t)((value >> 8) & 0xFF) };
  return appendBytes(buffer, capacity, offset, raw, sizeof(raw));
}

bool appendFloatLE(uint8_t* buffer, size_t capacity, size_t& offset, float value) {
  union { float f; uint8_t b[4]; } raw;
  raw.f = value;
  return appendBytes(buffer, capacity, offset, raw.b, sizeof(raw.b));
}

int16_t quantizeToI16(float value, float scale) {
  float scaled = value * scale;
  if (scaled > 32767.0f) scaled = 32767.0f;
  if (scaled < -32768.0f) scaled = -32768.0f;
  return (int16_t)lroundf(scaled);
}

size_t buildSensorPayloadLegacy(uint8_t* payload, size_t capacity) {
  size_t offset = 0;
  uint8_t version = SENSOR_PACKET_VERSION_LEGACY;
  uint8_t flags = currentWindow.fall_detected ? 0x01 : 0x00;
  uint8_t fsmState = (uint8_t)currentWindow.fsm_state;
  uint8_t reserved = 0;
  uint16_t windowSize = WINDOW_SIZE;
  uint16_t sampleInterval = SAMPLE_INTERVAL;
  uint32_t windowStartMs = (uint32_t)max(0L, lroundf(currentWindow.samples[0].timestamp * 1000.0f));
  uint16_t bpm = (uint16_t)max(currentWindow.bpm, 0);
  uint32_t ir = (uint32_t)max(currentWindow.ir, 0L);

  if (!appendBytes(payload, capacity, offset, &version, 1)) return 0;
  if (!appendBytes(payload, capacity, offset, &flags, 1)) return 0;
  if (!appendBytes(payload, capacity, offset, &fsmState, 1)) return 0;
  if (!appendBytes(payload, capacity, offset, &reserved, 1)) return 0;
  if (!appendU16LE(payload, capacity, offset, windowSize)) return 0;
  if (!appendU16LE(payload, capacity, offset, sampleInterval)) return 0;
  if (!appendU32LE(payload, capacity, offset, windowStartMs)) return 0;
  if (!appendU16LE(payload, capacity, offset, bpm)) return 0;
  if (!appendU32LE(payload, capacity, offset, ir)) return 0;
  if (!appendFloatLE(payload, capacity, offset, currentWindow.mag_avg)) return 0;
  if (!appendFloatLE(payload, capacity, offset, currentWindow.sma)) return 0;
  if (!appendFloatLE(payload, capacity, offset, currentWindow.max_a)) return 0;
  if (!appendFloatLE(payload, capacity, offset, currentWindow.max_g)) return 0;
  if (!appendFloatLE(payload, capacity, offset, currentWindow.std_accel)) return 0;
  if (!appendFloatLE(payload, capacity, offset, currentWindow.jerk_peak)) return 0;

  for (uint16_t i = 0; i < WINDOW_SIZE; i++) {
    const SensorSample& sample = currentWindow.samples[i];
    if (!appendI16LE(payload, capacity, offset, quantizeToI16(sample.ax, ACCEL_SCALE_LEGACY))) return 0;
    if (!appendI16LE(payload, capacity, offset, quantizeToI16(sample.ay, ACCEL_SCALE_LEGACY))) return 0;
    if (!appendI16LE(payload, capacity, offset, quantizeToI16(sample.az, ACCEL_SCALE_LEGACY))) return 0;
    if (!appendI16LE(payload, capacity, offset, quantizeToI16(sample.gx, GYRO_SCALE_LEGACY))) return 0;
    if (!appendI16LE(payload, capacity, offset, quantizeToI16(sample.gy, GYRO_SCALE_LEGACY))) return 0;
    if (!appendI16LE(payload, capacity, offset, quantizeToI16(sample.gz, GYRO_SCALE_LEGACY))) return 0;
  }

  return offset;
}

size_t buildSensorPayloadChunk(uint8_t* payload, size_t capacity,
                               uint8_t chunkIndex, uint8_t totalChunks,
                               uint8_t chunkSamples, uint16_t startIndex) {
  size_t offset = 0;
  uint8_t version = 2;
  uint8_t flags = currentWindow.fall_detected ? 0x01 : 0x00;
  uint8_t fsmState = (uint8_t)currentWindow.fsm_state;
  uint8_t reserved = 0;
  uint32_t windowStartMs = (uint32_t)max(0L, lroundf(currentWindow.samples[0].timestamp * 1000.0f));
  uint16_t bpm = (uint16_t)max(currentWindow.bpm, 0);
  uint32_t ir = (uint32_t)max(currentWindow.ir, 0L);

  if (!appendBytes(payload, capacity, offset, &version, 1)) return 0;
  if (!appendBytes(payload, capacity, offset, &flags, 1)) return 0;
  if (!appendBytes(payload, capacity, offset, &fsmState, 1)) return 0;
  if (!appendBytes(payload, capacity, offset, &reserved, 1)) return 0;
  if (!appendU16LE(payload, capacity, offset, WINDOW_SIZE)) return 0;
  if (!appendBytes(payload, capacity, offset, &chunkIndex, 1)) return 0;
  if (!appendBytes(payload, capacity, offset, &totalChunks, 1)) return 0;
  if (!appendBytes(payload, capacity, offset, &chunkSamples, 1)) return 0;
  if (!appendBytes(payload, capacity, offset, &reserved, 1)) return 0;
  if (!appendU16LE(payload, capacity, offset, SAMPLE_INTERVAL)) return 0;
  if (!appendU32LE(payload, capacity, offset, windowStartMs)) return 0;
  if (!appendU16LE(payload, capacity, offset, bpm)) return 0;
  if (!appendU32LE(payload, capacity, offset, ir)) return 0;
  if (!appendFloatLE(payload, capacity, offset, currentWindow.mag_avg)) return 0;
  if (!appendFloatLE(payload, capacity, offset, currentWindow.sma)) return 0;
  if (!appendFloatLE(payload, capacity, offset, currentWindow.max_a)) return 0;
  if (!appendFloatLE(payload, capacity, offset, currentWindow.max_g)) return 0;
  if (!appendFloatLE(payload, capacity, offset, currentWindow.std_accel)) return 0;
  if (!appendFloatLE(payload, capacity, offset, currentWindow.jerk_peak)) return 0;

  for (uint8_t localIndex = 0; localIndex < chunkSamples; localIndex++) {
    const SensorSample& sample = currentWindow.samples[startIndex + localIndex];
    if (!appendFloatLE(payload, capacity, offset, sample.timestamp)) return 0;
    if (!appendFloatLE(payload, capacity, offset, sample.ax)) return 0;
    if (!appendFloatLE(payload, capacity, offset, sample.ay)) return 0;
    if (!appendFloatLE(payload, capacity, offset, sample.az)) return 0;
    if (!appendFloatLE(payload, capacity, offset, sample.gx)) return 0;
    if (!appendFloatLE(payload, capacity, offset, sample.gy)) return 0;
    if (!appendFloatLE(payload, capacity, offset, sample.gz)) return 0;
  }

  return offset;
}

size_t buildCoapPostPacket(uint8_t* packet, size_t capacity, const char* path,
                           const uint8_t* payload, size_t payloadLen,
                           uint16_t messageId, uint16_t token,
                           bool confirmable) {
  size_t offset = 0;
  if (capacity < 6) return 0;
  packet[offset++] = confirmable ? 0x42 : 0x52;
  packet[offset++] = 0x02;
  packet[offset++] = (uint8_t)(messageId >> 8);
  packet[offset++] = (uint8_t)(messageId & 0xFF);
  packet[offset++] = (uint8_t)(token >> 8);
  packet[offset++] = (uint8_t)(token & 0xFF);

  uint16_t lastOption = 0;
  const char* segment = path;
  while (*segment) {
    while (*segment == '/') segment++;
    if (!*segment) break;
    const char* end = segment;
    while (*end && *end != '/') end++;
    uint8_t len = (uint8_t)(end - segment);
    uint8_t delta = (lastOption == 0) ? 11 : 0;
    if (len >= 13 || offset + 1 + len > capacity) return 0;
    packet[offset++] = (uint8_t)((delta << 4) | len);
    memcpy(packet + offset, segment, len);
    offset += len;
    lastOption = 11;
    segment = end;
  }

  if (payloadLen > 0) {
    if (offset + 1 + payloadLen > capacity) return 0;
    packet[offset++] = 0xFF;
    memcpy(packet + offset, payload, payloadLen);
    offset += payloadLen;
  }
  return offset;
}

bool isSuccessCoapReply(const uint8_t* packet, size_t len, uint16_t messageId, uint16_t token) {
  if (len < 6) return false;
  uint8_t version = packet[0] >> 6;
  uint8_t type = (packet[0] >> 4) & 0x03;
  uint8_t tokenLen = packet[0] & 0x0F;
  uint8_t code = packet[1];
  uint16_t responseId = (uint16_t(packet[2]) << 8) | uint16_t(packet[3]);
  uint16_t responseToken = (uint16_t(packet[4]) << 8) | uint16_t(packet[5]);
  if (version != 1 || tokenLen != 2) return false;
  if (type != 1 && type != 2) return false;
  if (responseId != messageId || responseToken != token) return false;
  return code >= 64 && code < 96;
}

bool sendCoapPost(const char* path, const uint8_t* payload, size_t payloadLen,
                  uint16_t timeoutMs, bool waitAck = true) {
  if (WiFi.status() != WL_CONNECTED) return false;
  if (!resolveServerIP()) return false;

  bool wasBuzzerOn = (buzzerActive || beepPinHigh);
  if (wasBuzzerOn) setBuzzer(false);

  bool success = false;
  const int maxAttempts = waitAck ? COAP_MAX_RETRIES : 1;
  for (int attempt = 0; attempt < maxAttempts && !success; attempt++) {
    uint16_t messageId = nextCoapMessageId++;
    uint16_t token = nextCoapToken++;
    size_t packetLen = buildCoapPostPacket(coapPacketBuf, sizeof(coapPacketBuf),
                                           path, payload, payloadLen, messageId, token, waitAck);
    if (packetLen == 0) break;
    if (!coapUdp.beginPacket(serverIP, serverPort)) continue;
    coapUdp.write(coapPacketBuf, packetLen);
    if (!coapUdp.endPacket()) continue;

    if (!waitAck) {
      success = true;
      break;
    }

    unsigned long startedAt = millis();
    while (millis() - startedAt < timeoutMs) {
      int incoming = coapUdp.parsePacket();
      if (incoming > 0) {
        int toRead = min(incoming, (int)sizeof(coapReplyBuf));
        int readLen = coapUdp.read(coapReplyBuf, toRead);
        if (readLen > 0 && isSuccessCoapReply(coapReplyBuf, (size_t)readLen, messageId, token)) {
          success = true;
          break;
        }
      }
      delay(5);
      yield();
    }
  }

  if (wasBuzzerOn) setBuzzer(true);
  return success;
}

// ================================================================
// 8. DEPLOY MODE: CHỈ GỬI DATA QUA CoAP (KHÔNG SESSION)
// ================================================================
bool shouldSendBatchByFSM(int fsmState) {
  return (fsmState == FSM_IMPACT);
}

void sendWindow() {
  if (!shouldSendBatchByFSM(currentWindow.fsm_state)) {
    return;
  }

  if (WiFi.status() != WL_CONNECTED) {
    Serial.printf("📡 Offline | Mag:%.1f | FSM:%d\n",
                  currentWindow.mag_avg, currentWindow.fsm_state);
    return;
  }

  size_t payloadLen = buildSensorPayloadLegacy(sensorPayloadBuf, sizeof(sensorPayloadBuf));
  if (payloadLen == 0) {
    Serial.printf("✗ Payload build failed | Heap:%u\n", ESP.getFreeHeap());
    return;
  }

  if (sendCoapPost(batchPath, sensorPayloadBuf, payloadLen, COAP_ACK_TIMEOUT_MS, false)) {
    Serial.printf("✓ Sent | CoAP 1 packet | %uB | Mag:%.1f | Heap:%u\n",
                  (unsigned)payloadLen, currentWindow.mag_avg, ESP.getFreeHeap());
  } else {
    Serial.printf("✗ CoAP send failed | Mag:%.1f | Heap:%u\n",
                  currentWindow.mag_avg, ESP.getFreeHeap());
  }
}

// ================================================================
// 9. SETUP
// ================================================================
void setup() {
  randomSeed((uint32_t)esp_random());

  Serial.begin(115200);
  pinMode(BUTTON_PIN, INPUT_PULLUP);
  
  // KHỞI TẠO BĂM XUNG PWM CHO CÒI (CHỐNG SẬP NGUỒN)
  pinMode(BUZZER_PIN, OUTPUT);
  #if defined(ESP_ARDUINO_VERSION_MAJOR) && ESP_ARDUINO_VERSION_MAJOR >= 3
    ledcAttach(BUZZER_PIN, 2000, 8); // Dành cho Arduino ESP32 Core 3.x
    ledcWrite(BUZZER_PIN, 0);
  #else
    ledcSetup(0, 2000, 8);           // Dành cho Arduino ESP32 Core 2.x
    ledcAttachPin(BUZZER_PIN, 0);
    ledcWrite(0, 0);
  #endif

  WiFi.begin(ssid, password);
  Serial.print("WiFi connecting");
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 20) { delay(500); Serial.print("."); attempts++; }

  #if defined(WIFI_POWER_8_5dBm)
    WiFi.setTxPower(WIFI_POWER_8_5dBm); // Giảm đỉnh dòng khi phát WiFi để chạy pin ổn định hơn
  #endif

  coapUdp.begin(COAP_LOCAL_PORT);
  nextCoapMessageId = (uint16_t)esp_random();
  nextCoapToken = (uint16_t)(esp_random() & 0xFFFF);

  if (WiFi.status() == WL_CONNECTED) {
    Serial.printf("\n✓ WiFi OK – IP: %s\n", WiFi.localIP().toString().c_str());
    if (resolveServerIP())
      Serial.printf("✓ CoAP target – %s:%u\n", serverIP.toString().c_str(), serverPort);
    else
      Serial.println("⚠ CoAP target unresolved");
  } else {
    Serial.println("\n⚠ WiFi FAIL – running OFFLINE");
  }

  Wire.begin(I2C_SDA, I2C_SCL);
  Wire.setClock(400000);
  Wire.setTimeout(10);

  max30102_detected = max30102_init();
  if (!max30102_detected)
    Serial.println("⚠ MAX30102 not found – BPM full random");

  if (mpu.begin()) {
    mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
    mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);
    Serial.println("✓ MPU6050 Ready");
  } else {
    Serial.println("✗ MPU6050 FAILED – halting");
    while (1) delay(1000);
  }

  systemState = false;
  Serial.println(">>> DEPLOY MODE – WAITING FOR SHORT PRESS TO START STREAM");

  Serial.println("\n╔═══════════════════════════════════════╗");
  Serial.println("║  FALL DETECTION DEPLOY – SINGLE CORE ║");
  Serial.println("╠═══════════════════════════════════════╣");
  Serial.println("║  Short press : START STREAM            ║");
  Serial.println("║  Long press  : STOP STREAM             ║");
  Serial.println("║  Transport   : CoAP / UDP             ║");
  Serial.println("║  Buzzer pin  : GPIO 18 (PWM MODE)     ║");
  Serial.println("╚═══════════════════════════════════════╝");
}

// ================================================================
// 10. MAIN LOOP
// ================================================================
void loop() {
  handleButton();
  updateBeepPattern();
  yield();

  if (!systemState) return;

  loopCount++;
  unsigned long now = millis();

  if (buzzerActive && (now - alertStartMs >= (ALERT_DURATION_MS + 200UL))) {
    buzzerOff();
    fallState           = FSM_MONITORING;
    freefallSampleCount = 0;
    fallDetectedFlag    = false;
    Serial.println("🔕 FALL alarm failsafe stop");
  }

  if (now - lastLoopReport >= 5000) {
    Serial.printf("[DBG] loops/5s=%lu heap=%u wifi=%d\n",
                  loopCount, ESP.getFreeHeap(), WiFi.status() == WL_CONNECTED);
    loopCount = 0;
    lastLoopReport = now;
  }

  if (now - lastHrCheck >= 2000) {
    lastHrCheck = now;

    // Đọc IR thật từ MAX30102 nếu có
    uint32_t realIR = 0;
    if (max30102_detected) {
      realIR = max30102_readIR();
    }

    bool fingerOn = max30102_detected && (realIR > IR_FINGER_THRESH);
    int roll = random(100);

    if (fingerOn) {
      // Có ngón tay → random BPM thực tế
      irValue = (long)realIR;
      if (roll < 6) {
        // 6% nhiễu: BPM bất thường
        beatAvg = (random(2) == 0) ? random(25, 40) : random(165, 191);
      } else {
        beatAvg = random(80, 121);
      }
    } else {
      // Không có ngón tay → BPM = 0
      irValue = max30102_detected ? (long)realIR : random(800, 2500);
      beatAvg = 0;
    }
  }

  if (now - lastSampleTime < SAMPLE_INTERVAL) return;
  lastSampleTime = now;

  sensors_event_t a, g, temp;
  mpu.getEvent(&a, &g, &temp);

  float raw_ax = safeFloat(a.acceleration.x);
  float raw_ay = safeFloat(a.acceleration.y);
  float raw_az = safeFloat(a.acceleration.z);
  float raw_mag = sqrtf(raw_ax * raw_ax + raw_ay * raw_ay + raw_az * raw_az);

  float ax = kf_ax.updateEstimate(raw_ax);
  float ay = kf_ay.updateEstimate(raw_ay);
  float az = kf_az.updateEstimate(raw_az);
  float gx = kf_gx.updateEstimate(safeGyro(g.gyro.x));
  float gy = kf_gy.updateEstimate(safeGyro(g.gyro.y));
  float gz = kf_gz.updateEstimate(safeGyro(g.gyro.z));

  static unsigned long lastMagPrint = 0;
  if (now - lastMagPrint >= 200) {
    Serial.printf("[MAG] raw=%.1f filt=%.1f | FSM:%d\n", raw_mag, sqrtf(ax*ax+ay*ay+az*az), (int)fallState);
    lastMagPrint = now;
  }

  updateFSM(raw_mag);

  SensorSample& s = currentWindow.samples[bufferIndex];
  s.timestamp = (now - startTime) / 1000.0f;
  s.ax = ax;  s.ay = ay;  s.az = az;
  s.gx = gx;  s.gy = gy;  s.gz = gz;
  bufferIndex++;

  if (bufferIndex < WINDOW_SIZE) return;

  float m_sum = 0, s_sum = 0, m_peak_a = 0, m_peak_g = 0;
  static float magnitudes[WINDOW_SIZE];  
  float jerk_max = 0;

  for (int i = 0; i < WINDOW_SIZE; i++) {
    float mx = sqrtf(sq(currentWindow.samples[i].ax) +
                     sq(currentWindow.samples[i].ay) +
                     sq(currentWindow.samples[i].az));
    magnitudes[i] = mx;
    m_sum += mx;
    if (mx > m_peak_a) m_peak_a = mx;

    float gm = sqrtf(sq(currentWindow.samples[i].gx) +
                     sq(currentWindow.samples[i].gy) +
                     sq(currentWindow.samples[i].gz));
    if (gm > m_peak_g) m_peak_g = gm;

    s_sum += fabsf(currentWindow.samples[i].ax)
           + fabsf(currentWindow.samples[i].ay)
           + fabsf(currentWindow.samples[i].az);

    if (i > 0) {
      float jerk = fabsf(magnitudes[i] - magnitudes[i - 1]) / (SAMPLE_INTERVAL / 1000.0f);
      if (jerk > jerk_max) jerk_max = jerk;
    }
  }

  float mag_avg = m_sum / WINDOW_SIZE;

  float var_sum = 0;
  for (int i = 0; i < WINDOW_SIZE; i++) {
    float d = magnitudes[i] - mag_avg;
    var_sum += d * d;
  }

  currentWindow.mag_avg   = mag_avg;
  currentWindow.sma       = s_sum / (3.0f * WINDOW_SIZE);
  currentWindow.max_a     = m_peak_a;
  currentWindow.max_g     = m_peak_g;
  currentWindow.std_accel = sqrtf(var_sum / WINDOW_SIZE);
  currentWindow.jerk_peak = jerk_max;
  currentWindow.bpm       = beatAvg;
  currentWindow.ir        = irValue;
  currentWindow.fsm_state     = (int)fallState;
  currentWindow.fall_detected = fallDetectedFlag;

  if (systemState && shouldSendBatchByFSM(currentWindow.fsm_state)) sendWindow();
  bufferIndex = 0;
}

// ================================================================
// 11. BUTTON HANDLER 
// ================================================================
void handleButton() {
  bool lockShortPress = (fallState == FSM_FALL_CONFIRMED || buzzerActive);

  static int buttonState = HIGH;
  static int lastButtonState = HIGH;
  static unsigned long lastDebounceTime = 0;
  static unsigned long pressTime = 0;
  static bool isPressing = false;
  static bool longPressHandled = false;

  int reading = digitalRead(BUTTON_PIN);

  if (reading != lastButtonState) {
    lastDebounceTime = millis();
  }

  if ((millis() - lastDebounceTime) > debounceMs) {
    if (reading != buttonState) {
      buttonState = reading;

      if (buttonState == LOW) {
        pressTime = millis();
        isPressing = true;
        longPressHandled = false;
      } 
      else {
        isPressing = false;
        
        if (!longPressHandled && !lockShortPress) {
          if (!systemState) {
            systemState  = true;
            startTime    = millis();
            bufferIndex  = 0;
            currentWindow.ready = false;
            kf_ax.reset(); kf_ay.reset(); kf_az.reset();
            kf_gx.reset(); kf_gy.reset(); kf_gz.reset();
            fallState = FSM_MONITORING; freefallSampleCount = 0; fallDetectedFlag = false;

            beepStarted();
            Serial.println("\n>>> STREAM STARTED");
          } else {
            Serial.println("\n>>> STREAM ALREADY RUNNING");
          }
        }
      }
    }
  }

  if (isPressing && !longPressHandled && (millis() - pressTime >= longPressMs)) {
    longPressHandled = true;

    if (systemState) {
      systemState = false;
      buzzerOff();
      fallState           = FSM_MONITORING;
      freefallSampleCount = 0;
      fallDetectedFlag    = false;

      beepStopped();           
      Serial.println("\n>>> STREAM STOPPED");
    }
  }

  lastButtonState = reading;
}
