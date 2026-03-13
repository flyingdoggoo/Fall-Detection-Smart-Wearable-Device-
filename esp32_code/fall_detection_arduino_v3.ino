#include <WiFi.h>
#include <HTTPClient.h>
#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <math.h>
#include <ArduinoJson.h>

// ================================================================
// 1. CẤU HÌNH WIFI & SERVER
// ================================================================
// const char* ssid       = "Giat Ui Tigon 1";
// const char* password   = "789789789";
// const char* serverBase = "http://192.168.1.4:3000";
const char* ssid       = "ITF - Da Nang";
const char* password   = "888888888";
const char* serverBase = "http://172.20.10.2:3000";
#define I2C_SDA         4
#define I2C_SCL         5
#define BUTTON_PIN      9
#define BUZZER_PIN      18      // Chân buzzer cảnh báo ngã (OFFLINE)
#define SAMPLE_INTERVAL 20      // 50 Hz = 20 ms
#define WINDOW_SIZE     100     // 100 mẫu = 2 giây / window

// ================================================================
// FSM – NGƯỠNG PHÁT HIỆN NGÃ OFFLINE
// Các giá trị này phù hợp cho thiết bị đeo CỔ TAY, ±8G range.
// ================================================================
#define FREE_FALL_THRESH      5.6f   
#define IMPACT_THRESH         11.0f  
#define MIN_FREEFALL_SAMPLES    3    
#define IMPACT_WINDOW_MS      800    
#define POST_FALL_VARIANCE   10.0f   
#define POST_FALL_CHECK_MS   2000    
#define ALERT_DURATION_MS    2000    
#define FSM_TIMEOUT_MS       2800   
#define IMPACT_DELTA_THRESH   5.0f   // Impact phải chênh đủ lớn so với mức free-fall thấp nhất
#define FSM_REARM_DELAY_MS   1200    // Sau reject thì chờ ngắn trước khi cho FSM bắt lại

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
  // ---- Features gốc ----
  float mag_avg;     // mean(magnitude)
  float sma;         // Signal Magnitude Area
  float max_a;       // peak accel magnitude
  float max_g;       // peak gyro magnitude
  // ---- Features mới (v3) ----
  float std_accel;   // Std-dev magnitude  – nhận biết biến đổi mạnh
  float jerk_peak;   // Peak jerk (m/s³)   – đạo hàm gia tốc tức thời
  // ---- FSM info ----
  int   fsm_state;
  bool  fall_detected;
  bool  ready = false;
} currentWindow;

// ================================================================
// FSM STATES
// ================================================================
enum FallState {
  FSM_MONITORING     = 0,  // Giám sát bình thường
  FSM_FREEFALL       = 1,  // Phát hiện trọng lực giảm → rơi tự do
  FSM_IMPACT         = 2,  // Va chạm mạnh sau freefall
  FSM_FALL_CONFIRMED = 3   // Xác nhận ngã → kêu buzzer
};

volatile FallState fallState       = FSM_MONITORING;
unsigned long freefallStartMs      = 0;
int           freefallSampleCount  = 0;
unsigned long impactDetectedMs     = 0;
unsigned long alertStartMs         = 0;
float         postFallMagBuf[150]; // 3 s @ 50 Hz
int           postFallBufIdx       = 0;
bool          buzzerActive         = false;
bool          fallDetectedFlag     = false;
float         freefallMinMag       = 99.0f;
unsigned long fsmRearmUntilMs      = 0;

// ================================================================
// Đối tượng cảm biến
// ================================================================
Adafruit_MPU6050  mpu;

// Heart-Rate (mô phỏng)
int   beatAvg        = 0;
long  irValue        = 0;
unsigned long lastHrCheck = 0;   // Rate-limit MAX30102 reads

// Debug
unsigned long loopCount       = 0;
unsigned long lastLoopReport  = 0;

// Trạng thái hệ thống
bool          systemState    = false;
unsigned long lastSampleTime = 0;
unsigned long startTime      = 0;
int           bufferIndex    = 0;

// Button
unsigned long buttonPressedAt = 0;
const unsigned long longPressMs = 1000;
const unsigned long debounceMs  = 40;

// Beep scheduler (non-blocking, tránh delay gây treo)
bool          beepRunning      = false;
bool          beepPinHigh      = false;
int           beepRemainingOn  = 0;
unsigned long beepNextToggleAt = 0;
int           beepOnMs         = 0;
int           beepOffMs        = 0;

// Endpoints (xây từ serverBase)
String batchEndpoint, sessionNewEndpoint, sessionStopEndpoint;

// ================================================================
// 3. NaN / Inf GUARD
//    Bảo vệ dữ liệu trước khi đưa vào Kalman hoặc tính toán.
// ================================================================
float safeFloat(float v, float fallback = 0.0f) {
  if (isnan(v) || isinf(v)) return fallback;
  return constrain(v, -160.0f, 160.0f);   // ±16 g
}

float safeGyro(float v, float fallback = 0.0f) {
  if (isnan(v) || isinf(v)) return fallback;
  return constrain(v, -35.0f, 35.0f);     // ~2000 °/s
}

// ================================================================
// 4. BỘ LỌC KALMAN CẢI TIẾN
//    • NaN-safe input
//    • First-sample initialisation
//    • Gain clamped [0.01 … 0.99] chống bão hoà
//    • Adaptive Q (tăng khi innovation lớn)
//    • err_estimate sàn 0.001
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

// Accel: mea_e=2, est_e=2, q=0.02 (tăng so với v2 để phản ứng nhanh hơn)
SimpleKalmanFilter kf_ax(2, 2, 0.02), kf_ay(2, 2, 0.02), kf_az(2, 2, 0.02);
// Gyro: nhiễu nhỏ hơn accel
SimpleKalmanFilter kf_gx(0.5, 0.5, 0.015), kf_gy(0.5, 0.5, 0.015), kf_gz(0.5, 0.5, 0.015);

// ================================================================
// 5. BUZZER CONTROL  (active-buzzer hoặc passive đều OK)
// ================================================================
void buzzerOn()  {
  // Alarm còi ưu tiên cao nhất, dừng mọi beep pattern đang chạy.
  beepRunning = false;
  beepPinHigh = false;
  digitalWrite(BUZZER_PIN, HIGH);
  buzzerActive = true;
}

void buzzerOff() {
  digitalWrite(BUZZER_PIN, LOW);
  buzzerActive = false;
}

// ---- Beep patterns: phân biệt các hành động bằng âm thanh ----
void startBeepPattern(int count, int onMs, int offMs) {
  if (count <= 0 || onMs <= 0) return;
  if (buzzerActive) return;  // Không chồng beep thường lên alarm ngã

  beepRunning      = true;
  beepPinHigh      = true;
  beepRemainingOn  = count;
  beepOnMs         = onMs;
  beepOffMs        = max(offMs, 0);
  beepNextToggleAt = millis() + (unsigned long)beepOnMs;
  digitalWrite(BUZZER_PIN, HIGH);
}

void updateBeepPattern() {
  if (buzzerActive) return;
  if (!beepRunning) return;

  unsigned long now = millis();
  if (now < beepNextToggleAt) return;

  if (beepPinHigh) {
    digitalWrite(BUZZER_PIN, LOW);
    beepPinHigh = false;
    beepRemainingOn--;

    if (beepRemainingOn <= 0) {
      beepRunning = false;
      return;
    }
    beepNextToggleAt = now + (unsigned long)beepOffMs;
  } else {
    digitalWrite(BUZZER_PIN, HIGH);
    beepPinHigh = true;
    beepNextToggleAt = now + (unsigned long)beepOnMs;
  }
}
void beepSilenced()   { startBeepPattern(2, 50, 80);  }  // Tắt báo ngã:   🔔🔔
void beepNewSession() { startBeepPattern(3, 50, 80);  }  // New session:    🔔🔔🔔
void beepStopped()    { startBeepPattern(1, 400, 0);  }  // Stop hệ thống:  🔔———
void beepStarted()    { startBeepPattern(2, 120, 100); } // Start hệ thống: 🔔-🔔

// ================================================================
// 6. FSM – PHÁT HIỆN NGÃ OFFLINE
//
// MONITORING ──(free-fall)──► FREEFALL ──(impact)──► IMPACT
//      ▲                         │ timeout              │
//      └─────────────────────────┘            (bất động) │
//      ▲                                                 ▼
//      │                                        FALL_CONFIRMED
//      │                                           │ timeout
//      └───────────────────────────────────────────┘
// ================================================================
void updateFSM(float magnitude) {
  unsigned long now = millis();

  switch (fallState) {
    // ---- MONITORING: Theo dõi gia tốc liên tục ----
    case FSM_MONITORING:
      if (now < fsmRearmUntilMs) {
        freefallSampleCount = 0;
        break;
      }

      if (magnitude < FREE_FALL_THRESH) {
        freefallSampleCount++;
        if (freefallSampleCount >= MIN_FREEFALL_SAMPLES) {
          fallState       = FSM_FREEFALL;
          freefallStartMs = now;
          freefallMinMag  = magnitude;
          Serial.println("⚡ FSM → FREE-FALL");
        }
      } else {
        freefallSampleCount = 0;
      }
      break;

    // ---- FREEFALL: Chờ va chạm ----
    case FSM_FREEFALL:
      if (magnitude < freefallMinMag) {
        freefallMinMag = magnitude;
      }

      if (magnitude > IMPACT_THRESH &&
          (magnitude - freefallMinMag) >= IMPACT_DELTA_THRESH) {
        fallState        = FSM_IMPACT;
        impactDetectedMs = now;
        postFallBufIdx   = 0;
        Serial.printf("💥 FSM → IMPACT (%.1f m/s²)\n", magnitude);
      } else if (now - freefallStartMs > IMPACT_WINDOW_MS) {
        fallState           = FSM_MONITORING;
        freefallSampleCount = 0;
        fsmRearmUntilMs     = now + FSM_REARM_DELAY_MS;
      }
      break;

    // ---- IMPACT: Kiểm tra bất động sau va chạm ----
    case FSM_IMPACT:
      if (postFallBufIdx < 150) {
        postFallMagBuf[postFallBufIdx++] = magnitude;
      }

      if (now - impactDetectedMs >= POST_FALL_CHECK_MS) {
        // Chỉ tính variance trên NỬA SAU buffer (bỏ phần settling sau impact)
        int n = min(postFallBufIdx, 150);
        int halfStart = n / 2;  // Bắt đầu từ giữa buffer
        int halfN = n - halfStart;
        if (halfN > 2) {
          float sum = 0, sumSq = 0;
          for (int i = halfStart; i < n; i++) { sum += postFallMagBuf[i]; sumSq += postFallMagBuf[i] * postFallMagBuf[i]; }
          float mean = sum / halfN;
          float var  = (sumSq / halfN) - (mean * mean);

          if (var < POST_FALL_VARIANCE) {
            // Nằm yên → xác nhận ngã
            fallState        = FSM_FALL_CONFIRMED;
            alertStartMs     = now;
            fallDetectedFlag = true;
            buzzerOn();
            Serial.printf("🚨 FSM → FALL CONFIRMED (var=%.2f)\n", var);
          } else {
            fallState           = FSM_MONITORING;
            freefallSampleCount = 0;
            fsmRearmUntilMs     = now + FSM_REARM_DELAY_MS;
            Serial.printf("↩ FSM → MONITORING (var=%.2f – active)\n", var);
          }
        } else {
          fallState = FSM_MONITORING;
          freefallSampleCount = 0;
          fsmRearmUntilMs = now + FSM_REARM_DELAY_MS;
          Serial.println("↩ FSM → MONITORING (not enough samples)");
        }
      }

      if (now - impactDetectedMs > FSM_TIMEOUT_MS) {
        fallState           = FSM_MONITORING;
        freefallSampleCount = 0;
        fsmRearmUntilMs     = now + FSM_REARM_DELAY_MS;
      }
      break;

    // ---- FALL_CONFIRMED: Kêu buzzer ----
    case FSM_FALL_CONFIRMED:
      if (now - alertStartMs >= ALERT_DURATION_MS) {
        buzzerOff();
        fallState           = FSM_MONITORING;
        freefallSampleCount = 0;
        fallDetectedFlag    = false;
        fsmRearmUntilMs     = now + 400;
        Serial.println("🔕 FSM → Alert ended");
      }
      break;
  }
}

// ================================================================
// 7. HTTP POST HELPER
// ================================================================
bool postJson(const String& url, const String& payload) {
  if (WiFi.status() != WL_CONNECTED) return false;
  HTTPClient http;
  http.begin(url);
  http.addHeader("Content-Type", "application/json");
  http.setTimeout(3000);
  int code = http.POST(payload);
  http.end();
  return code > 0;
}

void newSession() {
  if (postJson(sessionNewEndpoint, "{}"))
    Serial.println("✓ Session created (server)");
  else
    Serial.println("⚠ Server unreachable – offline mode");
}

void stopSession() { postJson(sessionStopEndpoint, "{}"); }

// ================================================================
// 8. GỬI DATA TRỰC TIẾP (single-core, không cần task/mutex)
// ================================================================
StaticJsonDocument<16384> gDoc;  // GLOBAL – tránh stack overflow

void sendWindow() {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.printf("📡 Offline | Mag:%.1f | FSM:%d\n",
                  currentWindow.mag_avg, currentWindow.fsm_state);
    return;
  }

  gDoc.clear();
  gDoc["status"]        = "active";
  gDoc["bpm"]           = currentWindow.bpm;
  gDoc["ir_raw"]        = currentWindow.ir;
  gDoc["window_size"]   = WINDOW_SIZE;
  gDoc["sample_rate"]   = 50;
  gDoc["fsm_state"]     = currentWindow.fsm_state;
  gDoc["fall_detected"] = currentWindow.fall_detected;

  JsonObject features = gDoc.createNestedObject("features");
  features["magnitude_avg"] = currentWindow.mag_avg;
  features["sma"]           = currentWindow.sma;
  features["max_accel"]     = currentWindow.max_a;
  features["max_gyro"]      = currentWindow.max_g;
  features["std_accel"]     = currentWindow.std_accel;
  features["jerk_peak"]     = currentWindow.jerk_peak;

  JsonArray accel_data = gDoc.createNestedArray("accel_data");
  JsonArray gyro_data  = gDoc.createNestedArray("gyro_data");

  for (int i = 0; i < WINDOW_SIZE; i++) {
    JsonObject a = accel_data.createNestedObject();
    a["t"] = currentWindow.samples[i].timestamp;
    a["x"] = currentWindow.samples[i].ax;
    a["y"] = currentWindow.samples[i].ay;
    a["z"] = currentWindow.samples[i].az;

    JsonObject g = gyro_data.createNestedObject();
    g["t"] = currentWindow.samples[i].timestamp;
    g["x"] = currentWindow.samples[i].gx;
    g["y"] = currentWindow.samples[i].gy;
    g["z"] = currentWindow.samples[i].gz;
  }

  String jsonStr;
  size_t jsonLen = measureJson(gDoc);
  jsonStr.reserve(jsonLen + 1);
  serializeJson(gDoc, jsonStr);

  HTTPClient http;
  http.begin(batchEndpoint);
  http.addHeader("Content-Type", "application/json");
  http.setTimeout(900);
  int code = http.POST(jsonStr);

  if (code > 0)
    Serial.printf("✓ Sent | HTTP %d | Mag:%.1f | Heap:%u\n",
                  code, currentWindow.mag_avg, ESP.getFreeHeap());
  else
    Serial.printf("✗ HTTP %s | Heap:%u\n",
                  http.errorToString(code).c_str(), ESP.getFreeHeap());
  http.end();
}

// ================================================================
// 9. SETUP
// ================================================================
void setup() {
  randomSeed((uint32_t)esp_random());

  Serial.begin(115200);
  pinMode(BUTTON_PIN, INPUT_PULLUP);
  pinMode(BUZZER_PIN, OUTPUT);
  digitalWrite(BUZZER_PIN, LOW);

  batchEndpoint      = String(serverBase) + "/api/sensor-batch";
  sessionNewEndpoint = String(serverBase) + "/api/session/new";
  sessionStopEndpoint= String(serverBase) + "/api/session/stop";

  // WiFi – thử 10 s, nếu không được → chạy offline
  WiFi.begin(ssid, password);
  Serial.print("WiFi connecting");
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 20) { delay(500); Serial.print("."); attempts++; }

  if (WiFi.status() == WL_CONNECTED)
    Serial.printf("\n✓ WiFi OK – IP: %s\n", WiFi.localIP().toString().c_str());
  else
    Serial.println("\n⚠ WiFi FAIL – running OFFLINE");

  // I²C & cảm biến – timeout 10ms chống treo bus
  Wire.begin(I2C_SDA, I2C_SCL);
  Wire.setClock(400000);
  Wire.setTimeout(10);

  Serial.println("✓ MAX30102 skipped – BPM simulated");

  if (mpu.begin()) {
    mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
    mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);
    Serial.println("✓ MPU6050 Ready");
  } else {
    Serial.println("✗ MPU6050 FAILED – halting");
    while (1) delay(1000);
  }

  // Gửi data trên loop() – single-core, không cần FreeRTOS task

  // Tự động bắt đầu khi boot – không cần bấm nút
  systemState = true;
  startTime   = millis();
  newSession();
  Serial.println(">>> AUTO-START ON BOOT");

  Serial.println("\n╔═══════════════════════════════════════╗");
  Serial.println("║  FALL DETECTION V3 – SINGLE-CORE     ║");
  Serial.println("╠═══════════════════════════════════════╣");
  Serial.println("║  Short press : START / NEW SESSION    ║");
  Serial.println("║  Long press  : STOP                   ║");
  Serial.println("║  Buzzer pin  : GPIO 18                ║");
  Serial.println("╚═══════════════════════════════════════╝");
}

// ================================================================
// 10. MAIN LOOP – SINGLE-CORE (thu thập + gửi tuần tự)
// ================================================================
void loop() {
  handleButton();
  updateBeepPattern();
  yield();

  if (!systemState) return;

  loopCount++;
  unsigned long now = millis();

  // Failsafe: dù FSM lỗi trạng thái, còi alarm vẫn phải tự tắt đúng hạn.
  if (buzzerActive && (now - alertStartMs >= (ALERT_DURATION_MS + 200UL))) {
    buzzerOff();
    fallState           = FSM_MONITORING;
    freefallSampleCount = 0;
    fallDetectedFlag    = false;
    Serial.println("🔕 FALL alarm failsafe stop");
  }

  // ---- Debug: báo cáo mỗi 5 giây ----
  if (now - lastLoopReport >= 5000) {
    Serial.printf("[DBG] loops/5s=%lu heap=%u wifi=%d\n",
                  loopCount, ESP.getFreeHeap(), WiFi.status() == WL_CONNECTED);
    loopCount = 0;
    lastLoopReport = now;
  }

  // ---- Heart Rate: giả lập BPM hoàn toàn (không đọc I2C MAX30102) ----
  if (now - lastHrCheck >= 2000) {
    lastHrCheck = now;
    int roll = random(100);

    // Không detect tay: BPM = 0
    if (roll < 10) {
      irValue = random(800, 2500);
      beatAvg = 0;
    } else {
      // Có tay: chủ yếu 80..120, thỉnh thoảng nhiễu thấp/cao
      irValue = random(45000, 75000);
      if (roll < 16) {
        beatAvg = (random(2) == 0) ? random(25, 40) : random(165, 191);
      } else {
        beatAvg = random(80, 121);
      }
    }
  }

  // ---- IMU @ 50 Hz ----
  if (now - lastSampleTime < SAMPLE_INTERVAL) return;
  lastSampleTime = now;

  sensors_event_t a, g, temp;
  mpu.getEvent(&a, &g, &temp);

  // RAW magnitude (chưa Kalman) – dùng cho FSM vì Kalman san phẳng spike
  float raw_ax = safeFloat(a.acceleration.x);
  float raw_ay = safeFloat(a.acceleration.y);
  float raw_az = safeFloat(a.acceleration.z);
  float raw_mag = sqrtf(raw_ax * raw_ax + raw_ay * raw_ay + raw_az * raw_az);

  // Kalman-filtered – dùng cho data gửi server (mượt, đẹp)
  float ax = kf_ax.updateEstimate(raw_ax);
  float ay = kf_ay.updateEstimate(raw_ay);
  float az = kf_az.updateEstimate(raw_az);
  float gx = kf_gx.updateEstimate(safeGyro(g.gyro.x));
  float gy = kf_gy.updateEstimate(safeGyro(g.gyro.y));
  float gz = kf_gz.updateEstimate(safeGyro(g.gyro.z));

  // Debug magnitude mỗi 500ms
  static unsigned long lastMagPrint = 0;
  if (now - lastMagPrint >= 500) {
    Serial.printf("[MAG] raw=%.1f filt=%.1f | FSM:%d\n", raw_mag, sqrtf(ax*ax+ay*ay+az*az), (int)fallState);
    lastMagPrint = now;
  }

  // FSM dùng RAW magnitude – bắt spike impact chính xác
  updateFSM(raw_mag);

  // ---- Buffer vào window ----
  SensorSample& s = currentWindow.samples[bufferIndex];
  s.timestamp = (now - startTime) / 1000.0f;
  s.ax = ax;  s.ay = ay;  s.az = az;
  s.gx = gx;  s.gy = gy;  s.gz = gz;
  bufferIndex++;

  // ---- Đủ 100 samples → tính features + gửi ----
  if (bufferIndex < WINDOW_SIZE) return;

  float m_sum = 0, s_sum = 0, m_peak_a = 0, m_peak_g = 0;
  static float magnitudes[WINDOW_SIZE];  // static – tránh stack overflow
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

    // Jerk = |Δmag / Δt|
    if (i > 0) {
      float jerk = fabsf(magnitudes[i] - magnitudes[i - 1]) / (SAMPLE_INTERVAL / 1000.0f);
      if (jerk > jerk_max) jerk_max = jerk;
    }
  }

  float mag_avg = m_sum / WINDOW_SIZE;

  // Standard deviation of magnitude
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

  // Gửi data trực tiếp (block ~100-200ms, chấp nhận được)
  sendWindow();
  bufferIndex = 0;
}

// ================================================================
// 11. BUTTON HANDLER
// ================================================================
void handleButton() {
  // Không cho phép nút tắt còi khi đang cảnh báo ngã.
  if (fallState == FSM_FALL_CONFIRMED || buzzerActive) {
    return;
  }

  static bool stableState = HIGH;
  static bool rawLast     = HIGH;
  static unsigned long lastChangeAt = 0;
  bool rawReading = digitalRead(BUTTON_PIN);

  if (rawReading != rawLast) {
    rawLast = rawReading;
    lastChangeAt = millis();
  }

  if (millis() - lastChangeAt < debounceMs) return;
  bool reading = rawReading;

  // Nhấn xuống
  if (stableState == HIGH && reading == LOW) {
    buttonPressedAt = millis();
  }

  // Nhả ra
  if (stableState == LOW && reading == HIGH) {
    unsigned long dur = millis() - buttonPressedAt;

    if (dur >= longPressMs) {
      // ---- Long press: STOP + tắt cảnh báo (nếu đang kêu) ----
      if (systemState) {
        beepRunning = false;
        systemState = false;
        buzzerOff();
        fallState           = FSM_MONITORING;
        freefallSampleCount = 0;
        fallDetectedFlag    = false;
        stopSession();
        beepStopped();   // 🔔——— – đã dừng
        Serial.println("\n>>> SYSTEM STOPPED");
      }
    }
    else {
      // ---- Short press: START / NEW SESSION ----
      if (!systemState) {
        beepRunning = false;
        systemState  = true;
        startTime    = millis();
        bufferIndex  = 0;
        currentWindow.ready = false;
        kf_ax.reset(); kf_ay.reset(); kf_az.reset();
        kf_gx.reset(); kf_gy.reset(); kf_gz.reset();
        fallState = FSM_MONITORING; freefallSampleCount = 0; fallDetectedFlag = false;
        newSession();
        beepStarted();     // 🔔-🔔 – đã bắt đầu
        Serial.println("\n>>> SYSTEM RUNNING");
      } else {
        // Xoay session mới (không dừng hệ thống)
        beepRunning = false;
        startTime   = millis();
        bufferIndex = 0;
        currentWindow.ready = false;
        kf_ax.reset(); kf_ay.reset(); kf_az.reset();
        kf_gx.reset(); kf_gy.reset(); kf_gz.reset();
        fallState = FSM_MONITORING; freefallSampleCount = 0;
        newSession();
        beepNewSession();  // 🔔🔔🔔 – session mới
        Serial.println("\n>>> NEW SESSION");
      }
    }
  }
  stableState = reading;
}
