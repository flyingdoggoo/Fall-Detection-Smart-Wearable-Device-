#include <WiFi.h>
#include <HTTPClient.h>
#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include "MAX30105.h"
#include "heartRate.h"
#include <math.h>
#include <ArduinoJson.h>

// ================================================================
// 1. CẤU HÌNH WIFI & SERVER
// ================================================================
const char* ssid = "Giat Ui Tigon 1";
const char* password = "789789789";
const char* serverBase = "http://192.168.1.4:3000";
// const char* ssid = "ITF - Da Nang";
// const char* password = "888888888"; 
// const char* serverBase = "http://172.20.10.2:3000";
#define I2C_SDA 4
#define I2C_SCL 5
#define BUTTON_PIN 9
#define SAMPLE_INTERVAL 20 // 50Hz = 20ms
#define WINDOW_SIZE 100

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
  int bpm;
  long ir;
  float mag_avg, sma, max_a, max_g;
  bool ready = false;
} currentWindow;

// Đối tượng cảm biến
Adafruit_MPU6050 mpu;
MAX30105 particleSensor;
SemaphoreHandle_t xMutex;

// Biến Heart Rate
const byte RATE_SIZE = 4;
byte rates[RATE_SIZE];
byte rateSpot = 0;
long lastBeat = 0;
float beatsPerMinute = 0;
int beatAvg = 0;
long irValue = 0;

// Trạng thái hệ thống
bool systemState = false;
unsigned long lastSampleTime = 0;
unsigned long startTime = 0;
int bufferIndex = 0;

// Button behavior
unsigned long buttonPressedAt = 0;
const unsigned long longPressMs = 1500;

// Endpoint URLs
String batchEndpoint;
String sessionNewEndpoint;
String sessionStopEndpoint;

bool postJson(const String& url, const String& payload) {
  if (WiFi.status() != WL_CONNECTED) return false;
  HTTPClient http;
  http.begin(url);
  http.addHeader("Content-Type", "application/json");
  int httpCode = http.POST(payload);
  http.end();
  return httpCode > 0;
}

void newSession() {
  if (postJson(sessionNewEndpoint, "{}")) {
    Serial.println("✓ Session rotated (server created new folder)");
  } else {
    Serial.println("✗ Failed to rotate session");
  }
}

void stopSession() {
  postJson(sessionStopEndpoint, "{}");
}

// ================================================================
// 3. BỘ LỌC KALMAN
// ================================================================
class SimpleKalmanFilter {
  private:
    float _err_measure, _err_estimate, _q, _current_estimate = 0, _last_estimate = 0;
  public:
    SimpleKalmanFilter(float mea_e, float est_e, float q) : _err_measure(mea_e), _err_estimate(est_e), _q(q) {}
    float updateEstimate(float mea) {
      float gain = _err_estimate / (_err_estimate + _err_measure);
      _current_estimate = _last_estimate + gain * (mea - _last_estimate);
      _err_estimate = (1.0 - gain) * _err_estimate + fabs(_last_estimate - _current_estimate) * _q;
      _last_estimate = _current_estimate;
      return _current_estimate;
    }
    void reset() { _current_estimate = 0; _last_estimate = 0; }
};

SimpleKalmanFilter kf_ax(2, 2, 0.01), kf_ay(2, 2, 0.01), kf_az(2, 2, 0.01);
SimpleKalmanFilter kf_gx(0.5, 0.5, 0.01), kf_gy(0.5, 0.5, 0.01), kf_gz(0.5, 0.5, 0.01);

// ================================================================
// 4. CORE 0: GỬI DỮ LIỆU QUA WIFI (CHẠY NGẦM)
// ================================================================
void SendDataTask(void * pvParameters) {
  while(1) {
    if (currentWindow.ready) {
      if (WiFi.status() == WL_CONNECTED) {
        // Tăng dung lượng JSON lên 16KB để chứa đủ 100 samples + features
        DynamicJsonDocument doc(16384); 
        
        doc["status"] = "active";
        doc["bpm"] = currentWindow.bpm;
        doc["ir_raw"] = currentWindow.ir;
        doc["window_size"] = WINDOW_SIZE;
        doc["sample_rate"] = 50;

        JsonObject features = doc.createNestedObject("features");
        features["magnitude_avg"] = currentWindow.mag_avg;
        features["sma"] = currentWindow.sma;
        features["max_accel"] = currentWindow.max_a;
        features["max_gyro"] = currentWindow.max_g;

        JsonArray accel_data = doc.createNestedArray("accel_data");
        JsonArray gyro_data = doc.createNestedArray("gyro_data");

        // Copy dữ liệu an toàn bằng Mutex
        xSemaphoreTake(xMutex, portMAX_DELAY);
        for (int i = 0; i < WINDOW_SIZE; i++) {
          JsonObject a = accel_data.createNestedObject();
          a["t"] = currentWindow.samples[i].timestamp;
          a["x"] = currentWindow.samples[i].ax; a["y"] = currentWindow.samples[i].ay; a["z"] = currentWindow.samples[i].az;
          
          JsonObject g = gyro_data.createNestedObject();
          g["t"] = currentWindow.samples[i].timestamp;
          g["x"] = currentWindow.samples[i].gx; g["y"] = currentWindow.samples[i].gy; g["z"] = currentWindow.samples[i].gz;
        }
        currentWindow.ready = false; // Giải phóng cho Core 1
        xSemaphoreGive(xMutex);

        String jsonString;
        serializeJson(doc, jsonString);

        HTTPClient http;
        http.begin(batchEndpoint);
        http.addHeader("Content-Type", "application/json");
        int httpCode = http.POST(jsonString);
        
        if (httpCode > 0) Serial.printf("✓ Sent Window | HTTP: %d | Mag: %.2f | BPM: %d\n", httpCode, currentWindow.mag_avg, currentWindow.bpm);
        else Serial.printf("✗ HTTP Error: %s\n", http.errorToString(httpCode).c_str());
        http.end();
      }
    }
    vTaskDelay(5 / portTICK_PERIOD_MS); // Nghỉ 5ms tránh Watchdog
  }
}

// ================================================================
// 5. SETUP & LOOP (CORE 1)
// ================================================================
void setup() {
  Serial.begin(115200);
  pinMode(BUTTON_PIN, INPUT_PULLUP);
  xMutex = xSemaphoreCreateMutex();

  batchEndpoint = String(serverBase) + "/api/sensor-batch";
  sessionNewEndpoint = String(serverBase) + "/api/session/new";
  sessionStopEndpoint = String(serverBase) + "/api/session/stop";

  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) { delay(500); Serial.print("."); }
  Serial.println("\n✓ WiFi Connected!");

  Wire.begin(I2C_SDA, I2C_SCL);
  Wire.setClock(400000);

  if (particleSensor.begin(Wire, I2C_SPEED_FAST)) {
    particleSensor.setup();
    particleSensor.setPulseAmplitudeRed(0x0A);
    Serial.println("✓ MAX30102 Ready");
  }

  if (mpu.begin()) {
    mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
    mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);
    Serial.println("✓ MPU6050 Ready");
  }

  // Khởi chạy Task gửi dữ liệu trên Core 0
  xTaskCreatePinnedToCore(SendDataTask, "SendTask", 16384, NULL, 1, NULL, 0);
  
  Serial.println("\n╔══════════════════════════════╗");
  Serial.println("║  SYSTEM READY - PRESS BUTTON ║");
  Serial.println("╚══════════════════════════════╝");
  Serial.println("Short press: START (if stopped) / NEW SESSION (if running)");
  Serial.println("Long press (>=1.5s): STOP");
}

void loop() {
  handleButton();

  if (systemState) {
    // 1. Đọc Heart Rate nhanh nhất có thể (Bắt buộc để BPM không bằng 0)
    irValue = particleSensor.getIR();
    if (checkForBeat(irValue)) {
      long delta = millis() - lastBeat;
      lastBeat = millis();
      beatsPerMinute = 60 / (delta / 1000.0);
      if (beatsPerMinute < 255 && beatsPerMinute > 20) {
        rates[rateSpot++] = (byte)beatsPerMinute;
        rateSpot %= RATE_SIZE;
        beatAvg = 0;
        for (byte x = 0; x < RATE_SIZE; x++) beatAvg += rates[x];
        beatAvg /= RATE_SIZE;
      }
    }

    // 2. Đọc IMU 50Hz
    unsigned long currentMillis = millis();
    if (currentMillis - lastSampleTime >= SAMPLE_INTERVAL) {
      lastSampleTime = currentMillis;

      sensors_event_t a, g, temp;
      mpu.getEvent(&a, &g, &temp);

      // Chờ Core 0 gửi xong window trước mới nạp tiếp
      if (!currentWindow.ready) { 
        currentWindow.samples[bufferIndex].timestamp = (currentMillis - startTime) / 1000.0;
        currentWindow.samples[bufferIndex].ax = kf_ax.updateEstimate(a.acceleration.x);
        currentWindow.samples[bufferIndex].ay = kf_ay.updateEstimate(a.acceleration.y);
        currentWindow.samples[bufferIndex].az = kf_az.updateEstimate(a.acceleration.z);
        currentWindow.samples[bufferIndex].gx = kf_gx.updateEstimate(g.gyro.x);
        currentWindow.samples[bufferIndex].gy = kf_gy.updateEstimate(g.gyro.y);
        currentWindow.samples[bufferIndex].gz = kf_gz.updateEstimate(g.gyro.z);

        bufferIndex++;

        // Khi đủ 2 giây (100 samples)
        if (bufferIndex >= WINDOW_SIZE) {
          float m_sum = 0, s_sum = 0, m_accel = 0, m_gyro = 0;
          
          for(int i=0; i<WINDOW_SIZE; i++) {
            float m = sqrt(sq(currentWindow.samples[i].ax)+sq(currentWindow.samples[i].ay)+sq(currentWindow.samples[i].az));
            m_sum += m;
            if (m > m_accel) m_accel = m;
            
            float gm = sqrt(sq(currentWindow.samples[i].gx)+sq(currentWindow.samples[i].gy)+sq(currentWindow.samples[i].gz));
            if (gm > m_gyro) m_gyro = gm;

            s_sum += abs(currentWindow.samples[i].ax) + abs(currentWindow.samples[i].ay) + abs(currentWindow.samples[i].az);
          }

          // Chốt dữ liệu
          currentWindow.mag_avg = m_sum / WINDOW_SIZE;
          currentWindow.sma = s_sum / (3.0 * WINDOW_SIZE);
          currentWindow.max_a = m_accel;
          currentWindow.max_g = m_gyro;
          currentWindow.bpm = (irValue < 20000) ? 0 : beatAvg; // Ngưỡng IR nhạy hơn
          currentWindow.ir = irValue;
          
          currentWindow.ready = true; // Kích hoạt Core 0 gửi đi
          bufferIndex = 0;
        }
      }
    }
  }
}

void handleButton() {
  static bool lastBtn = HIGH;
  bool reading = digitalRead(BUTTON_PIN);

  // pressed
  if (lastBtn == HIGH && reading == LOW) {
    delay(30);
    if (digitalRead(BUTTON_PIN) == LOW) {
      buttonPressedAt = millis();
    }
  }

  // released
  if (lastBtn == LOW && reading == HIGH) {
    unsigned long pressDuration = millis() - buttonPressedAt;

    if (pressDuration >= longPressMs) {
      if (systemState) {
        systemState = false;
        stopSession();
        Serial.println("\n>>> SYSTEM: STOPPED");
      }
    } else {
      if (!systemState) {
        systemState = true;
        startTime = millis();
        bufferIndex = 0;
        currentWindow.ready = false;
        kf_ax.reset(); kf_ay.reset(); kf_az.reset();
        kf_gx.reset(); kf_gy.reset(); kf_gz.reset();
        newSession();
        Serial.println("\n>>> SYSTEM: RUNNING");
      } else {
        // Rotate session while still running
        startTime = millis();
        bufferIndex = 0;
        currentWindow.ready = false;
        newSession();
        Serial.println("\n>>> NEW SESSION");
      }
    }
  }

  lastBtn = reading;
}