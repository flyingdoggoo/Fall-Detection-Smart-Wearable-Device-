# Hướng dẫn Thu thập Dữ liệu – Fall Detection PBL5
> **Tài liệu này là "bộ nhớ AI" cho dự án.** Đọc file này trước khi thắc mắc bất kỳ điều gì liên quan đến thu data.

---

## 1. Tổng quan Hệ thống

| Thành phần | Chi tiết |
|---|---|
| **Phần cứng** | ESP32-C3 + MPU6050 (gia tốc + con quay) + MAX30102 (nhịp tim) |
| **Tần số lấy mẫu** | 50 Hz (SAMPLE_INTERVAL = 20ms) |
| **Window size** | 100 mẫu = **2 giây** mỗi batch |
| **Server** | Node.js – `server/server.js`, port 3000 |
| **Lưu file** | `server/data/collected/Fall/` và `server/data/collected/Normal/` |
| **Label** | `0` = Không ngã (Normal) · `1` = Ngã (Fall) |

### Luồng dữ liệu
```
MPU6050 (50Hz)
    │
    ▼
ESP32-C3 buffer 100 mẫu (2 giây)
    │  tính features: magnitude_avg, sma, max_accel, max_gyro
    ▼
POST /api/sensor-batch  ──►  server.js  ──►  accel.csv / gyro.csv
                                          └──►  metadata.json (label, sample_rate...)
                                          └──►  fall_markers.json
```

---

## 2. Cách Điều khiển bằng Nút bấm (Phần cứng)

| Thao tác | Hành động |
|---|---|
| **Nhấn ngắn** (khi đang dừng) | Bắt đầu session mới (`/api/session/new`) |
| **Nhấn ngắn** (khi đang chạy) | Kết thúc session hiện tại → bắt đầu session mới ngay |
| **Nhấn giữ ≥ 1.5 giây** | Dừng hoàn toàn (`/api/session/stop`) |

**Đặt label trước khi thu:** Sửa dòng 24 trong `server/server.js`:
```javascript
const CURRENT_LABEL = '0';  // 0 = Normal, 1 = Fall
```
Sau đó restart server. Mỗi session tạo ra sẽ tự động mang label này.

---

## 3. Giải đáp: Thu trong bao lâu?

### 3.1 Session FALL (label = 1)

Một cú ngã thực tế gồm 3 giai đoạn:

```
|── Pre-Fall ──|── Impact ──|── Post-Fall ──|
  (đứng/đi bộ)  (va chạm)    (nằm yên)
    ~3–4 giây     ~1–2 giây    ~3–4 giây
```

**Khuyến nghị: Thu 10–15 giây mỗi session ngã.**
- Bắt đầu thu → đợi ~3 giây (đứng tự nhiên / đi bộ) → thực hiện ngã → nằm yên ~3 giây → nhấn giữ dừng.
- Kết quả: 1 session 12 giây → **6 windows** (6 batch × 2s).

> ⚠️ **Không cần thu lâu hơn 20 giây** cho mỗi session ngã. Thu quá dài làm loãng signal ngã trong data.

### 3.2 Session NORMAL (label = 0)

Thu các hoạt động sinh hoạt thường ngày (ADL) để model học phân biệt với ngã:

| Hoạt động | Thời gian thu khuyến nghị |
|---|---|
| Đứng yên | 15–20 giây |
| Đi bộ | 20–30 giây |
| Ngồi xuống / đứng lên | 15–20 giây |
| Nhặt đồ (cúi người) | 10–15 giây |
| Vẫy tay, vỗ tay | 10–15 giây |
| Chạy nhẹ (jogging) | 20–30 giây |

**Mỗi hoạt động = 1 session riêng biệt.** Nhấn ngắn để tạo session mới giữa các hoạt động.

---

## 4. Giải đáp: Label Dữ liệu như thế nào?

### Hiện tại (Per-Session Label)
Hệ thống gán **1 label cho cả session**. Mọi window 2-giây trong session đều mang label đó.

```
Session label=1 (12s):
  Window 1 (0–2s):   đứng tự nhiên  → label=1  ← KHÔNG CHÍNH XÁC
  Window 2 (2–4s):   đi bộ          → label=1  ← KHÔNG CHÍNH XÁC
  Window 3 (4–6s):   BẮT ĐẦU NGÃ   → label=1  ✓
  Window 4 (6–8s):   ĐANG NGÃ       → label=1  ✓
  Window 5 (8–10s):  nằm yên        → label=1  ← có thể coi là ok
  Window 6 (10–12s): nằm yên        → label=1  ← có thể coi là ok
```

**Đây là cách đơn giản nhất, chấp nhận được cho prototype ban đầu.**  
Model sẽ học "session ngã" mang nhiễu ở đầu, nhưng nếu có đủ data, model vẫn học được.

### Nâng cao (Per-Window Label với Fall Markers)
Server đã có endpoint `/api/mark-fall` và lưu `fall_markers.json`. Kế hoạch dùng sau:
- Khi ngã xảy ra, gọi API mark-fall để đánh dấu timestamp.
- Khi training: window nào chứa timestamp ngã ± 2s → label=1, còn lại → label=0.
- **Cách này cho kết quả chính xác hơn nhưng cần thêm tool post-processing.**

---

## 5. Giải đáp: Thu bao nhiêu là đủ?

### Số lượng tối thiểu (Prototype / Thử nghiệm)

| Class | Sessions | Windows (batch 2s) |
|---|---|---|
| Normal | 20 sessions × 20s = 400s | ~200 windows |
| Fall | 30 sessions × 12s = 360s | ~180 windows |
| **Tổng** | ~50 sessions | **~380 windows** |

Model đủ để chạy thử và đánh giá baseline.

### Số lượng khuyến nghị (Để train model tốt)

| Class | Sessions | Windows |
|---|---|---|
| Normal | 50 sessions (đa dạng ADL) | ~500+ windows |
| Fall | 60 sessions (nhiều kiểu ngã) | ~400+ windows |
| **Tổng** | ~110 sessions | **~900+ windows** |

> **Quy tắc vàng về cân bằng:** Tỉ lệ Normal:Fall không nên vượt quá 3:1.  
> Nếu có ít Fall, dùng kỹ thuật **SMOTE** hoặc **class_weight** khi train.

### Đa dạng hóa dữ liệu FALL (quan trọng!)
Thu nhiều **kiểu ngã khác nhau** để model tổng quát hóa tốt:

| Loại ngã | Mô tả | Sessions cần |
|---|---|---|
| Ngã về phía trước | Vấp, trượt chân → ngã úp | 10 |
| Ngã về phía sau | Trượt → ngã ngửa | 10 |
| Ngã sang trái/phải | Mất thăng bằng sang ngang | 10 |
| Ngã từ tư thế ngồi | Đứng dậy → ngã | 10 |
| Ngã khi đang đi | Vừa đi vừa ngã | 10 |
| Ngã chậm (có bám vào tường) | Trượt từ từ xuống | 10 |

---

## 6. Quy trình Thu Data Chuẩn (Step-by-step)

### Chuẩn bị
```
1. Đeo thiết bị vào CỔ TAY (không phải túi)
2. Khởi động server: cd server && node server.js
3. Mở server.js, đặt CURRENT_LABEL = '0' hoặc '1'
4. Đợi Serial Monitor ESP32 in "✓ WiFi Connected"
```

### Thu NORMAL data
```
1. Đặt CURRENT_LABEL = '0' trong server.js → restart server
2. Nhấn ngắn nút → bắt đầu session
3. Thực hiện 1 hoạt động liên tục (ví dụ: đi bộ 20s)
4. Nhấn ngắn để tạo session mới → tiếp tục hoạt động khác
5. Sau ~10 sessions → nhấn giữ để dừng
```

### Thu FALL data
```
1. Đặt CURRENT_LABEL = '1' trong server.js → restart server
2. Chuẩn bị thảm/đệm an toàn
3. Nhấn ngắn nút → đứng tự nhiên ~3 giây
4. Thực hiện ngã (có kiểm soát, an toàn)
5. Nằm yên ~3 giây
6. Nhấn ngắn để tạo session ngã tiếp theo
   (không cần dừng hẳn, short press sẽ tự lưu session cũ và mở session mới)
7. Lặp lại cho tới khi đủ số lượng
8. Nhấn giữ để dừng hoàn toàn
```

---

## 7. Cấu trúc File Được Lưu

```
server/data/collected/
├── Fall/
│   └── label1_2026-03-04T10-30-00/
│       ├── accel.csv          # accel_time_list, accel_x, accel_y, accel_z
│       ├── gyro.csv           # gyro_time_list, gyro_x, gyro_y, gyro_z
│       ├── label.txt          # nội dung: "1"
│       ├── metadata.json      # session_id, sample_rate=50, window_size=100, label=1
│       └── fall_markers.json  # timestamps khi nhấn /api/mark-fall
└── Normal/
    └── label0_2026-03-04T10-45-00/
        ├── accel.csv
        ├── gyro.csv
        ├── label.txt          # nội dung: "0"
        ├── metadata.json
        └── fall_markers.json
```

### Ý nghĩa các cột CSV
```
accel.csv:
  accel_time_list  → thời gian (giây) kể từ lúc bắt đầu session
  accel_x_list     → gia tốc trục X (m/s²), đã qua Kalman filter
  accel_y_list     → gia tốc trục Y (m/s²), đã qua Kalman filter
  accel_z_list     → gia tốc trục Z (m/s²), đã qua Kalman filter

gyro.csv:
  gyro_time_list   → thời gian (giây)
  gyro_x_list      → vận tốc góc trục X (rad/s), đã qua Kalman filter
  gyro_y_list      → vận tốc góc trục Y (rad/s)
  gyro_z_list      → vận tốc góc trục Z (rad/s)
```

### Features tính sẵn trên ESP32 (trong mỗi batch POST)
| Feature | Công thức | Ý nghĩa |
|---|---|---|
| `magnitude_avg` | mean(√(ax²+ay²+az²)) | Độ lớn gia tốc trung bình của window |
| `sma` | Σ(|ax|+|ay|+|az|) / (3×N) | Signal Magnitude Area – phân biệt nghỉ/hoạt động |
| `max_accel` | max(√(ax²+ay²+az²)) | Peak gia tốc trong window – rất cao khi ngã |
| `max_gyro` | max(√(gx²+gy²+gz²)) | Peak góc xoay trong window |
| `bpm` | nhịp tim (MAX30102) | 0 nếu không đeo sát |
| `ir_raw` | giá trị IR sensor | 0 hoặc rất thấp nếu không tiếp xúc da |

---

## 8. Dấu hiệu Nhận biết khi Ngã trong Data

Khi xem `accel.csv` của session ngã, cần thấy rõ:

1. **Giai đoạn Free-fall:** `az` giảm đột ngột xuống gần 0 (mất trọng lực) → magnitude_avg < 5 m/s²
2. **Giai đoạn Impact:** spike lớn ở magnitude_avg, có thể vượt 20–40 m/s²  
3. **Giai đoạn Post-fall:** tín hiệu ổn định lại, ít dao động

Nếu không thấy pattern này trong data ngã → cần kiểm tra lại vị trí đeo thiết bị.

---

## 9. Checklist Trước khi Train Model

- [ ] Đủ ≥ 300 windows cho mỗi class (Fall và Normal)
- [ ] Tỉ lệ Fall:Normal ≤ 1:3 (nếu lệch nhiều → dùng class_weight)
- [ ] Data Normal đa dạng: đi bộ, ngồi, đứng, cúi, vỗ tay...
- [ ] Data Fall đa dạng: nhiều hướng ngã khác nhau
- [ ] Kiểm tra không có session nào bị thiếu file (metadata.json, label.txt)
- [ ] Xác nhận label.txt khớp với thư mục lưu (Fall/ chứa label=1)

---

## 10. Thông số Kỹ thuật ESP32 (Tham khảo nhanh)

```cpp
#define SAMPLE_INTERVAL  20    // ms → 50 Hz
#define WINDOW_SIZE      100   // samples → 2 giây / window
#define BUTTON_PIN       9     // GPIO9
#define I2C_SDA          4     // GPIO4
#define I2C_SCL          5     // GPIO5
longPressMs = 1500             // giữ 1.5s = STOP
```

**MPU6050 config:**
- Accelerometer range: ±8G (`MPU6050_RANGE_8_G`)
- Filter bandwidth: 21 Hz (`MPU6050_BAND_21_HZ`)
- Kalman filter: mea_e=2, est_e=2, q=0.01 (accel) · mea_e=0.5 (gyro)

---

## 11. Lỗi Thường Gặp

| Triệu chứng | Nguyên nhân | Cách sửa |
|---|---|---|
| Serial in `✗ HTTP Error` liên tục | Sai IP server trong `.ino` | Sửa `serverBase` cho đúng IP máy tính |
| Session folder tạo ra nhưng CSV rỗng | ESP32 gửi batch nhưng server lỗi | Kiểm tra console server.js |
| `label.txt` toàn là "0" dù đang thu ngã | Quên đổi `CURRENT_LABEL` | Sửa dòng 24 `server.js` → restart |
| BPM luôn = 0 | MAX30102 không tiếp xúc da | Chỉnh lại dây đeo hoặc bỏ qua BPM khi train |
| Data rất nhiều noise | Không có Kalman / loose mounting | Thiết bị đã có Kalman, kiểm tra cố định |

---

*Cập nhật lần cuối: 2026-03-04*
