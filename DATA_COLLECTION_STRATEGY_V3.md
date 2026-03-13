# Chiến Thuật Thu Data & Đánh Nhãn – Fall Detection PBL5 v3
> Cập nhật: 2026-03-09

---

## 1. Đánh giá cách đánh nhãn hiện tại

### Per-Session Label – Có vấn đề gì?

Cách hiện tại: gán **1 label cho toàn bộ session** → mọi window 2 giây đều mang label đó.

**Vấn đề cụ thể với session FALL (label=1):**
```
Session 12 giây, label=1:
  Window 0-2s:   đứng tự nhiên    → label=1  ❌ (thực ra là Normal)
  Window 2-4s:   đi bộ            → label=1  ❌ (thực ra là Normal)
  Window 4-6s:   BẮT ĐẦU NGÃ     → label=1  ✅
  Window 6-8s:   VA CHẠM          → label=1  ✅
  Window 8-10s:  nằm yên          → label=1  ⚠️ (nhiễu nhãn)
  Window 10-12s: nằm yên          → label=1  ⚠️ (nhiễu nhãn)
```

**Kết luận:** Chấp nhận được cho prototype nhưng sẽ giới hạn accuracy. Model học nhiễu → giảm precision.

### Giải pháp đề xuất: **Chiến thuật "Thu ngắn + Fall marker"**

Thay vì sửa kiến trúc phức tạp, áp dụng 2 chiến thuật đơn giản:

---

## 2. Chiến thuật thu FALL Sessions (Label = 1)

### Nguyên tắc: THU NGẮN NHẤT CÓ THỂ

```
Quy trình:
1. Đặt label = 1 trên giao diện web (không cần restart server)
2. Nhấn Start → ĐỢI 1-2 GIÂY
3. THỰC HIỆN NGÃ NGAY
4. Nằm yên 2-3 giây
5. Nhấn Stop

Tổng: 6-8 giây = 3-4 windows
```

**Tại sao thu ngắn tốt hơn?**
- Session 6s, 3 windows: chỉ 1 window "pre-fall" bị nhiễu → tỉ lệ nhiễu 33%
- Session 15s, 7 windows: 2-3 windows pre-fall + 2 post-fall bị nhiễu → tỉ lệ nhiễu 57-71%
- **Thu ngắn = tự động giảm label noise mà không cần per-window labeling!**

### Bổ sung: Fall Marker

- Nhấn nút **"⚡ Mark Fall Timestamp"** trên giao diện ngay lúc ngã
- Server lưu timestamp chính xác → khi train, có thể lọc thêm nếu cần
- Không bắt buộc – chỉ là data phụ cho phiên bản nâng cao

### Kịch bản thu Fall chi tiết

| # | Loại ngã | Mô tả | SL Sessions |
|---|----------|--------|-------------|
| 1 | Ngã trước | Vấp chân → ngã úp | 15 |
| 2 | Ngã sau | Trượt → ngã ngửa | 15 |
| 3 | Ngã trái | Mất thăng bằng → ngã trái | 10 |
| 4 | Ngã phải | Mất thăng bằng → ngã phải | 10 |
| 5 | Ngã từ ngồi | Đứng dậy → ngã | 8 |
| 6 | Ngã khi đi | Đang đi → vấp → ngã | 12 |
| 7 | Ngã chậm | Trượt từ từ xuống tường | 5 |
| **Tổng** | | | **~75 sessions** |

**Mỗi session 6-8s → 3-4 windows → ~270 fall windows**

---

## 3. Chiến thuật thu NORMAL Sessions (Label = 0)

### Nguyên tắc: ĐA DẠNG HOẠT ĐỘNG

```
Quy trình:
1. Đặt label = 0 trên giao diện web
2. Nhấn Start → thực hiện 1 hoạt động liên tục
3. Nhấn nút ngắn để chuyển session mới → hoạt động tiếp theo
4. Sau 10-15 sessions → nhấn giữ dừng
```

| # | Hoạt động | Thời gian/session | SL Sessions |
|---|-----------|-------------------|-------------|
| 1 | Đứng yên | 15s | 8 |
| 2 | Đi bộ chậm | 20s | 10 |
| 3 | Đi bộ nhanh | 20s | 8 |
| 4 | Ngồi xuống + đứng lên | 15s | 10 |
| 5 | Nhặt đồ (cúi) | 10s | 8 |
| 6 | Vỗ tay/vẫy tay | 10s | 5 |
| 7 | Chạy nhẹ (jogging) | 20s | 8 |
| 8 | Leo cầu thang | 15s | 5 |
| 9 | Xoay người/quay đầu | 10s | 5 |
| 10 | Nằm xuống giường (có kiểm soát) | 10s | 8 |
| **Tổng** | | | **~75 sessions** |

**~500+ normal windows (vì session normal dài hơn)**

---

## 4. Kế hoạch thu data theo ngày

### Ngày 1: Baseline (tối thiểu)
- [ ] 20 sessions Fall (5 kiểu ngã × 4)
- [ ] 20 sessions Normal (5 hoạt động × 4)
- → Train thử mô hình, kiểm tra pipeline

### Ngày 2: Bổ sung
- [ ] 25 sessions Fall (thêm kiểu ngã)
- [ ] 25 sessions Normal (thêm hoạt động)
- → Train lại, so sánh cải thiện

### Ngày 3: Hoàn thiện
- [ ] 30 sessions Fall (đa dạng tốc độ, góc ngã)
- [ ] 30 sessions Normal (thu trong nhiều môi trường khác nhau)
- → Train final model

---

## 5. Lưu ý kỹ thuật khi thu

1. **Đeo thiết bị cố định ở cổ tay** – dùng dây đeo chặt, không rung lắc
2. **Đổi label trên giao diện web** – KHÔNG cần sửa code hay restart server
3. **Kiểm tra session count** trên giao diện – đảm bảo data đã lưu
4. **An toàn khi ngã**: Dùng thảm, đệm. Quan trọng hơn data!
5. **Đa dạng người thu**: Nếu có thể, nhờ 2-3 người thu để model tổng quát hơn
6. **Đa dạng tốc độ ngã**: Ngã nhanh + ngã chậm
7. **Mỗi session chỉ 1 hoạt động**: Đừng trộn đi bộ + ngồi trong cùng 1 session
8. **Sau khi thu xong 1 đợt**: Mở 1-2 file CSV kiểm tra data có hợp lý không

---

## 6. Cấu trúc file sau khi thu (v3)

```
server/data/collected/
├── Fall/
│   └── label1_2026-03-09T10-30-00/
│       ├── accel.csv          # ax, ay, az
│       ├── gyro.csv           # gx, gy, gz
│       ├── features.csv       # [MỚI] magnitude, sma, std, jerk, fsm_state...
│       ├── label.txt          # "1"
│       ├── metadata.json      # sample_rate=50, window_size=100, label=1
│       └── fall_markers.json  # timestamps ngã (nếu đánh dấu)
└── Normal/
    └── label0_2026-03-09T10-45-00/
        ├── accel.csv
        ├── gyro.csv
        ├── features.csv       # [MỚI]
        ├── label.txt          # "0"
        ├── metadata.json
        └── fall_markers.json
```

---

## 7. Checklist trước khi Train

- [ ] ≥ 200 windows mỗi class (Fall / Normal)
- [ ] Tỉ lệ Fall:Normal trong khoảng 1:1 → 1:2.5
- [ ] Data Normal đa dạng ≥ 6 loại hoạt động
- [ ] Data Fall đa dạng ≥ 5 kiểu ngã
- [ ] Kiểm tra file CSV không rỗng
- [ ] label.txt khớp với thư mục cha (Fall/ chứa "1", Normal/ chứa "0")
- [ ] Upload thư mục `collected/` lên Google Drive
- [ ] Mở notebook `fall_detection_cnn_lstm.py` trên Colab → chạy

---

## 8. Tóm tắt thay đổi V3

| Hạng mục | V2 (cũ) | V3 (mới) |
|----------|---------|----------|
| Đổi label | Sửa code `server.js` → restart | Nhấn nút trên web UI |
| Features ESP32 | 4 (mag, sma, max_a, max_g) | 6 + FSM state + fall_detected |
| Offline detection | Không có | FSM 4 trạng thái + buzzer GPIO18 |
| NaN/Inf | Không xử lý | Guard ở sensor + Kalman |
| Kalman filter | Basic | Adaptive Q + first-sample init |
| Giao diện | Light theme, basic | Dark theme, FSM banner, label control |
| Features CSV | Không có | Tự động lưu per-window |
| Fall marker | Có API nhưng chưa UI | Button "Mark Fall" trên UI |
