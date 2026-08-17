# Kế hoạch: `infer_realworld.py` — Inference trên video bay thực tế (Phan Rang)

## Bối cảnh

File [`infer.py`](file:///home/thang/UAVAntiUAV/infer.py) hiện tại **phụ thuộc hoàn toàn** vào 2 file nhãn từ dataset:
- `groundtruth_rect.txt` → cung cấp bounding box mỗi frame (giả lập Detector/Tracker)
- `absent.txt` → cho biết UAV có đang hiển thị hay không (giả lập trạng thái mất dấu)

Khi có video bay thực tế, **KHÔNG** có 2 file nhãn trên. Vì vậy cần thay thế bằng một **Object Detector + Tracker thực tế** để tự động phát hiện và theo dõi UAV trong từng frame.

**Mục tiêu:** Tạo file `infer_realworld.py` sao cho:
- Đầu vào: 1 file video thực tế bất kỳ (`.mp4`, `.avi`, ...)
- Đầu ra: giống `infer.py` (video có annotation, `metrics.txt`)
- Giữ nguyên 100% logic ReID (Memory Bank, Hybrid Sampling, Rolling Verification)

---

## 1. Phân tích: Cái gì giữ nguyên, cái gì thay đổi

### 1.1. Giữ nguyên 100% (copy từ `infer.py`)

| Hàm / Logic | Dòng trong `infer.py` | Mô tả |
|---|---|---|
| `extract_cnn_feature()` | L15-26 | Trích xuất CNN feature từ 1 frame qua backbone GASNet |
| `compute_reid_embedding()` | L28-34 | Tính ReID embedding vector qua Mamba + BN Head |
| `crop_and_resize()` | L47-68 | Crop ảnh theo bbox + padding rồi resize về 256x256 |
| Load model + skip shape mismatch | L126-140 | Load `UAVReIDNet`, bỏ qua lớp Classifier bị lệch shape |
| Transform pipeline | L147-152 | `ToPILImage → CenterCrop(224) → ToTensor → Normalize` |
| State machine 4 trạng thái | L191-332 | `INITIAL_TRACKING → LOST → REAPPEARED_VERIFYING → REID_SUCCESS` |
| Memory Bank | L205-206, L226-232 | Lưu gallery mỗi 2 giây, tối đa 50 poses |
| Hybrid Sampling | L212-213, L253 | Sparse (3 lấy 1) khi tracking, Dense khi verifying |
| Rolling Verification | L271-284 | So khớp query với toàn bộ memory bank, pop frame cũ nếu fail |
| HUD overlay | L334-341 | Vẽ FPS, CNN time, false alarms lên video |
| Metrics report + lưu file | L351-372 | Xuất báo cáo hiệu năng ra console và `metrics.txt` |

### 1.2. Cần thay đổi / viết mới

| Phần | Hành động | Khối lượng |
|---|---|---|
| `parse_args()` | Bỏ `--seq-dir`, thêm 4 tham số mới cho detector | Sửa nhẹ |
| Đọc nhãn GT | **XÓA HOÀN TOÀN** phần đọc `groundtruth_rect.txt` + `absent.txt` | Xóa ~15 dòng |
| Khởi tạo YOLO detector | **VIẾT MỚI** | ~5 dòng |
| Chạy detector + tracker mỗi frame | **VIẾT MỚI** — thay thế cho việc đọc bbox/absent từ file | ~25 dòng |
| Lost Threshold (bộ đếm chống nhấp nháy) | **VIẾT MỚI** | ~10 dòng |
| Config YAML | Thêm block `infer_realworld` | Sửa nhẹ |

> **Ước tính: ~85% code copy nguyên từ `infer.py`, chỉ ~15% viết mới.**

---

## 2. Thiết kế chi tiết phần viết mới

### 2.1. Chọn Detector: YOLOv8 (ultralytics)

**Lý do chọn:**
- Cài đặt 1 dòng: `pip install ultralytics`
- Có sẵn tracker tích hợp (ByteTrack / BoTSORT) → không cần cài thêm thư viện tracker
- Chạy tốt trên Jetson AGX Orin, hỗ trợ export TensorRT
- API cực đơn giản

**Lưu ý quan trọng:** YOLO pretrained trên COCO **KHÔNG có class "drone/UAV"**. Giải pháp tạm:
- Bản đầu tiên: chấp nhận **tất cả detection** hoặc lọc class gần nhất (`airplane`=4, `bird`=14)
- Thêm tham số `--detector-classes` để bạn tuỳ chỉnh, mặc định `null` = nhận tất cả
- Sau này nếu cần chính xác hơn: fine-tune YOLO trên dataset drone detection

**Cài đặt:**
```bash
pip install ultralytics
```

Trên Jetson, để tối đa tốc độ:
```bash
yolo export model=yolov8n.pt format=engine device=0
# Rồi dùng: --yolo-model yolov8n.engine
```

### 2.2. Logic chuyển đổi: Detector output → State Machine input

Hiện tại state machine cần 3 biến mỗi frame:
```python
is_absent   # bool: True = UAV không có trong frame
bbox        # [x, y, w, h]: bounding box
valid_bbox  # bool: True = bbox hợp lệ (w > 0 và h > 0)
```

**Code mới thay thế:**

```python
from ultralytics import YOLO

# Khởi tạo (1 lần duy nhất trước vòng lặp)
yolo_model = YOLO(args.yolo_model)  # vd: "yolov8n.pt"

# Biến theo dõi trạng thái detector
lost_count = 0
last_valid_bbox = [0, 0, 0, 0]

# Trong vòng lặp, mỗi frame:
results = yolo_model.track(frame, persist=True, conf=args.detector_conf, verbose=False)

# Lọc detection theo class (nếu cần)
filtered_boxes = []
if results[0].boxes is not None and len(results[0].boxes) > 0:
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        if args.detector_classes is None or cls_id in args.detector_classes:
            filtered_boxes.append(box)

# Chuyển đổi sang is_absent / bbox / valid_bbox
if len(filtered_boxes) == 0:
    lost_count += 1
    if lost_count >= args.lost_threshold:
        # Mất detection đủ lâu → coi là UAV thật sự biến mất
        is_absent = True
        bbox = [0, 0, 0, 0]
        valid_bbox = False
    else:
        # Mới mất 1-2 frame → có thể detector nhấp nháy, giữ bbox cũ
        is_absent = False
        bbox = last_valid_bbox
        valid_bbox = bbox[2] > 0 and bbox[3] > 0
else:
    lost_count = 0
    is_absent = False
    # Lấy detection có confidence cao nhất
    best_box = max(filtered_boxes, key=lambda b: float(b.conf[0]))
    x1, y1, x2, y2 = best_box.xyxy[0].cpu().numpy().astype(int)
    bbox = [x1, y1, x2 - x1, y2 - y1]  # convert xyxy → xywh
    valid_bbox = bbox[2] > 0 and bbox[3] > 0
    last_valid_bbox = bbox

# >>> Từ đây trở đi, state machine GIỮA NGUYÊN y hệt infer.py <<<
# Dùng is_absent, bbox, valid_bbox như bình thường
```

### 2.3. Giải thích Lost Threshold

Detector không hoàn hảo. Nó có thể bị miss 1-2 frame ngẫu nhiên rồi detect lại bình thường. Nếu không có bộ đếm, state machine sẽ nhảy liên tục giữa `TRACKING ↔ LOST` gây ra:
- Memory Bank bị flush liên tục
- Feature buffer bị reset
- Video output nhấp nháy

**Giải pháp:** Thêm biến `lost_count`, chỉ chuyển sang `is_absent = True` khi detector mất liên tục `>= lost_threshold` frames (mặc định 10 frames ≈ 0.33 giây ở 30fps). Khi đang trong vùng đệm này, giữ nguyên `last_valid_bbox` để state machine vẫn hoạt động bình thường.

---

## 3. Cấu trúc file `infer_realworld.py` (từ trên xuống dưới)

```
┌─────────────────────────────────────────────┐
│  1. IMPORTS                                  │
│     - Giữ nguyên tất cả import từ infer.py   │
│     - Thêm: from ultralytics import YOLO     │
├─────────────────────────────────────────────┤
│  2. HÀM TIỆN ÍCH (COPY NGUYÊN)             │
│     - extract_cnn_feature()                  │
│     - compute_reid_embedding()               │
│     - crop_and_resize()                      │
├─────────────────────────────────────────────┤
│  3. parse_args() — SỬA                      │
│     Bỏ:                                     │
│       --seq-dir                              │
│     Giữ:                                    │
│       --checkpoint, --config, --output-video │
│       --threshold, --num-frames, --bbox-padding │
│     Thêm mới:                               │
│       --video           (đường dẫn video)    │
│       --yolo-model      (mặc định yolov8n.pt)│
│       --detector-conf   (mặc định 0.3)      │
│       --detector-classes (mặc định None)     │
│       --lost-threshold  (mặc định 10)        │
├─────────────────────────────────────────────┤
│  4. main()                                   │
│                                              │
│  4a. Load config YAML                        │
│      → đọc từ block "infer_realworld"        │
│                                              │
│  4b. Khởi tạo YOLO detector     ← VIẾT MỚI  │
│      yolo_model = YOLO(args.yolo_model)      │
│                                              │
│  4c. Khởi tạo ReID model        ← COPY      │
│      UAVReIDNet + load checkpoint            │
│                                              │
│  4d. Mở video, tạo VideoWriter   ← SỬA NHẸ  │
│      Đọc trực tiếp args.video               │
│      (không ghép đường dẫn từ seq_dir nữa)   │
│      Output dir: infer_realworld/{video_name}/│
│                                              │
│  4e. Vòng lặp chính:                        │
│      ┌──────────────────────────────────┐    │
│      │ Đọc frame                        │    │
│      │ Chạy YOLO detect+track  ← MỚI   │    │
│      │ Chuyển đổi → is_absent/bbox ← MỚI│    │
│      │ Lost threshold logic     ← MỚI   │    │
│      │ State machine            ← COPY  │    │
│      │ Vẽ HUD (+ detector conf) ← SỬA  │    │
│      └──────────────────────────────────┘    │
│                                              │
│  4f. Xuất metrics report         ← COPY     │
└─────────────────────────────────────────────┘
```

---

## 4. Config YAML

Thêm block mới vào `config_jetson.yaml`:

```yaml
# --- Real-world Inference ---
infer_realworld:
  video: ./data/realworld/test_flight_01.mp4
  yolo_model: yolov8n.pt
  detector_conf: 0.3
  detector_classes: null        # null = tất cả class, [4, 14] = airplane + bird
  lost_threshold: 10
  model_path: ./best_model.pth
  output_video: output_realworld.mp4
  reid_threshold: 0.75
  num_frames: 16
  bbox_padding: 0.2
```

Thêm block mới vào `config_colab.yaml`:

```yaml
# --- Real-world Inference ---
infer_realworld:
  video: /content/drive/MyDrive/UAV_Anti_UAV/realworld/test_flight_01.mp4
  yolo_model: yolov8n.pt
  detector_conf: 0.3
  detector_classes: null
  lost_threshold: 10
  model_path: /content/best_model.pth
  output_video: output_realworld.mp4
  reid_threshold: 0.7
  num_frames: 16
  bbox_padding: 0.2
```

---

## 5. Cách chạy

```bash
# Trên Jetson
python infer_realworld.py --config configs/config_jetson.yaml

# Hoặc truyền trực tiếp
python infer_realworld.py \
    --video ./data/realworld/test_flight_01.mp4 \
    --checkpoint ./best_model.pth \
    --yolo-model yolov8n.pt \
    --threshold 0.75

# Trên Colab
python infer_realworld.py --config configs/config_colab.yaml
```

---

## 6. Đầu ra kỳ vọng

Kết quả lưu vào thư mục `infer_realworld/{tên_video}/`:

```
infer_realworld/
└── test_flight_01/
    ├── output_realworld.mp4   ← Video có vẽ bbox, trạng thái, ReID result
    └── metrics.txt            ← Báo cáo: CNN time, Mamba time, FPS, False Alarms, Latency
```

Video output sẽ hiển thị:
- **Bbox màu xanh dương** + "Tracking" khi đang theo dõi ban đầu
- **"TARGET LOST"** màu đỏ khi UAV biến mất
- **Bbox màu cam** + "VERIFYING ID..." khi UAV xuất hiện lại và đang kiểm tra
- **Bbox màu xanh lá** + "ReID Tracked" + confidence khi nhận diện thành công
- **HUD góc dưới**: Model FPS, CNN Time, Detector Confidence, False Alarms

---

## 7. Tóm tắt bảng so sánh `infer.py` vs `infer_realworld.py`

| Tiêu chí | `infer.py` | `infer_realworld.py` |
|---|---|---|
| Nguồn bbox | `groundtruth_rect.txt` | YOLOv8 detector |
| Nguồn absent | `absent.txt` | Detector không thấy gì + lost threshold |
| Cần dataset | Có (cấu trúc thư mục Test/) | Không — chỉ cần 1 file video |
| Dependency thêm | Không | `pip install ultralytics` |
| Chống nhấp nháy | Không cần (GT hoàn hảo) | Lost Threshold (10 frames) |
| State machine | 4 trạng thái | 4 trạng thái (y hệt) |
| Memory Bank | Có | Có (y hệt) |
| Hybrid Sampling | Có | Có (y hệt) |
| Rolling Verification | Có | Có (y hệt) |
| Output | Video + metrics.txt | Video + metrics.txt (y hệt) |
