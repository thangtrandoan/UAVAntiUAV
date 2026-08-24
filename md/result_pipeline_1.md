# Báo Cáo Cấu Hình & Kết Quả Đánh Giá Pipeline ReID

*Báo cáo này được sử dụng làm mẫu (template) chuẩn để theo dõi và đối chiếu hiệu năng của hệ thống mỗi khi có sự thay đổi về tham số cấu hình (Config) hoặc thuật toán cốt lõi. Hãy điền kết quả vào các bảng bên dưới sau mỗi lần chạy thử nghiệm.*

---

## 1. Cấu Trúc Pipeline Chống Nhiễu & Bám Sát (Anti-Hijack & Re-Tracking)

Hệ thống được thiết kế theo mô hình State Machine (Máy trạng thái) với 4 trạng thái cốt lõi:
- **`T0_INIT`**: Bắt đầu theo dõi mục tiêu mới, liên tục trích xuất đặc trưng để xây dựng `Anchor Bank` (Ký ức nguyên thủy).
- **`T1_LOST`**: Mục tiêu bị mất dấu (khuất bóng, bay khỏi camera, hoặc YOLO lỡ nhịp). Khởi động chế độ chờ đợi.
- **`T2_SEARCH`**: Phát hiện các vật thể lạ xuất hiện. 
  - Bước 1: **Lọc Thô (Coarse Filter)** - Chấm điểm bề ngoài (GASNet) của từng vật thể (>= `soft_lock_threshold`) cho frame đầu tiên khi detect lại được.
  - Bước 2: **Thu thập cửa sổ trượt (Sliding Window)** - Quay đủ `num_frames` của vật thể có điểm Lọc thô cao nhất.
  - Bước 3: **Lọc Tinh (Fine Filter)** - Đưa chuỗi khung hình qua Mamba để chấm điểm quỹ đạo và không gian (>= `reid_threshold`).
- **`T3_VERIFIED`**: Chốt mục tiêu (Hard Lock). Liên tục cập nhật `Recent Bank` (Ký ức ngắn hạn) và kiểm tra chống cướp (Anti-Hijack) trong `hijack_check_count` chu kỳ đầu tiên.

---

## 2. Các Tham Số Cấu Hình (Config Parameters) Hiện Trạng

*Ghi chú: Thay đổi các tham số này trong `configs/config_local.yaml` sẽ làm thay đổi trực tiếp độ trễ, tốc độ chạy và tỷ lệ nhận nhầm của hệ thống.*

| Tham số | Giá trị | Vai trò / Ý nghĩa | Tác động khi thay đổi |
| :--- | :---: | :--- | :--- |
| `stride` | **2** | Tần suất lấy mẫu frame (vd 2 nghĩa là cách 1 frame lấy 1 ảnh). | ⬇️ Giảm = Mượt hơn nhưng trễ (latency) cao hơn. ⬆️ Tăng = Ít tốn RAM nhưng mất nét quỹ đạo. |
| `num_frames` | **8** | Độ dài chuỗi vận động yêu cầu để chạy Mamba. | ⬇️ Giảm = Khóa mục tiêu (Hard lock) siêu nhanh. ⬆️ Tăng = Nhận diện cực kỳ chính xác. |
| `max_anchor_size` | **5** | Số lượng ký ức "gốc" tối đa để đối chiếu. | ⬇️ Giảm = Nhẹ RAM. ⬆️ Tăng = Bám mục tiêu bền bỉ hơn khi bị biến dạng. |
| `max_recent_size` | **15** | Số lượng ký ức "ngắn hạn" tối đa để đối chiếu. | Phục vụ cho việc nhận diện khi UAV dần thay đổi góc bay theo thời gian. |
| `soft_lock_threshold`| **0.30** | Ngưỡng điểm Lọc Thô (Cosine) dùng GASNet. | ⬇️ Giảm = Khóa lầm nhiều rác. ⬆️ Tăng = Bỏ lọt mục tiêu thật khi mờ/nhiễu. |
| `reid_threshold` | **0.75** | Ngưỡng điểm Lọc Tinh (Cosine) dùng Mamba. | Cực kỳ quan trọng! ⬇️ Giảm = Dễ bị UAV chim mồi cướp (FAR tăng). ⬆️ Tăng = An toàn nhưng UAV thật dễ bị từ chối (FRR tăng). |
| `hijack_threshold` | **0.40** | Ngưỡng điểm phát hiện mất mục tiêu hoặc bị cướp. | Điểm so với Anchor tụt dưới ngưỡng này sẽ lập tức báo LOST. |

---

## 3. Kết Quả Thử Nghiệm

*(Hãy copy/paste các số liệu từ terminal vào bảng bên dưới sau mỗi lần tinh chỉnh model hoặc đổi file config)*

### A. Đánh giá Tốc Độ & Độ Trễ (Chạy `infer.py`)
Mục đích: Đo lường tốc độ thực thi của phần cứng và độ trễ (latency) từ lúc UAV xuất hiện đến lúc được khóa (Hard Lock). Bỏ qua hoàn toàn sai số của YOLO.

**Lệnh chạy:** `. .venv/bin/activate && python3 infer.py --config configs/config_local.yaml`
**Môi trường:** Local (NVIDIA GeForce RTX 3050 4GB)
**Lưu ý quan trọng:** Model chạy đánh giá sử dụng **trọng số ngẫu nhiên (random weights)** do thiếu file `best_model.pth`.

| Tiêu chí Đánh giá (Metrics) | Kết quả hiện tại (Ngày 21/08/2026) | Kết quả mong muốn |
| :--- | :--- | :--- |
| **Avg CNN Feature Extraction** (ms) | `23.69 ms` | Càng thấp càng tốt (< 15ms) |
| **Avg System Throughput** (FPS) | `43.05 FPS` | > 30 FPS là đạt chuẩn Real-time |
| **Avg Mamba + Head Time** (ms) | `4.67 ms` | Càng thấp càng tốt (< 10ms) |
| **Re-acquisition Latency** (frames) | `18.53 frames` | Càng thấp càng tốt (Khóa lại nhanh) |
| **False Alarms (Fine Fails)** | `7000` | Càng thấp càng tốt (⚠️ Cao do random weights) |

---

### B. Đánh giá Khả Năng Kháng Nhiễu (Chạy `evaluate_reid_robustness.py`)
Mục đích: Kiểm tra xem Lọc Tinh Mamba có bị nhầm lẫn giữa UAV mục tiêu thật (Genuine) và các UAV lạ mạo danh (Imposters) trong trường hợp trên trời xuất hiện nhiều UAV cùng lúc hay không.

**Lệnh chạy:** `python3 evaluate_reid_robustness.py --config configs/config_local.yaml`

| Tiêu chí Đánh giá (Metrics) | Kết quả hiện tại (Ngày ... ) | Phân tích & Hướng chỉnh sửa |
| :--- | :--- | :--- |
| **Total Genuine Queries** | `[Chưa chạy]` | Số lần test con thật. |
| **Total Imposter Queries** | `[Chưa chạy]` | Số lượng con giả bị đem ra test. |
| **Avg Genuine Similarity** | `[Chưa chạy]` | Phải gần 1.0 (vd: 0.95+). Nếu thấp -> Model chưa bắt được đặc trưng thật. |
| **Avg Imposter Similarity** | `[Chưa chạy]` | Phải cực kỳ thấp (vd: < 0.6). Nếu cao -> Model nhận diện kém, cần train lại hoặc nâng `reid_threshold`. |
| **Average Margin (Gen - Imp)** | `[Chưa chạy]` | Khoảng cách giữa thật và giả. Càng cao (vd: 0.4+) thì hệ thống càng phân biệt tốt. |
| **False Rejection Rate (FRR)** | `[Chưa chạy] %` | Tỷ lệ nhận diện trượt con thật. Nếu cao -> Phải hạ `reid_threshold` xuống. |
| **False Acceptance Rate (FAR)**| `[Chưa chạy] %` | Tỷ lệ nhận nhầm con giả. Nếu cao -> Phải nâng `reid_threshold` lên hoặc thu hẹp `num_frames`. |

*Đính kèm biểu đồ `score_distribution.png` sinh ra từ thư mục kết quả vào đây để quan sát trực quan sự giao thoa (Overlap) giữa hai đường màu Xanh (Thật) và Đỏ (Giả).*
