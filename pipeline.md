# Kiến trúc End-to-End UAV Tracking & Re-Identification Pipeline

Tài liệu này mô tả chi tiết toàn bộ luồng xử lý (Pipeline) của hệ thống UAVAntiUAV trong môi trường thực chiến (`infer_realworld.py`), kết hợp giữa phát hiện vật thể (Object Detection), theo dõi quỹ đạo (Kalman Tracking) và định danh sinh trắc học (Mamba ReID).

---

## 1. Tầng Phát hiện & Bám đuổi Không gian (Spatial Detection & Soft Lock)

Tầng này đóng vai trò như "Đôi mắt" và "Phản xạ thần kinh", chịu trách nhiệm lọc nhiễu diện rộng và cung cấp các hộp tọa độ (Bounding Box).

- **Object Detection (YOLOv8):** 
  - Kích hoạt chế độ nhìn độ phân giải cao (`imgsz = 1080`) để không bỏ sót các UAV siêu nhỏ ở xa.
  - Sử dụng màng lọc Class (`detector_classes = [4]`) để ép hệ thống chỉ quan tâm đến các vật thể mang hình dáng Máy bay/Fixed-wing UAV.
  - Hạ thấp ngưỡng tự tin (`conf = 0.15`) để bắt được cả những mục tiêu bị mờ do chuyển động (Motion blur).
  - *Chi phí tính toán ước tính (Jetson AGX Orin):* ~15-20ms/frame.
- **Soft Tracking (ByteTrack / Kalman Filter):** 
  - Đảm nhiệm việc duy trì **Soft Lock**. Nếu một UAV bị che khuất trong chớp mắt (vài frame), Kalman Filter dự đoán quỹ đạo và tự động nối lại ID khi UAV xuất hiện. Giai đoạn này vận hành liên tục, tốn cực kỳ ít tài nguyên (O(1)).
  - *Chi phí tính toán ước tính:* < 2ms/frame (Chủ yếu là tính toán ma trận tọa độ cơ bản).

---

## 2. Tầng Trích xuất Đặc trưng (Feature Extraction Engine)

Khi UAV lọt vào tầm ngắm, hình ảnh của nó được gửi vào hệ thống Thần kinh Trung ương để chuyển đổi thành chuỗi dữ liệu định danh sinh trắc học (Biometric Vectors).

1. **Visual Backbone (GASNet / ResNet50-IBN):** 
   - Nhận 1 khung hình (Frame).
   - Xuất ra Vector Không gian 2560 chiều (Bao gồm 2048 chiều Toàn cục Global + 512 chiều Cục bộ Fine-grained). Đại diện cho "Ngoại hình" của UAV.
   - *Chi phí tính toán ước tính:* ~5ms/crop.
2. **Temporal Mamba Encoder:** 
   - Nhận chuỗi 16 Vectors (tương ứng 16 frames).
   - Rút trích ra Vector Thời gian 512 chiều. Đại diện cho "Thói quen chuyển động" và "Độ rung cánh quạt" của UAV.
   - *Chi phí tính toán ước tính:* ~1-2ms/sequence (Tốc độ siêu nhanh nhờ kiến trúc State Space Model).
3. **ReID Head (Feature Fusion):** 
   - Hợp nhất Visual (2560) và Temporal (512) thành Vector hoàn chỉnh 3072 chiều.
   - Đi qua màng lọc `BatchNorm1d` để khuếch đại tín hiệu chuyển động, đảm bảo tỷ lệ Vàng 5:1 (Hình dáng quyết định, Chuyển động làm Trọng tài).
   - *Chi phí tính toán ước tính:* < 0.1ms.

---

## 3. Kiến trúc Đa Mục Tiêu: Lọc Thô đến Lọc Tinh (Coarse-to-Fine Multi-UAV)

Khi Soft Lock bị gãy (UAV mất dấu quá lâu), hệ thống rơi vào trạng thái `LOST` và kích hoạt thợ săn Mamba. Nếu có 15 chiếc UAV nhiễu xuất hiện cùng lúc, thuật toán giải quyết bài toán O(N) qua 3 bước:

- **Bước 1: Lọc Thô Siêu Tốc (Coarse ReID)**
  - Quét 1 frame duy nhất của tất cả 15 chiếc UAV.
  - Nhân bản frame đó thành ảo ảnh 16 frames và đẩy qua Mamba.
  - UAV nào có điểm `< 0.50` bị loại bỏ vĩnh viễn khỏi bộ nhớ ngay từ mili-giây đầu tiên.
  - *Chi phí tính toán ước tính:* ~6ms/UAV nhiễu (Gồm 5ms cho CNN + 1ms cho Mamba. Lọc 15 UAV nhiễu tốn tổng cộng chưa tới ~90ms).
- **Bước 2: Khóa Tạm Thời (Soft Lock)**
  - Đối với các UAV lọt qua vòng Lọc Thô, hệ thống chọn ra chiếc có điểm Coarse cao nhất.
  - Gán nhãn **SOFT LOCK (Khung Xanh Lơ)**. Nòng súng/Camera ngay lập tức bám theo chiếc này để không bỏ lỡ nhịp độ chiến đấu.
  - *Chi phí tính toán ước tính:* < 0.1ms (Chỉ là phép tính logic so sánh mảng).
- **Bước 3: Xác thực Chuyên sâu (Fine ReID)**
  - Theo dõi gắt gao các "Nghi phạm" trong đúng 16 frames (khoảng 0.5s).
  - Điểm Mamba thực tế `> 0.75` -> Chuyển thành **HARD LOCK (Khung Xanh Lá)**.
  - Nếu `< 0.75`, ném kẻ này vào Blacklist và tự động chuyển Soft Lock sang mục tiêu có điểm cao thứ 2.
  - *Chi phí tính toán ước tính:* Tải tính toán chia đều ra 16 frames (~5ms/frame cho trích xuất CNN). Cuối frame 16 chốt hạ bằng Mamba mất thêm 1ms. Cực kỳ nhẹ nhàng và phân tán.

---

## 4. Các Lớp Phòng Thủ Tối Thượng (Defensive Mechanisms)

Hệ thống được trang bị 3 cơ chế miễn nhiễm với các thủ đoạn giả mạo và nhiễu loạn môi trường:

### A. Bộ Phạt Không Gian (Spatio-Temporal Penalty)
- **Vấn đề:** Địch thả một UAV mồi nhử giống y hệt phe ta ở cách đó 2 cây số.
- **Giải pháp:** Hệ thống đo khoảng cách (Euclidean Distance) từ vị trí xuất hiện của chiếc UAV mới đến tọa độ cuối cùng mà phe ta biến mất. Khoảng cách càng xa, điểm ReID bị trừ càng nặng (với hệ số phạt `spatial_weight = 0.30`). UAV mồi nhử dù giống đến đâu cũng bị trừ điểm thê thảm và rớt đài.
- *Chi phí tính toán ước tính:* `0.0001 ms` (Thuần toán học tính cạnh huyền tam giác).

### B. Ngân Hàng Ký Ức Đa Tầng (2-Tier Memory Bank)
- **Vấn đề:** UAV xoay 180 độ, góc nhìn thay đổi khiến vector trung bình bị loãng (Score Dilution).
- **Giải pháp:** Chia ký ức làm `Anchor Bank` (Hình dáng gốc ban đầu) và `Recent Bank` (Hình dáng 30 frames gần nhất). Sử dụng hàm `max()`: Chỉ cần góc xoay hiện tại giống với *một trong những ký ức từng thấy* là đủ để nhận diện.

### C. Xác Thực Trực Tiếp (Continuous Verification / Anti-Hijack)
- **Vấn đề:** YOLO bị lú, gán nhầm Soft Lock của UAV phe ta cho một con chim bay ngang qua.
- **Giải pháp:** Cứ mỗi 60 frames, hệ thống lén lôi 16 frames của mục tiêu hiện tại ra đối chiếu với `Anchor Bank`. Nếu phát hiện điểm rớt xuống `< 0.40`, hệ thống lập tức phán quyết Soft Lock đã bị cướp (Tracking Hijack) -> Ép **Break Lock**, quay về trạng thái `LOST` để đi tìm lại UAV gốc.
- *Chi phí tính toán ước tính:* ~1ms thực thi Mamba. Khấu hao trên 60 frames nên chi phí trung bình cộng dồn là `0.016 ms/frame` (Gần như bằng 0).
