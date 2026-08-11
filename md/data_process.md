# Báo cáo: Pipeline Xử lý Dữ liệu (Data Processing)

## 1. Mục đích và Những Thay đổi Chính
Pipeline xử lý dữ liệu đã được viết lại hoàn toàn để phục vụ tốt hơn cho bài toán Re-Identification (Re-ID) nhận diện UAV qua các sự kiện che khuất. Những cải tiến chính bao gồm:

- **Lấy frame ngắt quãng (Interval Sampling):** Thay vì lấy các frame liên tục gây dư thừa thông tin (redundancy) vì các frame cạnh nhau rất giống nhau, hệ thống cho phép lấy frame theo một bước nhảy (thông số `frame-stride`). Ví dụ với bước nhảy = 2, ta sẽ trích xuất các frame: $t, t-2, t-4...$ Điều này giúp bao quát được một khoảng thời gian dài hơn với cùng số lượng ảnh, tăng độ đa dạng của dữ liệu.
- **Tách biệt Query và Gallery:** Chuyển đổi định dạng output từ các "cặp" (pairs) gộp chung thành định dạng chuẩn của bài toán Re-ID là Query và Gallery. Điều này tạo nền tảng để áp dụng nhiều chiến lược Evaluation phức tạp sau này (ví dụ: Data Association nối track trong cùng một video, hoặc Global Re-ID trên toàn bộ dataset).
- **Mở rộng hỗ trợ dữ liệu:** Hỗ trợ xử lý toàn diện 3 bộ dữ liệu: UAV-Anti-UAV, Anti-UAV-RGBT, và đặc biệt là bộ benchmark UAV123 (bộ này quy định sự kiện biến mất bằng toạ độ `NaN`). Toàn bộ ID của 3 bộ được cấp tự động và liên tục nhau không bị trùng lặp.

## 2. Kiến trúc Pipeline

Quy trình chuẩn bị dữ liệu gồm 3 script chạy nối tiếp nhau, tất cả đều được điều khiển bởi 1 file cấu hình YAML duy nhất:

1. `data_pipeline.py`: Xử lý bộ dữ liệu gốc `UAV-Anti-UAV`. Script này khởi tạo các thư mục đầu ra và sinh ra các file `query.json` / `gallery.json` ban đầu.
2. `process_rgbt_pipeline.py`: Đọc các file JSON vừa tạo, tiếp tục xử lý bộ `Anti-UAV-RGBT`, cấp `identity_id` nối tiếp từ ID lớn nhất hiện tại, và ghi nối (append) dữ liệu vào JSON.
3. `process_uav123_pipeline.py`: Xử lý bộ ảnh rời `UAV123` (parse toạ độ `NaN` để tìm sự kiện che khuất), cấp ID nối tiếp, và append vào JSON. Bộ dữ liệu này được tự động chia thành tập Train (80%) và Test (20%) bằng hàm băm (hash) từ tên video gốc. Kỹ thuật này đảm bảo các đối tượng chung một bối cảnh video (ví dụ `group1_1` và `group1_2`) sẽ luôn rơi vào cùng một tập, ngăn chặn triệt để hiện tượng rò rỉ dữ liệu (data leakage) giữa tập huấn luyện và tập kiểm thử.

## 3. Cấu hình Tham số (Config)
Toàn bộ thông số được tập trung tại file `./configs/config_local.yaml` (dưới mục `data_pipeline`). Các script sẽ được chạy với cú pháp `python script.py --config ./configs/config_local.yaml`.
Các tham số cốt lõi:
- `frame_stride`: Khoảng cách giữa các frame (bước nhảy).
- `num_before_frames`: Số lượng frame ngắt quãng cần lấy ngay trước khi UAV bị che khuất (làm Gallery).
- `num_after_frames`: Số lượng frame ngắt quãng cần lấy ngay sau khi UAV xuất hiện lại (làm Query).
- `crop_size`: Kích thước ảnh sau khi crop (mặc định 256x256).
- `bbox_padding`: Tỉ lệ nới rộng Bounding Box khi crop (mặc định 0.2 = 20%) để lấy thêm bối cảnh.

## 4. Định nghĩa Query, Gallery và Quản lý Identity
- **Gallery (Tập tham chiếu):** Bao gồm toàn bộ các khung hình (frames) của đối tượng **ngay trước khi** xảy ra sự kiện che khuất. Đại diện cho "trí nhớ ngắn hạn" của mô hình về ngoại hình gần nhất của mục tiêu.
- **Query (Tập truy vấn):** Bao gồm toàn bộ các khung hình của đối tượng **ngay sau khi** xuất hiện lại từ vật cản. 

**Quy tắc cấp ID (identity_id) và thiết kế Test:**
- Mỗi file annotation (ví dụ `UAV-Anti-UAV_Train_000001.txt` hay `group1_2.txt` của UAV123) đặc trưng cho một đối tượng (một track/UAV riêng biệt). Pipeline sẽ cấp cho mỗi file này một `identity_id` hoàn toàn khác nhau.
- Nếu một con UAV bị che khuất và xuất hiện lại nhiều lần trong video (được gọi là các *events*), nó sẽ tạo ra nhiều Query và nhiều Gallery. Tuy nhiên, tất cả chúng đều mang chung một `identity_id`.
- **Ứng dụng lúc Test:** Việc tách biệt này cho phép mô hình dễ dàng tuỳ biến luật tìm kiếm. Mặc định Query sẽ đi tìm ID trong một kho Gallery khổng lồ (từ mọi video). Nhưng bằng cách thêm code lọc `sequence_id` (Tên video) vào script Test, ta có thể giới hạn Query chỉ so sánh với các UAV ở trong chung một video để thực hiện nối track (Data Association).

## 5. Kết quả Đầu ra (Output)
Cấu trúc cây thư mục đầu ra chuẩn mực trong `output_dir` (VD: `./processed`):
```text
processed/
├── train/
│   ├── <sequence_id>_event_0/
│   │   ├── before/
│   │   │   ├── frame_0010.jpg
│   │   │   └── ... (Gallery frames)
│   │   └── after/
│   │       ├── frame_0050.jpg
│   │       └── ... (Query frames)
│   └── ...
├── query_train.json
├── gallery_train.json
├── query_test.json
└── gallery_test.json
```
Metadata JSON cung cấp đầy đủ thông tin về `sequence_id`, thời gian bị che khuất (`disappearance_duration_frames`), ngôn ngữ mô tả (`language_description`), và các cờ thuộc tính che khuất (`attributes`).

## 6. Note
Chỉ khi tôi nói, bạn mới code thêm 1 dòng filter nhỏ: Chỉ lấy những mẫu Gallery có cùng sequence_id (cùng video) với Query để so sánh