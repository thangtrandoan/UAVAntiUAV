# Báo cáo UAV ReID Pipeline

## I. Các Module Cốt lõi và Lý do sử dụng
Dưới đây là các thành phần chính được lựa chọn để tối ưu hóa pipeline:

- **Visual Backbone (CNN - GASNet/ResNet50-IBN):**
  - **Công dụng:** Trích xuất đặc trưng không gian tĩnh (hình dáng, màu sắc) từ từng frame đơn lẻ.
  - **Lý do:** Biến thể IBN (Instance-Batch Normalization) giúp ổn định nhận diện bất chấp thay đổi ánh sáng mạnh ngoài trời.

- **Temporal Mamba Encoder:**
  - **Công dụng:** Rút trích đặc trưng thời gian (tốc độ, quỹ đạo) từ chuỗi 16 frames.
  - **Lý do:** Độ phức tạp tuyến tính O(N) của Mamba cho phép xử lý Real-time trên thiết bị Edge AI.

- **Ngân Hàng Ký Ức (2-Tier Memory Bank):**
  - **Công dụng:** Quản lý vector đặc trưng qua 2 kho: Anchor (gốc) và Recent (m frames gần nhất).
  - **Lý do:** Chống "loãng" đặc trưng khi UAV thay đổi góc nhìn liên tục.

- **Cơ chế lọc:**
  - **Công dụng:** Phân cấp xác thực thành 2 bước: Quét nhanh (Thô) và Theo dõi sâu (Tinh).
  - **Lý do:** Tiết kiệm tài nguyên bằng cách loại bỏ 90% UAV "rác" trước khi phân tích chuyên sâu.

---

## II. Pipeline

### 1. Giai đoạn T0: Khởi tạo và Lưu trữ Ký ức
- **Trích xuất hình ảnh:** Bộ detect và tracking tạo bounding box, crop ảnh, resize về 256x256, chuẩn hoá rồi đưa vào mạng.
- **Trích xuất Đặc trưng Hình dáng vật thể:** Tạo Vector 2560 chiều cho k frame (stride = 2 hoặc 3 để có thể tổng quát hoá), lưu vào dạng cửa sổ trượt.
- **Trích xuất Đặc trưng Thời gian:** Trích xuất đặc trưng chuyển động k frame từ cửa sổ trượt đấy.
- **Dung hợp & Lưu trữ Vector đại diện:** Tính trung bình có trọng số dựa trên điểm độ nét từ đặc trưng của k frame từ GASNet và đặc trưng từ Mamba, nối lại được 1 Vector 3072 chiều đại diện cho vật thể, chuẩn hóa và cập nhật vào Anchor/Recent Bank 2 vector:
  - Vector trung bình 2560 chiều mang thông tin hình dáng.
  - Vector 3072 chiều mang thêm đặc trưng thông tin về thời gian.
- **Xây dựng Ngân Hàng Ký Ức (2-Tier Memory Bank):**
  Cứ mỗi t giây hoặc ngay trước khi mất dấu, Mamba trích xuất đặc trưng từ cửa sổ trượt rồi cập nhật Vector đại diện vào memory bank.
  - **Anchor Bank (size = n):** Lưu trữ các vector đại diện đầu tiên làm "Hình dáng gốc ban đầu" khi dễ dàng bắt được hình ảnh của UAV. Ký ức này là bất biến trong suốt video.
  - **Recent Bank (size = m):** Cập nhật và lưu trữ các vector đại diện của m frame gần nhất theo dạng cửa sổ trượt. Điều này giúp hệ thống cập nhật các thay đổi ngoại hình khi UAV xoay góc hoặc đi qua vùng ánh sáng khác.
  - **Công dụng:** Xây dựng memory bank giúp ghi nhớ đặc trưng vật thể 1 cách tổng quát, nhiều góc nhìn hơn là chỉ ghi nhớ trước khi biến mất; cơ chế chia ra 2 tier memory bank giúp giải quyết vấn đề giảm hiệu năng khi tracking.

### 2. Giai đoạn T1: Trạng thái Mất track
Khi UAV mục tiêu biến mất khỏi khung hình (bay ra sau tòa nhà, chui vào đám mây, hoặc bị mờ nhòe khiến Object Detection thất bại), bộ bám sát Tracking sẽ bị đứt gãy.
- **Ghi nhận** trạng thái LOST và lưu tọa độ cuối cùng.
- **Khóa và bảo toàn Memory Bank:** Toàn bộ bộ nhớ trong Anchor Bank và Recent Bank được khóa lại và bảo toàn nguyên vẹn.

### 3. Giai đoạn T2: Xuất hiện lại
Một hoặc nhiều UAV đột ngột xuất hiện lại trong khung hình (ví dụ: phát hiện 3 chiếc UAV xuất hiện cùng lúc). Hệ thống phải tìm ra đúng chiếc UAV mục tiêu cũ. Quá trình chọn lọc diễn ra qua 3 bước:
- **Bước 3.1: Lọc Thô** 
  Trích xuất đặc trưng ảnh tĩnh đầu tiên của các UAV bằng GASNet và tính điểm soft lock là độ tương đồng với các vector 2560 chiều trong memory bank.
  - **Loại trừ:** Bất kỳ UAV nào có điểm soft lock cao nhất < `soft_lock_threshold` sẽ bị đánh giá là khác biệt hoàn toàn và bị loại bỏ ngay lập tức khỏi quy trình kiểm tra, giúp giảm chi phí tính toán và giảm được nhiễu.
  - **Khóa tạm thời (Soft Lock):** Hệ thống chọn ra chiếc UAV có điểm soft lock cao nhất (phải > `soft_lock_threshold`). Gán nhãn SOFT LOCK. Lúc này UAV sẽ tự động bám theo chiếc này để theo dõi chuyển động, chuẩn bị cho bước thẩm định kỹ hơn.
- **Bước 3.3: Xác Thực Chuyên Sâu** 
  Những chiếc UAV trong danh sách Soft Lock sẽ được bám sát trong k frame liên tiếp tiếp theo.
  Lúc này, hệ thống đã thu thập đủ chuyển động thực sự của cánh quạt và đường bay. Nó trích xuất lại Vector 3072 chiều hoàn chỉnh và tiến hành so khớp với Memory Bank để ra được Điểm Tương đồng cuối (Fine Score, lấy max điểm tương đồng với các vector trong memory bank), quá trình này sẽ được thực hiện song song.
  - UAV có điểm cao nhất và fine score > `reid_threshold` sẽ chuyển sang HARD LOCK, giải phóng danh sách soft lock, chuyển sang giai đoạn tiếp theo.
  - Không có fine score nào > `reid_threshold`: quay lại trạng thái mất track bắt đầu tìm kiếm lại.

### 4. Giai đoạn T3: Cơ chế chống nhầm mục tiêu
Tương tự giai đoạn T0, nhưng sẽ có thêm cơ chế chống bám nhầm: trong x lần cập nhật recent bank đầu sau khi hard lock, sẽ tính toán độ tương đồng với các vector còn lại trong memory bank, nếu độ tương đồng cao nhất < `hijack_threshold` thì quay về trạng thái mất track để tìm kiếm lại.

---

## III. Phân Tích Chi Phí Tính Toán & Khả Năng Đáp Ứng Real-time

Dựa trên chỉ số hiệu năng thực tế của các mô hình trên thiết bị phần cứng mục tiêu:
- **Visual Backbone (GASNet):** mất **~50 ms / frame**.
- **Temporal Encoder (Mamba):** mất **~8.5 ms / chuỗi**.

Dưới đây là bảng phân tích chi phí nhằm đánh giá hệ thống có bị "nghẽn cổ chai" (bottleneck) và có theo kịp tốc độ của luồng Tracking hay không:

| Giai đoạn | Tác vụ xử lý | Khối lượng tính toán (Ước lượng) | Đánh giá khả năng đáp ứng & Tối ưu |
| :--- | :--- | :--- | :--- |
| **Giai đoạn T0 & T3 (Bám sát bình thường)** | Trích xuất đặc trưng hình dáng (GASNet) liên tục | **~50 ms / frame** (Tương đương tốc độ Infer 20 FPS cho riêng ReID) | **An toàn.** Vì ta dùng cơ chế cửa sổ trượt với **stride = 2 hoặc 3**, GASNet không cần chạy trên 100% frame. Nếu video 30FPS (33ms/frame), việc chia tải (asynchronous) hoặc trích xuất ngắt quãng giúp Tracking (10-15ms/frame) vẫn mượt mà. |
| | Tổng hợp Vector & Cập nhật Mamba (mỗi t giây) | **~8.5 ms / lượt** (Cực nhanh) | **Hoàn toàn theo kịp.** Độ phức tạp tuyến tính O(N) của Mamba thể hiện thế mạnh, chỉ mất khoảng 8.5ms cho 1 đợt cập nhật ngân hàng ký ức, không gây giật (lag spike). |
| **Giai đoạn T2 (Xuất hiện lại) - Lọc Thô (Bước 3.1)** | Trích xuất ảnh tĩnh đầu tiên cho **N** chiếc UAV lạ | **50 ms** | Có thể dùng *Batching* để gom chung các patch ảnh lại xử lý một lượt nhằm hạ tổng thời gian. |
| **Giai đoạn T2 - Xác Thực Chuyên Sâu (Bước 3.3)** | Trích xuất liên tục k frames cho UAV bị Soft Lock & chạy Mamba. So sánh vector. | Chạy song song với Tracking + **8.5 ms** (Mamba) + **< 1 ms** (So khớp cosine) | **Hiệu quả cao.** Việc quan sát k frames được rải đều trong k nhịp của luồng Tracking, không bị dồn tính toán vào 1 lúc. Sau khi đủ frame, mất vỏn vẹn ~10ms để ra quyết định cuối (Hard Lock hoặc bỏ qua). |
| **Các phép tính toán ma trận** | Tính toán tương đồng cosine, tính toán trung bình,... | **~ 1 ms** | Không đáng kể | 

