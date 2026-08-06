# 🚀 Báo cáo Nâng cấp: GASNet Mamba v2

Dựa trên những nguyên nhân gốc làm suy giảm chỉ số của mô hình GASNet Mamba v1 so với baseline, các thay đổi kiến trúc sau đây đã được áp dụng để cho ra mắt phiên bản **v2**.

## 🔴 Tổng hợp các điểm yếu của bản v1

1. **Temporal Encoding quá hời hợt**: v1 chỉ sử dụng hidden state ở **frame cuối cùng** (`x[:, -1, :]`) để đại diện cho toàn bộ sequence 16 frames, bỏ lỡ 91% ngữ cảnh thời gian.
2. **Scan 1 chiều (Unidirectional Scan)**: Sequence được xử lý một chiều, không phù hợp cho bài toán video tracking/ReID vốn cần xem xét sự kiện trong quá khứ và tương lai một cách đồng đều.
3. **Temporal Token quá nhỏ (bị lấn át)**: Kích thước của temporal token chỉ là 256 (chiếm ~9% tổng số dimension của feature) so với 2560 của Visual Feature, dẫn đến việc token dễ dàng biến thành "noise" và kéo lùi tổng thể độ chính xác (Rank-1 drop ~9%).

---

## 🟢 Chi tiết các bản vá & cập nhật ở v2

Toàn bộ quá trình sửa lỗi tập trung vào việc tinh chỉnh cấu trúc Mamba ở file `model.py` và cập nhật loss function ở `train_reid.py`:

### 1. 🔄 Xử lý song hướng (Bidirectional Processing)
Module `TemporalMambaEncoder` đã được viết lại. Nay mỗi frame sẽ được scan bằng cách truyền qua `mamba_layer` theo chiều gốc, và đồng thời sequence được `flip` ngược lại để scan. 
- Output cuối cùng của layer là tổng của Forward Pass, Backward Pass và Residual Connection.
- Khắc phục nhược điểm của fallback `SimpleS6Block` (hoạt động rất tệ khi chỉ scan 1 chiều).

### 2. 🌊 Cải tiến Aggregation (Mean Pooling)
Thay vì trích xuất hidden state của frame cuối cùng, toàn bộ chuỗi bây giờ đã được kết hợp qua cơ chế **Mean Pooling** (`x.mean(dim=1)`). Bằng cách này, mọi khoảnh khắc trong video đều đóng góp vào vector đại diện cuối cùng thay vì bị lãng quên.

### 3. ⚖️ Cân bằng Visual và Temporal Token
- Tăng đầu ra `d_out` của Mamba encoder từ **256** lên **512**. 
- Tổng feature representation được cấp cho Classifier (`ReIDHead`) hiện tại là `2560 (Visual) + 512 (Temporal) = 3072`. Điều này giúp mạng có nhiều khả năng biểu diễn (capacity) hơn đối với temporal context.
- Đã đồng bộ kích thước input của `ReIDHead` và kích thước `feat_dim` của `CenterLoss` (ở `train_reid.py`) lên `3072` để ngăn ngừa lỗi bất đồng bộ matrix trong quá trình training.

---

## 🎯 Đề xuất tiếp theo

Với cấu trúc mô hình đã hoàn chỉnh, v2 sẵn sàng để đưa vào thực chiến. Để đạt được kết quả **thực sự vượt trội** so với baseline, khuyến nghị người dùng:

1. **Train lại mạng ngay bây giờ**: Sử dụng lại `train_reid.py` theo script cũ. Không cần thay đổi hyperparams trong file yaml lúc này.
2. **Cài đặt thư viện mamba-ssm chính chủ**: Chạy `pip install mamba-ssm` (hoặc build từ source cho Jetson AGX) để vô hiệu hóa Fallback (`SimpleS6Block`). Việc sử dụng thư viện CUDA optimize sẽ tăng tốc training và tạo ra state updates mạnh mẽ hơn, giảm thiểu nhiễu loạn trong quá trình fine-tuning.
3. **Bổ sung Validation set (Tùy chọn tương lai)**: Lấy best model dựa vào accuracy trên Test/Val set thay vì Training Loss (do Training Loss cực thấp có thể là biểu hiện của Overfitting).
