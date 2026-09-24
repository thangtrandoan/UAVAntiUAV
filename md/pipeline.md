# Báo cáo UAV ReID Pipeline

> **Quy ước ký hiệu.** Tài liệu này cố ý **KHÔNG ghi giá trị cụ thể** — mọi giá trị đọc từ
> config lúc chạy. Bảng dưới là ánh xạ ký hiệu → key config.
>
> | Ký hiệu | Key trong config | Ý nghĩa |
> | :--- | :--- | :--- |
> | `N` | `train.num_frames` / `infer.num_frames` | số frame của một clip temporal |
> | `s` | `data_pipeline.frame_stride` | bước thời gian giữa 2 frame liên tiếp trong clip |
> | `D_vis` | suy ra từ `backbone` | số chiều đặc trưng hình dáng |
> | `D_tmp` | `d_out` của temporal encoder | số chiều đặc trưng thời gian |
> | `T_gap` | `infer.t2_search_gap_tolerance` | số frame vắng **liên tiếp** tối đa trước khi reset cửa sổ |
> | `A` / `R` | `infer.max_anchor_size` / `max_recent_size` | sức chứa 2 tầng memory bank |
> | `Δ` | `infer.update_interval_sec` | chu kỳ cập nhật memory bank (giây **thời gian video**) |
> | `n_cand` | — | số UAV ứng viên xuất hiện cùng lúc ở T2 |

---

## I. Các Module Cốt lõi và Lý do sử dụng
Dưới đây là các thành phần chính được lựa chọn để tối ưu hóa pipeline:

- **Visual Backbone (CNN - GASNet/ResNet50-IBN):**
  - **Công dụng:** Trích xuất đặc trưng không gian tĩnh (hình dáng, màu sắc) từ **từng frame đơn lẻ** → vector `D_vis` chiều.
  - **Lý do:** Biến thể IBN (Instance-Batch Normalization) giúp ổn định nhận diện bất chấp thay đổi ánh sáng mạnh ngoài trời.

- **Temporal Encoder (chọn qua `temporal_type`):**
  - **Công dụng:** Rút trích đặc trưng thời gian (tốc độ, quỹ đạo) từ chuỗi **`N` frame** → vector `D_tmp` chiều.
  - **Hai lựa chọn:**
    - `mamba` — SSM/S6, độ phức tạp **O(N)**, phù hợp Edge AI. Bản fallback thuần PyTorch dùng khi thiếu `mamba_ssm`.
    - `attention` — Self-Attention Transformer, độ phức tạp **O(N²)** nhưng `N` ngắn nên không đáng kể; **bidirectional tự nhiên** (không phải lật chuỗi thủ công) và không phụ thuộc `mamba_ssm`.
  - **Lưu ý:** hai loại **KHÔNG tương thích trọng số** — đổi `temporal_type` bắt buộc train lại từ đầu.

- **Ngân Hàng Ký Ức (2-Tier Memory Bank):**
  - **Công dụng:** Quản lý vector đặc trưng qua 2 kho: Anchor (gốc, sức chứa `A`) và Recent (`R` mẫu gần nhất).
  - **Lý do:** Chống "loãng" đặc trưng khi UAV thay đổi góc nhìn liên tục.

- **Cơ chế lọc:**
  - **Công dụng:** Phân cấp xác thực thành 2 bước: Quét nhanh (Thô) và Theo dõi sâu (Tinh).
  - **Lý do:** Tiết kiệm tài nguyên bằng cách loại bỏ UAV "rác" trước khi phân tích chuyên sâu.

---

## II. Pipeline

### 0. Quy ước lấy mẫu clip (áp dụng CHUNG cho train, infer và eval)

Mỗi clip `N` frame được cắt từ danh sách frame của sequence theo **cùng một quy tắc** ở cả
3 nơi (nếu lệch nhau, temporal token bị lệch phân phối giữa train và lúc chạy thật):

- **Bước thời gian giữa 2 frame liên tiếp = `s`** (lấy cách quãng đều), **không** nội suy trải
  đều cả danh sách — vì nội suy làm bước thời gian hiệu dụng lớn hơn `s`.
- **Gallery (trước khi mất dấu): lấy `N` frame CUỐI** → sát thời điểm mất dấu `t1`.
- **Query (sau khi tái xuất): lấy `N` frame ĐẦU** → sát thời điểm tái xuất `t2`.
- Danh sách **ngắn hơn `N`**: lặp frame cuối cho đủ `N`.
- Danh sách **dài hơn `N`**: cắt **liền mạch** (contiguous).

### 1. Giai đoạn T0: Khởi tạo và Lưu trữ Ký ức
- **Trích xuất hình ảnh:** Bộ detect và tracking tạo bounding box, crop ảnh, resize về `crop_size`, chuẩn hoá rồi đưa vào mạng.
- **Trích xuất Đặc trưng Hình dáng vật thể:** Tạo vector `D_vis` chiều cho các frame **lấy cách quãng `s`**, lưu vào dạng cửa sổ trượt.
- **Trích xuất Đặc trưng Thời gian:** Trích xuất đặc trưng chuyển động từ **`N` frame** của cửa sổ trượt.
- **Dung hợp & Lưu trữ Vector đại diện:** Tính **trung bình có trọng số theo điểm độ nét** của đặc trưng hình dáng, nối với đặc trưng temporal → 1 vector `D_vis + D_tmp` chiều đại diện cho vật thể, chuẩn hóa và cập nhật vào Anchor/Recent Bank **2 vector**:
  - Vector `D_vis` chiều mang thông tin hình dáng — dùng cho **lọc thô**.
  - Vector `D_vis + D_tmp` chiều mang thêm thông tin thời gian — dùng cho **xác thực tinh**.
- **Xây dựng Ngân Hàng Ký Ức (2-Tier Memory Bank):**
  Cứ mỗi `Δ` giây **thời gian video** (`time_source`), hoặc ngay trước khi mất dấu, temporal encoder trích xuất đặc trưng từ cửa sổ trượt rồi cập nhật Vector đại diện vào memory bank.
  - **Anchor Bank (size = `A`):** Lưu trữ các vector đại diện đầu tiên làm "Hình dáng gốc ban đầu" khi dễ dàng bắt được hình ảnh của UAV. Ký ức này là bất biến trong suốt video.
  - **Recent Bank (size = `R`):** Cập nhật và lưu trữ các vector đại diện gần nhất theo dạng cửa sổ trượt. Điều này giúp hệ thống cập nhật các thay đổi ngoại hình khi UAV xoay góc hoặc đi qua vùng ánh sáng khác.
  - **Công dụng:** Xây dựng memory bank giúp ghi nhớ đặc trưng vật thể 1 cách tổng quát, nhiều góc nhìn hơn là chỉ ghi nhớ trước khi biến mất; cơ chế chia ra 2 tier memory bank giúp giải quyết vấn đề giảm hiệu năng khi tracking.

### 2. Giai đoạn T1: Trạng thái Mất track
Khi UAV mục tiêu biến mất khỏi khung hình (bay ra sau tòa nhà, chui vào đám mây, hoặc bị mờ nhòe khiến Object Detection thất bại), bộ bám sát Tracking sẽ bị đứt gãy.
- **Ghi nhận** trạng thái LOST và lưu tọa độ cuối cùng.
- **Khóa và bảo toàn Memory Bank:** Toàn bộ bộ nhớ trong Anchor Bank và Recent Bank được khóa lại và bảo toàn nguyên vẹn.

### 3. Giai đoạn T2: Xuất hiện lại
Một hoặc nhiều UAV đột ngột xuất hiện lại trong khung hình (ví dụ: phát hiện `n_cand` chiếc UAV xuất hiện cùng lúc). Hệ thống phải tìm ra đúng chiếc UAV mục tiêu cũ. Quá trình chọn lọc diễn ra qua 2 bước:

#### Bước 3.1: Lọc Thô (SOFT LOCK)
- Trích xuất đặc trưng **ảnh tĩnh** của các UAV ứng viên bằng GASNet và tính điểm soft lock là độ tương đồng với các vector `D_vis` chiều trong memory bank.
- **Loại trừ:** Bất kỳ UAV nào có điểm soft lock cao nhất < `soft_lock_threshold` sẽ bị đánh giá là khác biệt hoàn toàn và bị loại bỏ ngay lập tức khỏi quy trình kiểm tra, giúp giảm chi phí tính toán và giảm được nhiễu.
- **Khóa tạm thời (Soft Lock):** Hệ thống chọn ra chiếc UAV có điểm soft lock cao nhất (phải > `soft_lock_threshold`). Gán nhãn SOFT LOCK. Lúc này UAV sẽ tự động bám theo chiếc này để theo dõi chuyển động, chuẩn bị cho bước thẩm định kỹ hơn.
- **Cửa sổ SOFT LOCK — `N` frame LIÊN TỤC (bước 1):** chỉ để **chọn ứng viên / xem điểm**, KHÔNG dùng để chốt. Thu liên tục nên phản ứng nhanh, chỉ cần `N` frame là đủ.

#### Bước 3.2: Xác Thực Chuyên Sâu (HARD LOCK)
- Những chiếc UAV trong danh sách Soft Lock sẽ được bám sát để thu đủ **`N` MẪU, mỗi mẫu cách nhau `s` frame**.
- **Hệ quả số học:** cần `(N − 1) · s + 1` **frame video** liên tục (vì chỉ lấy mẫu mỗi `s` frame). Đây là cửa sổ **dài hơn** cửa sổ soft lock.
- Lúc này, hệ thống đã thu thập đủ chuyển động thực sự của cánh quạt và đường bay. Nó trích xuất lại vector `D_vis + D_tmp` chiều hoàn chỉnh và tiến hành so khớp với Memory Bank để ra được **Điểm Tương đồng cuối** (Fine Score, lấy max điểm tương đồng với các vector trong memory bank), quá trình này sẽ được thực hiện song song.
- UAV có điểm cao nhất và fine score > `reid_threshold` sẽ chuyển sang HARD LOCK, giải phóng danh sách soft lock, chuyển sang giai đoạn tiếp theo.
- Không có fine score nào > `reid_threshold`: quay lại trạng thái mất track bắt đầu tìm kiếm lại.
- **Dung sai vắng mặt `T_gap` (trong T2_SEARCH):** trong lúc thu cửa sổ HARD LOCK, nếu target vắng **liên tiếp** quá `T_gap` frame thì mới **reset** cửa sổ về 0. Vắng ngắn (detection dropout) chỉ **bỏ qua frame đó và giữ** các mẫu đã thu.
  - **Lý do:** cửa sổ hard lock cần `(N − 1) · s + 1` frame liên tục; nếu reset ngay ở 1 frame vắng thì những event tái xuất ngắn sẽ **vĩnh viễn không bao giờ lock được**. `T_gap = 0` = hành vi cũ (reset ngay).

> **Phân vai bước lấy mẫu — quan trọng, dễ nhầm:**
> | Cửa sổ | Bước | Số frame video cần | Vai trò |
> | :--- | :--- | :--- | :--- |
> | SOFT LOCK | **1** (liên tục) | `N` | nhanh, chỉ chọn ứng viên |
> | HARD LOCK | `s` (cách quãng) | `(N − 1) · s + 1` | chính xác, mới dùng để chốt |
> | Memory Bank / tracking | `s` (cách quãng) | `(N − 1) · s + 1` | khớp bước thời gian với lúc train |
>
> HARD LOCK và Memory Bank **phải** cùng bước `s` — nếu không, temporal token lệch phân phối
> so với lúc train. SOFT LOCK cố ý dùng bước 1 vì nó chỉ so vector hình dáng (`D_vis`), không
> dùng nhánh temporal.

### 4. Giai đoạn T3: Cơ chế chống nhầm mục tiêu
Tương tự giai đoạn T0, nhưng sẽ có thêm cơ chế chống bám nhầm: trong `hijack_check_count` lần cập nhật recent bank đầu sau khi hard lock, sẽ tính toán độ tương đồng với các vector còn lại trong memory bank, nếu độ tương đồng cao nhất < `hijack_threshold` thì quay về trạng thái mất track để tìm kiếm lại.

---

## III. Phân Tích Chi Phí Tính Toán & Khả năng Đáp ứng Real-time

Ký hiệu chi phí (đo trên phần cứng mục tiêu, **không cố định trong tài liệu**):

| Ký hiệu | Ý nghĩa |
| :--- | :--- |
| `T_vis` | thời gian backbone trích đặc trưng **1 frame** |
| `T_tmp` | thời gian temporal encoder chạy **1 chuỗi `N` frame** |
| `fps_v` | FPS của video đầu vào |
| `n_cand` | số UAV ứng viên ở T2 |

**Điều kiện "theo kịp" tổng quát** — tải trung bình mỗi frame phải ≤ `1 / fps_v`:

```
T_vis / s  +  T_tmp / (Δ · fps_v)   ≤   1 / fps_v
```

Số hạng thứ nhất là backbone (chỉ chạy mỗi `s` frame); số hạng thứ hai là temporal encoder
(chỉ chạy mỗi `Δ` giây video).

| Giai đoạn | Tác vụ xử lý | Khối lượng tính toán | Đánh giá khả năng đáp ứng & Tối ưu |
| :--- | :--- | :--- | :--- |
| **T0 & T3 (Bám sát bình thường)** | Trích xuất đặc trưng hình dáng liên tục | `T_vis / s` mỗi frame | **An toàn.** Vì dùng cửa sổ trượt với bước `s`, backbone không cần chạy trên 100% frame. Phần dư ra dành cho luồng Tracking. |
| | Tổng hợp Vector & cập nhật temporal (mỗi `Δ` giây) | `T_tmp / (Δ · fps_v)` mỗi frame | **Không đáng kể.** Đây là lý do chọn temporal encoder có chi phí thấp: chỉ chạy 1 lần mỗi `Δ` giây video, không gây giật (lag spike). |
| **T2 — Lọc Thô (Bước 3.1)** | Trích xuất ảnh tĩnh cho `n_cand` UAV lạ | `n_cand · T_vis` (một lần) | Có thể dùng *Batching* để gom chung các patch ảnh lại xử lý một lượt nhằm hạ tổng thời gian. |
| **T2 — Xác Thực Chuyên Sâu (Bước 3.2)** | Trích xuất `N` mẫu cho UAV bị Soft Lock & chạy temporal | rải trên `(N − 1) · s + 1` nhịp + `T_tmp` một lần ở cuối | **Hiệu quả cao.** Việc quan sát được rải đều trong các nhịp của luồng Tracking, không bị dồn tính toán vào 1 lúc. Sau khi đủ mẫu, chỉ tốn thêm `T_tmp` để ra quyết định cuối. |
| **Các phép tính toán ma trận** | cosine similarity, trung bình có trọng số,… | không đáng kể so với `T_vis` | |
