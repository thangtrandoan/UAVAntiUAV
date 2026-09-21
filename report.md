# BÁO CÁO TUẦN

**Ngày báo cáo:** 15/9/2026

---

## 1. Tóm tắt

Tuần này em tập trung trả lời một câu hỏi: **vì sao `coarse score` (GASNet) rất cao (~0.99) mà
`fine score` (đầu ReID) lại thấp, khiến pipeline không bao giờ HARD LOCK?**

Kết quả chính: **model không hỏng, và lỗi không nằm ở Mamba hay BatchNorm như giả thuyết ban đầu.**
Embedding với BatchNorm vẫn ReID tốt. Vấn đề nằm ở:

1. **Cửa sổ temporal lúc train và lúc infer được dựng theo hai cách khác nhau**, khiến encoder
   nhận đầu vào lệch phân phối ở mọi độ dài chuỗi.

2. Frame stride đang chưa đồng bộ trong suốt pipeline.

---

## 2. Những việc đã làm

### 2.1. Định vị điểm số bị mất ở tầng nào

Trước đây chỉ đo được điểm cosine cuối cùng của pipeline fusion, nên biết "điểm thấp" nhưng không
biết mất ở tầng nào. Vậy nên em debug bằng cách in ra điểm của nhiều tầng khác nhau: `visual_mean`,
`visual_plain`, `temporal`, `raw` (trước bnneck), `fused` (sau bnneck).

### 2.2. Phép thử quyết định: tách "xếp hạng" khỏi "quyết định"

Mở rộng bộ đánh giá để so sánh **4 không gian đặc trưng** (`visual` / `temporal` / `pre_bn` /
`fused`) trong cùng một lượt chạy, đo song song **Rank-1/mAP** và **TAR@FAR ở nhiều mốc FAR**.
Mục đích: phân biệt "model không phân biệt được danh tính" với "model phân biệt được nhưng ngưỡng
tuyệt đối là sai công cụ".

Kèm theo: thử nghiệm **z-norm** để đo trần của hướng "sửa luật quyết định" mà không cần train lại.

### 2.3. Thí nghiệm tách ảnh hưởng của độ dài chuỗi

6 run trên **cùng một video**, chỉ đổi `num_frames` (và backbone ở 3 run sau). Số liệu lấy
**DEBUG đầu tiên của mỗi chu kỳ** — đúng khoảnh khắc quyết định — vì khi fine FAILED pipeline in
DEBUG mỗi frame nên trung bình toàn bộ dòng bị lệch nặng.

### 2.4. Đồng bộ `frame_stride` toàn pipeline

Phát hiện **bước thời gian** giữa các frame trong cửa sổ temporal không thống nhất giữa ba khâu:
tạo dữ liệu, train và infer. Đã đồng bộ để cả ba khâu dùng đúng
`frame_stride`, với nguyên tắc: **chỉ soft lock thu liên tục, mọi bước còn lại đều lấy cách quãng (stride đang để: 4)**.

---

## 3. Kết quả

### 3.1. Bác bỏ giả thuyết "BatchNorm1d là thủ phạm"

Giả thuyết ban đầu cho rằng `BatchNorm1d` trong đầu ReID khuếch đại các chiều ít biến thiên,
bẻ cong hướng vector và làm cosine sụp. **Số liệu bác bỏ điều này:**

- `pre_bn` (đầu vào bnneck) có cosine **cao hơn** `fused` (0.953 vs 0.734) nhưng **kém hơn về
  khả năng phân biệt** (Rank-1 67.32% < 78.15%). Nếu BN thật sự phá hoại thì bỏ BN phải tốt hơn.
⇒ **BN không phải thủ phạm chính.**

*(Giả thuyết thay thế — "mức điểm lệch nhau giữa các truy vấn" — được kiểm tra riêng ở mục 3.3.)*

### 3.2. Kiểm tra lại tác dụng của từng tầng mô hình

Ablation trên eval-split (1,699 cặp, mốc ngẫu nhiên Rank-1 = 2.90%):

| Không gian | dim | Rank-1 | mAP | TAR@FAR 0.1% |
|---|---|---|---|---|
| `visual` | 960 | 76.38% | 31.49% | **9.94%** |
| `temporal` | 512 | 65.24% | 26.90% | 5.29% |
| `pre_bn` | 1472 | 67.32% | 27.81% | 5.79% |
| `fused` | 1472 | **78.15%** | **35.59%** | 8.45% |

Đọc bảng này:

- **`temporal`-only đạt 65.24%**, gấp 22 lần mốc ngẫu nhiên → nhánh temporal **có học được
  thông tin danh tính**, không vô dụng.
- **`fused` tốt nhất về xếp hạng** (Rank-1 +1.77%, mAP +4.10% so với `visual`) → fusion có ích.
- **Nhưng `visual`-only lại tốt nhất về TAR@FAR 0.1%** (9.94%) → temporal giúp **xếp hạng**
  nhưng **làm hại điểm làm việc theo ngưỡng tuyệt đối**.

⇒ Cần tách bạch: **xếp hạng** và **cổng quyết định** là hai bài toán khác nhau.

### 3.3. Thử nghiệm z-norm: bác bỏ giả thuyết "mức điểm lệch nhau"

**Giả thuyết cần kiểm tra.** Bảng ở mục 3.2 cho một nghịch lý:

| Chỉ số | Yêu cầu | Kết quả |
|---|---|---|
| Rank-1 | cặp genuine phải **thắng các cặp khác trong cùng truy vấn** | 78.15% |
| TAR@FAR 0.1% | một **ngưỡng cosine tuyệt đối dùng chung cho mọi truy vấn** phải tách được | 8.45% |

Giả thuyết được đặt ra: **thứ tự trong mỗi truy vấn thì đúng, nhưng MỨC điểm không so sánh được
giữa các truy vấn.** Mỗi truy vấn có offset/scale riêng; một nhóm nhỏ truy vấn có impostor điểm rất
cao kéo phân vị 99.9% toàn cục lên 0.852, và ngưỡng đó vô hiệu hoá phần lớn truy vấn còn lại.

Nếu giả thuyết này đúng thì nút thắt nằm ở **luật quyết định** (cách so điểm), **không** nằm ở chất
lượng embedding — và đây là hướng sửa rẻ nhất vì không cần train lại.

**z-norm là phép thử đúng cho giả thuyết này.** Thay vì so điểm thô với ngưỡng chung, chuẩn hoá
điểm theo từng truy vấn:

```
z = (s − μ_i) / σ_i      μ_i, σ_i = trung bình / độ lệch của điểm truy vấn i so với một cohort
```

Vì đây là biến đổi **affine theo từng hàng** nên **Rank-1 không đổi** — chỉ ngưỡng/DET thay đổi.
Nghĩa là nếu giả thuyết đúng thì TAR phải **tăng mạnh**, và kết quả đo được chính là **trần** của
hướng "sửa luật quyết định".

**Hai biến thể cohort — và một thiên vị cần loại bỏ.** Lần đo đầu dùng `cohort='all'` (μ/σ trên
**toàn bộ gallery**). Nhưng gallery có **~49 cặp genuine mỗi truy vấn** (2.9%), và chúng **chính là
các điểm cao nhất**. Đưa chúng vào μ/σ làm μ, σ phồng lên → **đè z-score của chính genuine xuống**.
Đây là **thiên vị chống z-norm**, không phải phép thử công bằng.

Nên đã thêm biến thể thứ hai `cohort='impostor'`: μ/σ **chỉ tính trên các cột khác danh tính** —
đây mới là **upper bound đúng** (t-norm kinh điển).

**Kết quả** (TAR@FAR 0.1%, so với **baseline của chính từng không gian**):

| Không gian | Baseline | `cohort='all'` | `cohort='impostor'` |
|---|---|---|---|
| `fused` | 8.45% | 6.13% (−2.32%) | 7.27% (**−1.18%**) |
| `pre_bn` | 5.79% | 4.78% (−1.01%) | 5.65% (**−0.13%**) |
| `visual` | 9.94% | 4.65% (−5.29%) | 5.19% (**−4.75%**) |
| `temporal` | 5.29% | 4.63% (−0.67%) | **5.51% (+0.22%)** |

**Vì sao "+0.22%" bác bỏ giả thuyết:**

1. Bản `cohort='impostor'` **đúng là tốt hơn** bản `all` ở cả 4 không gian → phép thử đã công bằng.
2. Nhưng ngay cả bản công bằng, kết quả tốt nhất chỉ **+0.22%** (`temporal` 5.29% → 5.51%), và
   **3/4 không gian còn tệ hơn** — `visual` mất tới **4.75%**.
3. Nếu "mức điểm lệch giữa các truy vấn" thật sự là nút thắt, thì xoá đúng cái lệch đó phải kéo
   TAR lên **gần mức Rank-1 (78%)**, chứ không phải nhích **0.22 điểm phần trăm**.

⇒ **Điểm cosine vốn đã tương thích giữa các truy vấn.** Offset/scale không phải vấn đề, và
**luật quyết định không phải nút thắt.**

**Hệ quả — mất mát nằm ở chỗ khác.** Bác bỏ giả thuyết này không có nghĩa luật quyết định đã hoàn
hảo, mà có nghĩa mất mát không nằm ở đó:

- Ngay ở **FAR 10%** (rất dễ dãi) TAR vẫn chỉ **~52%** → ngoài luật quyết định, **chất lượng cặp
  genuine cũng còn yếu**, khớp với mAP chỉ 35.59%.
- Offline TAR **38.86%** tại đúng ngưỡng đang chạy (0.75), nhưng online lock **≤ 6.08%** → cửa sổ
  T2_SEARCH **khó hơn dữ liệu offline ~6×**. Nút thắt nằm ở **chất lượng cửa sổ / bank tham chiếu**.

**Lưu ý phương pháp:** `cohort='impostor'` **dùng nhãn** để chọn cohort → triển khai thật phải
thay bằng một cohort tham chiếu có nhãn cố định. Nên đây là **trần lý thuyết**, không phải phương
pháp dùng được ngay. Mà trần chỉ **+0.22%** thì hướng này coi như hết.

### 3.4. `num_frames` đang là "nút bịt", không phải siêu tham số

6 lần chạy trên cùng video, chỉ đổi `N`:

| Backbone | N_train | N_infer | temporal | visual_plain |
|---|---|---|---|---|
| resnet50-ibn | 12 | 8 | **0.977** | 0.990 |
| resnet50-ibn | 12 | 12 | **0.349** ⬇ | 0.990 |
| resnet50-ibn | 12 | 16 | 0.680 | 0.991 |
| dinov3-convnext | 8 | 8 | **0.579** ⬇ | 0.986 |
| dinov3-convnext | 8 | 12 | 0.939 | 0.990 |
| dinov3-convnext | 8 | 16 | **0.958** | 0.994 |

Ba thứ rút ra được:

1. **Nhánh thị giác miễn nhiễm với `N`** — `visual_plain` ∈ [0.986, 0.994] ở **cả 6 run**, kể cả
   run tệ nhất. Cùng video, cùng bbox ⇒ khung hình/backbone **không phải** vấn đề.
2. **Chỉ nhánh temporal sụp, và sụp ngay tại `N = N_train`** (resnet 0.349@12, dinov3 0.579@8).
   Đây là hiệu ứng thuần theo **độ dài chuỗi**.
3. **`N` còn điều khiển số cơ hội quyết định** — thuần cơ học. Cửa sổ tái xuất trong video rất
   ngắn (có chu kỳ chỉ 8 frame), nên `N` lớn thì **không bao giờ đủ frame để ra quyết định**.

**Nguyên nhân gốc**: train và infer dựng chuỗi temporal theo hai cách khác nhau —
train lấy mẫu **trải đều cả event** (bước thời gian lớn), infer lấy **liên tục** (bước = 1).
Khớp `N = N_train` chỉ khớp **số lượng**, không khớp **động học**.

---

## 4. Hướng dự định làm tiếp

### 1. ĐỒng bộ frame stride cho toàn bộ pipeline và đánh giá lại

- **Ở infer**: cho cửa sổ dùng stride thích ứng để **khoảng thời gian**
  khớp lúc train.*
- **Ở train**: train với cùng cách lấy mẫu sẽ dùng lúc infer, **và thử randomize số frame**
  (ví dụ `N ∈ {8, 12, 16}` mỗi batch) để encoder **bất biến theo số lượng frame**.

### 2. Thử nghiệm với LayerNorm

Tuy Batchnorm không phải là nguyên nhân chính, nhưng vì ConvNext sử dụng Layernorm, kết hợp với bài báo "InceptionMamba: An Efficient Hybrid Network with Large Band Convolution and Bottleneck Mamba" cũng sử dụng Layernorm, nên đây là hướng đáng để thử nghiệm.

### 3. Nâng cấp backbone ConvNext

Thử các thay đổi lên backbone ConvNext như dùng SE block, eSE block, sử dụng Batchnorm trong ConvNext

### 4. `mamba_ssm` thật + train lại

Làm **sau khi** đã hoàn thiện được phần GASNet.


---