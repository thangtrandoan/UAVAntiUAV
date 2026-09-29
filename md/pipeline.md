# Báo cáo UAV ReID Pipeline

---

# 🔒 PIPELINE ĐÓNG BĂNG (chốt 29/9/2026)

**Từ nay CHỈ SỬA MODEL.** Mọi tham số pipeline là hằng số.

| | |
| :--- | :--- |
| 🔓 **Biến tự do DUY NHẤT** | `train.temporal_type` = `"mamba"` \| `"attention"` (+ mọi thứ trong package `model/`, xem §dưới) |
| 🔒 **Nguồn sự thật** | `configs/config_colab.yaml` (block `train` + `data_pipeline`) |
| 🔒 **Cơ chế ép** | `pipeline_lock.py` → `resolve_pipeline(cfg, <section>)`. Mọi script ĐỌC LẠI `num_frames`/`stride`/`backbone`/`bbox_padding` từ nguồn và **IN CẢNH BÁO** nếu section của nó ghi lệch |
| 🔒 **Kiểm tra hằng số** | `assert_frozen(cfg)` — chạy ở đầu `train_reid.py`, cảnh báo nếu ai mở lại biến đã khóa |
| 📋 **Config ablation** | `configs/ablation/config_mamba.yaml` + `config_attention.yaml` — **sửa TAY**, phải giữ chỉ khác nhau **5 dòng** (4 đường dẫn + `temporal_type`). Kiểm bằng `diff` (xem §dưới) |
| 📌 **Truy vết** | `calibrated_threshold.json → provenance` ghi `num_frames`, `frame_stride`, `temporal_type`, `temporal_pool`, `backbone` |

### Hằng số đã chốt

| Tham số | Giá trị | Vì sao |
| :--- | :--- | :--- |
| `backbone` | `dinov3_convnext` | M1/M2 dùng; `D_vis=960`, `fused=1472` |
| `num_frames` | **12** | = M1/M2 = bản tham chiếu 15/9 |
| `n_frames_choices` | **KHÔNG KHAI BÁO** | Memory bank chỉ cập nhật khi `is_ready()` (= đủ `N` frame) và HARD LOCK cũng chờ đủ `N` mẫu ⇒ lúc ra quyết định `N` **luôn** = `num_frames`. Random N làm BN running-stats thành **hỗn hợp** qua nhiều `N` (không `N` nào khớp) |
| `frame_stride` | **4** | Bước thời gian, nguồn duy nhất; `infer.py` ép `infer.stride` theo |
| `temporal_pool` | **`attn`** | Tổ hợp lồi (softmax tổng = 1) ⇒ scale bất biến theo `N`. `mean` chỉ bất biến khi frame iid |
| `temporal_pe` | **true** | = bản tham chiếu (có `pos_embed`) |
| `√N` ở nhánh `mean` | **ĐÃ BỎ** | `Var(mean) ≈ σ²(ρ + (1−ρ)/N)`; frame UAV có ρ ≈ 0.9 nên `mean·√N` làm **scale tăng theo N** |
| `lam2` | **0.0** | `TemporalConsistencyLoss` bản cũ có nghiệm tầm thường (độc hại), bản mới bão hoà 0.0000 từ epoch 2 (trơ). Tắt **tường minh** |
| `batch_size` / `num_instances` | **12 / 3** | = bản tham chiếu |
| `stage1` / `stage2` epochs | **30 / 30** | Ngân sách cố định |
| `stop_on_target` / `early_stop_patience` | **false / 0** | BẮT BUỘC TẮT: nếu bật, hai model dừng ở epoch khác nhau ⇒ so ở hai lượng train khác nhau ⇒ nhiễu |
| `val_n_list` | `[8, 12, 16]` | **Chỉ ĐỂ ĐO** độ bền theo `N` (diagnostic). Không ảnh hưởng train |
| `fine_space` | **`fused`** | TAR@FAR tốt nhất offline (8.45% vs 5.79% của `pre_bn` ở FAR 0.1%) |
| `soft_lock_threshold` | **0.0** | Đo thực tế coarse score = 0.98–0.99 ⇒ ngưỡng 0.3 **chưa bao giờ chặn ai**. `0.0` cho trung thực với hành vi thật: cổng thô chỉ **CHỌN MAX**, không **LỌC** |
| `reid_threshold` | 0.75 | = `thresholds.fused`. **PHẢI calibrate lại mỗi khi đổi model** |
| `update_interval_sec` / `time_source` | 2.0 / `video` | Đếm theo thời gian **video**, không theo đồng hồ tường |
| `max_anchor_size` / `max_recent_size` | 5 / 15 | |
| `t2_search_gap_tolerance` | 2 | Số frame vắng **liên tiếp** tối đa trước khi reset cửa sổ HARD LOCK |

### Khi đổi model, phải đổi 4 đường dẫn

| # | Key | Thành |
| :--- | :--- | :--- |
| 1 | `paths.checkpoint_dir` | thư mục riêng cho model mới |
| 2 | `paths.log_dir` | thư mục riêng |
| 3 | `eval.output_dir` | thư mục riêng |
| 4 | `infer.out_dir` | thư mục riêng |

Đổi **bất kỳ key nào khác** ⇒ kết quả **KHÔNG** so được với các run trước.

### Quy trình chuẩn

```bash
# 0. Kiểm 2 config ablation không trôi khỏi nhau (xem quy tắc ngay dưới)
diff configs/ablation/config_mamba.yaml configs/ablation/config_attention.yaml

# 1. Train 2 nhánh (BẮT BUỘC cả hai — chỉ một nhánh thì không có gì để so)
python train_reid.py --config configs/ablation/config_mamba.yaml
python train_reid.py --config configs/ablation/config_attention.yaml

# 2. Calibrate ngưỡng (ghi kèm provenance)
python calibrate_threshold.py --config configs/ablation/config_attention.yaml

# 3. Eval offline (Rank-1 / mAP / TAR@FAR)
python evaluate_reid.py --config configs/ablation/config_attention.yaml

# 4. Infer online (T0→T3, HARD LOCK)
python infer.py --config configs/ablation/config_attention.yaml
```

> 📋 **Quy tắc sửa 2 config ablation (sửa tay):** hai file **PHẢI** giống nhau, chỉ khác
> **5 dòng**: `train.temporal_type`, `paths.checkpoint_dir`, `paths.log_dir`, `eval.output_dir`,
> `infer.out_dir`. Sửa bất kỳ dòng nào khác ⇒ phải sửa **cả hai file**. Kiểm bằng:
> ```bash
> diff configs/ablation/config_mamba.yaml configs/ablation/config_attention.yaml
> ```
> Nếu `diff` ra dòng nào **không phải** 5 dòng trên ⇒ hai nhánh đã trôi khỏi nhau, kết quả
> so sánh **không còn hợp lệ**. `configs/config_colab.yaml` là bản gốc để đối chiếu.

> ⚠️ **Đổi model ⇒ phải calibrate lại `reid_threshold`.** Ngưỡng là thuộc tính của
> **thang điểm của model**, không phải hằng số của pipeline.

---

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

### Package `model/` — nơi DUY NHẤT được sửa khi thử nghiệm model

`model.py` cũ (791 dòng) đã tách thành package. Mọi script vẫn chỉ gọi
`from model import UAVReIDNet, load_checkpoint_verbose` — **không đổi**.

```
model/
  __init__.py            API công khai + ràng buộc cần giữ
  components.py          weights_init_*, AttentionPooling, ReIDHead
  temporal_mamba.py      SimpleS6Block + TemporalMambaEncoder
  temporal_attention.py  TemporalAttentionEncoder
  registry.py            TEMPORAL_ENCODERS + build_temporal_encoder()
  reidnet.py             UAVReIDNet (backbone + temporal encoder + head)
  checkpoint.py          load_checkpoint_verbose()
```

**Thêm temporal encoder mới — chỉ 2 việc:**

1. Tạo `model/temporal_<ten>.py`, ví dụ:

   ```python
   from .components import AttentionPooling
   from .registry import register_temporal

   @register_temporal('my_encoder')
   class MyEncoder(nn.Module):
       def __init__(self, d_in=2560, d_model=512, d_out=512, max_seq_len=64,
                    num_layers=2, pool='attn', use_pe=True, **kwargs): ...
       def forward(self, x):        # [B, N, d_in] -> ([B, d_out], [B, N, d_model])
           ...
   ```

2. Thêm 1 dòng import vào `model/__init__.py`, rồi đặt
   `train.temporal_type: "my_encoder"` trong config. **Không sửa `reidnet.py`.**

Bỏ kwarg không dùng được tự động: `TemporalMambaEncoder` không có `num_heads`/`dropout`
nên `build_temporal_encoder` lọc theo chữ ký của từng lớp.

**Ba ràng buộc phải giữ** (nếu vi phạm, checkpoint cũ không nạp được hoặc pipeline vỡ):

| Ràng buộc | Vì sao |
|---|---|
| `UAVReIDNet` giữ tên `self.backbone` / `self.temporal_encoder` / `self.head` | key state_dict sinh từ tên thuộc tính |
| Encoder giữ interface `[B, N, d_in] -> ([B, d_out], [B, N, d_model])` | `reidnet.py` và mọi script dựa vào |
| Temporal pooling là tổ hợp lồi (softmax) | scale bất biến theo N; `mean` thì không |

⚠️ **Đổi model ⇒ PHẢI chạy lại `calibrate_threshold.py`.** `reid_threshold` là thuộc tính
của thang điểm model, không phải hằng số pipeline.

### Các chốt bảo vệ tự động (bổ sung 29/9)

| Cơ chế | Ở đâu | Chặn được gì |
|---|---|---|
| `resolve_pipeline(cfg, section)` | `pipeline_lock.py`, gọi trong `infer.py`, `evaluate_reid.py`, `calibrate_threshold.py`, `evaluate_reid_robustness.py`, `phan_rang/infer_realworld.py` | `num_frames` / `stride` / `backbone` / `bbox_padding` / `temporal_*` ghi lệch giữa các section |
| `assert_frozen(cfg)` | Đầu `train_reid.py` | Mở lại `n_frames_choices`, `pool='mean'`, `lam2 != 0`, thiếu `temporal_type`, `num_before/after_frames` < N cần dùng |
| `provenance(cfg)` | Ghi vào `calibrated_threshold.json` **và mọi checkpoint** | Không biết một checkpoint / ngưỡng thuộc protocol nào |
| `check_data_meta(cfg)` | Tự gọi trong `resolve_pipeline` + đầu `train_reid.py` | Config đổi `frame_stride` / `num_before/after_frames` / `bbox_padding` / `crop_size` mà **chưa sinh lại dữ liệu** |
| Chặn `data_pipeline.py` khi thiếu key | `data_pipeline.py` | Sinh lại dữ liệu ở bước 1 (argparse mặc định) trong khi train dùng bước 4 |

**`check_data_meta` cần `pipeline_meta.json`.** `data_pipeline.py` ghi file này vào
`<paths.data_dir>/` khi sinh dữ liệu:

```json
{ "frame_stride": 4, "num_before_frames": 16, "num_after_frames": 16,
  "bbox_padding": 0.2, "crop_size": 256 }
```

Bốn file `query/gallery_*.json` chỉ chứa **tên file**, không cho biết bước thời gian — nên
không có `pipeline_meta.json` thì không cách nào biết dữ liệu cũ hay mới. Thiếu file này thì
cảnh báo **im lặng bỏ qua** (tương thích ngược với dữ liệu sinh trước 29/9); có file thì lệch
là **báo động**.

### Ba việc duy nhất phải làm

1. **Sửa model** — trong package `model/` (xem §trên)
2. **Sửa config** — `train.temporal_type` + 4 đường dẫn. Đổi tham số trong `data_pipeline`
   thì **phải sinh lại dữ liệu**
3. **Chạy** `train_reid.py` → `calibrate_threshold.py` → `evaluate_reid.py` → `infer.py`

⚠️ Đổi model ⇒ **PHẢI** chạy lại `calibrate_threshold.py`: `reid_threshold` là thuộc tính
của thang điểm model, không phải hằng số pipeline.

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
