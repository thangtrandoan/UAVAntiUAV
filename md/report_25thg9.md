# BÁO CÁO SO SÁNH HAI MÔ HÌNH

**Ngày báo cáo:** 25/9/2026

| | Mô hình |
|---|---|
| **M1** | Mô hình **tham chiếu** (mamba, `TemporalConsistencyLoss` **bản cũ** `1 − cos`) |
| **M2** | Mô hình **mới** (attention, `TemporalConsistencyLoss` **bản mới** dạng xếp hạng) |

Cả hai đều dùng backbone `dinov3_convnext`, `num_frames = 12`, không gian `fused` 1472-d.

---

## 0. Tóm tắt

1. Ở không gian triển khai `fused`, M2 thấp hơn 1.83% so với M1: Rank-1 76.32% so với 78.15%, mAP 37.54% so với 35.59% (+1.95), TAR@FAR 0.1% 7.66% so với 8.45% (−0.79).
2. Ở cả hai mô hình, fusion làm tăng khả năng xếp hạng (Rank-1, mAP) nhưng làm giảm điểm làm việc theo ngưỡng tuyệt đối (TAR@FAR 0.1% đều giảm)
3. Cả hai mô hình chạm ngưỡng ~52% TAR ở FAR 10% dù kiến trúc khác nhau ⇒ nút thắt nằm ở **chất lượng cặp/cửa sổ**, không nằm ở kiến trúc.
4. Ngưỡng calibrate khác nhau giữa 2 mô hình (0.633 so với 0.852).
5. Kiểm thử **online** với `num_frames` 16 / 12 / 8 đều cho **HARD LOCK 0%**; thủ phạm là **BN nén cosine từ ~0.85 xuống ~0.15**, **không** phải `num_frames` (mục 1.5).

---

## 1. Kết quả

> 📌 **Hai tập đánh giá khác nhau trong báo cáo này — không phải mâu thuẫn:**
>
> - **eval-split** (dùng ở mục 1.1–1.4): `calibrate_threshold.py` **chia** 340 sequence của
>   `query_test.json` thành **136 cal + 204 eval** (`cal_ratio 0.4`, `seed 42`). Chia theo
>   **`sequence_id`** (`split_sequences`, dòng 524) nên **mỗi sequence nằm hoàn toàn ở một split**,
>   không lẫn danh tính. Ngưỡng tìm trên **cal**, số liệu TAR/FAR **báo cáo** trên **eval** ⇒
>   **không bị thiên lệch** do tự chọn ngưỡng trên chính tập mình báo cáo.
> - **toàn bộ test** (dùng ở bảng ngưỡng thủ công mục 1.4): `evaluate_reid.py` chạy **cả 340
>   sequence**, không chia.
>
> Hệ quả: **Rank-1 khác nhau** — `fused` **76.32%** (gallery 204 seq) so với **73.46%** (gallery 340
> seq). Gallery càng lớn thì càng nhiều impostor ⇒ Rank-1 **càng khó**, nên `73.46% < 76.32%` là
> **đúng chiều**, không phải kết quả mâu thuẫn. Con số **73.46%** khớp chính xác log train (Stage 2,
> epoch 25, `N=12`) vì khâu validation dùng `evaluate_reid.py`.
>
> ⇒ Khi so M1 với M2 phải so **cùng một tập**: mục 1.1–1.4 dùng **eval-split** cho **cả hai**.

### 1.1. Ablation 4 không gian — cùng eval-split 1.699 cặp

| Không gian | dim | M1 Rank-1 | M2 Rank-1 | Δ | M1 mAP | M2 mAP | Δ | M1 TAR@0.1% | M2 TAR@0.1% | Δ |
|---|---|---|---|---|---|---|---|---|---|---|
| `visual` | 960 | 76.38% | 72.46% | **−3.92** | 31.49% | 30.38% | −1.11 | 9.94% | 8.56% | −1.38 |
| `temporal` | 512 | 65.24% | **44.00%** | **−21.24** | 26.90% | 20.10% | −6.80 | 5.29% | 3.01% | −2.28 |
| `pre_bn` | 1472 | 67.32% | **50.80%** | **−16.52** | 27.81% | 21.48% | −6.33 | 5.79% | 3.86% | −1.93 |
| `fused` | 1472 | **78.15%** | 76.32% | −1.83 | 35.59% | **37.54%** | **+1.95** | 8.45% | 7.66% | −0.79 |

**Nhận xét:**

- Ở `fused`, không gian thực sự đem đi triển khai M2 **gần như ngang** M1: Rank-1 −1.83 nhưng
  mAP **+1.95**. M2 xếp hạng **tốt hơn ở phần sâu** (mAP) nhưng kém hơn ở đỉnh (Rank-1).
- `temporal` only của M2 chỉ **44.00%** so với **65.24%** của M1.
- `pre_bn` của M2 sụp **−16.52**. Nguyên nhân: `pre_bn = cat(visual, temporal)` là vector **thô**,
  không có BatchNorm; M2 nhân token temporal với **`√N`** (M1 dùng `mean` trần) ⇒ tại `N=12` hệ số là
  **3.46**, làm nửa temporal **át** nửa visual khi tính cosine. Và vì ablation train với
  `N ∈ {8,12,16}` nên `√N` **thay đổi theo từng sample** (2.83 / 3.46 / 4.00) ⇒ BatchNorm cũng không
  hấp thụ hết được.

### 1.2. Đóng góp của fusion: Δ(`fused` − `visual`)

| Chỉ số | M1 | M2 |
|---|---|---|
| Rank-1 | +1.77 | +3.86 |
| mAP | +4.10 | +7.16 |
| **TAR@FAR 0.1%** | **−1.49** | **−0.90** |

Ở **cả hai mô hình độc lập**, nhánh temporal:
- **giúp** xếp hạng (Rank-1 và mAP đều tăng),
- **làm hại** điểm làm việc theo ngưỡng tuyệt đối (TAR@FAR 0.1% **đều giảm**).

### 1.3. Đường cong TAR@FAR (eval-split, calibrate trên cal-split)

**M1**:

| Không gian | FAR 0.1% | FAR 1% | FAR 5% | FAR 10% |
|---|---|---|---|---|
| `fused` | 8.5% | **23.2%** | **41.4%** | **51.9%** |
| `visual` | **9.94%** | **23.7%** | — | — |
| `pre_bn` | 5.8% | 15.9% | 31.0% | 41.4% |
| `temporal` | 5.29% | 14.7% | — | — |

**M2**:

| Không gian | FAR 0.1% | FAR 1% | FAR 5% | FAR 10% |
|---|---|---|---|---|
| `fused` | 7.7% | 21.5% | 40.9% | **52.1%** |
| `visual` | 8.6% | 21.5% | 38.4% | 49.1% |
| `pre_bn` | 3.9% | 13.3% | 29.8% | 41.4% |
| `temporal` | 3.0% | 10.5% | 25.3% | 36.7% |

**So trực tiếp trên `fused`** — điểm làm việc duy nhất có đủ số cho cả hai:

| Mốc FAR | M1 | M2 | Δ |
|---|---|---|---|
| 0.1% | 8.5% | 7.7% | −0.8 |
| 1% | **23.2%** | 21.5% | **−1.7** |
| 5% | 41.4% | 40.9% | −0.5 |
| 10% | 51.9% | **52.1%** | **+0.2** |

**Nhận xét:**

- **Hai đường cong gần như chồng khít.** Chênh lớn nhất là **1.7 điểm ở FAR 1%**; ở FAR 5% và 10% thì
  chênh dưới 0.5 điểm và **M2 nhỉnh hơn ở FAR 10%**. Đây là bằng chứng mạnh rằng **đặc tính làm việc
  theo ngưỡng của hai mô hình gần như giống nhau**, nhất quán với mAP 35.59% so với 37.54%.
- **Cả hai đều bão hoà ở ~52% ngay khi FAR đã rất dễ dãi (10%)** — M1 51.9%, M2 52.1%. Nếu chỉ còn
  vấn đề ngưỡng thì FAR 10% phải cho TAR gần mức Rank-1 (~78%); thực tế chỉ ~52% ⇒ **chất lượng cặp
  genuine mới là nút thắt**, và nút thắt đó **giống nhau ở cả hai mô hình**.
- Đường cong cũng cho thấy vì sao **không nên đọc mỗi mốc FAR 0.1%**: từ 0.1% → 1% TAR tăng **~2.7×**
  ở cả hai mô hình. Kết luận *"M1 hơn M2"* chỉ đúng ở mốc 0.1% và 1%; ở mốc 10% thì **ngược lại**.

### 1.4. Ngưỡng và điểm làm việc

| Không gian | t\* (FAR 0.1%) M1 | t\* (FAR 0.1%) M2 | Nhận xét |
|---|---|---|---|
| `fused` | **0.852396** | **0.633** | **lệch 0.22** — hai thang đo rất khác |
| `pre_bn` | 0.971264 | 0.956 | gần nhau |
| `visual` | 0.995573 | 0.996 | **gần như trùng** |
| `temporal` | 0.962562 | 0.868 | lệch 0.09 |

**Tại ngưỡng triển khai `0.75` — bảng đầy đủ cả 4 không gian:**

| Không gian | M1 FAR@0.75 | M1 TAR@0.75 | M2 FAR@0.75 | M2 TAR@0.75 |
|---|---|---|---|---|
| `fused` | 6.14% | **38.86%** | **0.0154%** | **3.30%** |
| `pre_bn` | 99.998% | 100.00% | 97.892% | 99.58% |
| `visual` | **100.000%** | 100.00% | 100.000% | 100.00% |
| `temporal` | 99.749% | 99.94% | 2.671% | 18.00% |

**Nhận xét:**

- **Hai mô hình lệch nhau ở đúng chỗ quan trọng:** tại 0.75, M1 cho TAR **38.86%** với FAR 6.14%,
  còn M2 chỉ cho TAR **3.30%** với FAR 0.0154%. Nhưng đây **không phải** M2 kém hơn — 0.75 nằm **dưới**
  t\* của M1 (0.852) nên lỏng, và **trên** t\* của M2 (0.633) nên chặt. So ở **cùng FAR** (bảng mục 1.3)
  thì hai mô hình gần như bằng nhau.
- Hệ quả thực hành: đem M2 vào pipeline mà **giữ `reid_threshold = 0.75`** thì tỉ lệ lock sẽ **thấp hơn
  cả mức ≤ 6.08%** hiện tại. Ngưỡng calibrate cho M2 là **~0.63** — **nhưng xem mục 1.5**: đo online
  cho thấy ngay cả 0.63 cũng **chưa đủ**.

Bảng đầy đủ theo ngưỡng thủ công (M2, toàn bộ 340 sequence):

| Ngưỡng | Tập | FAR thực | TAR |
|---|---|---|---|
| 0.633 (calibrate) | eval-split | 0.1445% | 7.66% |
| 0.633 (calibrate) | toàn bộ test | 0.1272% | 9.13% |
| 0.649 (in-sample) | toàn bộ test | 0.1000% | 8.29% |
| **0.700 (thủ công)** | toàn bộ test | 0.0396% | 5.85% |
| 0.750 (đang triển khai) | eval-split | 0.0154% | 3.30% |

### 1.5. Kiểm tra infer với tham số `num_frames` khác nhau với cùng 1 video với M1


| `num_frames` | Cửa sổ re-acquire | HARD LOCK | PRE-BN raw | POST-BN `fused` | BN delta | `temporal` (khoảng) |
|---|---|---|---|---|---|---|
| 16 | 26 | **0/26 = 0.0%** | 0.831 | 0.155 | +0.676 | 0.378 – 0.581 |
| **12** ← đúng lúc train | 36 | **0/36 = 0.0%** | 0.825 | 0.112 | +0.713 | 0.270 – 0.630 |
| 8 | 46 | **0/46 = 0.0%** | **0.900** | 0.174 | +0.726 | **0.691 – 0.920** |

Khoảng giá trị `fused` thực đo: `N=16` → 0.122–0.249 · `N=12` → 0.072–0.244 · `N=8` → 0.121–0.298

**Nhận xét:**

2. **`fused` online nằm ở thang hoàn toàn khác offline.** Giá trị **lớn nhất** đo được là **0.298**
   (`N=8`), trong khi ngưỡng đang triển khai là **0.75** và ngưỡng calibrate offline là **0.633**.
   ⇒ **Kể cả hạ ngưỡng về 0.633 thì HARD LOCK vẫn 0%** — đổi ngưỡng một mình là **không đủ**. Mục 1.4
   dự đoán TAR 3.30% tại 0.75; online xác nhận còn thấp hơn: **0.00%**.
3. **BN là mắt xích chặn, không phải temporal.** Ở cả ba lần chạy `PRE-BN raw` = **0.825 – 0.900**
   ⇒ **vượt 0.75 ở 100% số cửa sổ**; nhưng sau BN chỉ còn **0.112 – 0.174** ⇒ **0%**. BN nén mất
   **0.68 – 0.73** cosine. Trong khi đó `visual_shot` / `visual_plain` vẫn **0.98 – 0.99** ⇒ nhánh visual
   nói "đúng vật" rất chắc; **BN + head mới là chỗ phá quyết định**.
4. **`N = 8` cho tín hiệu temporal tốt hơn hẳn** (0.691–0.920 so với 0.270–0.630 ở `N=12`) nhưng `fused`
   chỉ nhích 0.112 → 0.174 ⇒ **tín hiệu tốt hơn vẫn bị BN nén mất**. Bằng chứng độc lập rằng nút thắt
   nằm **sau** encoder.
6. **Tốc độ gần như không đổi theo `N`**: 54.85 / 54.96 / 53.79 FPS; CNN 36.8 / 36.6 / 37.4 ms;
   temporal+head 3.17 / 3.15 / 2.79 ms ⇒ chọn `N` là đánh đổi **độ chính xác**, gần như **không** tốn
   tốc độ.
7. **Số cửa sổ tăng khi `N` giảm** (26 → 36 → 46) vì cửa sổ ngắn hơn lấp được nhiều lần hơn trong cùng
   khoảng vắng ⇒ `N=8` có **nhiều cơ hội lock nhất** mà vẫn **0%**, càng khẳng định vấn đề **không phải**
   số cơ hội.


## 2. Nhận xét

### 2.1. So sánh 2 mô hình

Ở không gian `fused`, **M2 không tệ hơn M1 một cách có ý nghĩa**: Rank-1 −1.83, mAP **+1.95**,
TAR@0.1% −0.79. Với mAP cao hơn, có thể nói **thứ hạng tổng thể của M2 tốt hơn**, trong khi **đỉnh
xếp hạng (Rank-1) hơi kém hơn**. Nếu tiêu chí là mAP hoặc chất lượng xếp hạng tổng thể thì M2 là một
bước tiến nhẹ; nếu tiêu chí là Rank-1 thì là một bước lùi nhẹ.

Và **hai quy luật tái lập được** (mục 1.2, 1.3) là kết quả có giá trị độc lập với việc mô hình nào
hơn:
- fusion đánh đổi **xếp hạng ↔ cổng quyết định**;
- trần TAR ở FAR 10% là **~52%** ở cả hai mô hình.

Khi kiểm tra khả năng suy luận với video thực với tham số num frames khác nhau, có thể thấy được rõ vấn đề ở batchnorm, hoặc do attention kết hợp không tốt với batchnorm.
