# 🔍 Phân tích: Tại sao GASNet Mamba v1 tệ hơn GASNet gốc?

## 📊 So sánh kết quả (CÓ SỐ LIỆU THỰC TẾ)

| Metric | GASNet Baseline | GASNet Mamba v1 | Chênh lệch |
|--------|:------:|:------:|:------:|
| **Rank-1** | **50.29%** | 41.47% | **−8.82%** 📉 |
| **Rank-5** | **61.59%** | 53.23% | **−8.36%** 📉 |
| **mAP** | **10.95%** | 8.85% | **−2.10%** 📉 |
| **mINP** | **2.53%** | 2.32% | **−0.21%** 📉 |
| **False Alarm Rate** | **49.71%** | 58.53% | **+8.82%** 📉 |
| **Re-acq Latency** | **2.42 frames** | 2.55 frames | +0.13 |

> [!CAUTION]
> Mamba v1 **tệ hơn baseline trên MỌI metric**. Rank-1 giảm gần 9 điểm phần trăm — rất đáng kể. Thêm Temporal Mamba Encoder đang **gây hại** thay vì giúp ích.

---

## 🔴 5 Nguyên nhân gốc (Root Causes)

### 1. 🚨 **Không có `mamba_ssm` — dùng SimpleS6Block fallback** (CRITICAL)

Cả training log LẪN eval log đều show:
```
Warning: Không tìm thấy mamba_ssm. Sử dụng Simplified S6 Block (fallback).
```

`SimpleS6Block` là implementation fallback với **sequential Python for-loop** (line 97 trong `model.py`):
```python
for i in range(L):
    dt_i = dt[:, i, :].unsqueeze(-1)
    dA = torch.exp(dt_i * A)
    ...
```
- Gradient flow qua sequential loop rất kém
- Không tương đương chính xác với CUDA-optimized Mamba
- Kết quả: Temporal encoder **không học được temporal patterns hữu ích**

### 2. 🚨 **Chỉ lấy last hidden state — bỏ phí temporal context**

```python
# model.py line 163-164
x = x[:, -1, :]  # Chỉ lấy frame CUỐI CÙNG
```

Với chỉ **2 Mamba layers** (shallow model), hidden state ở vị trí cuối không đủ capture thông tin từ toàn bộ 12-16 frames. Đặc biệt với SSM 1 chiều (trái→phải), thông tin từ frames đầu bị suy giảm rất nhiều khi đến frame cuối.

### 3. 🔴 **Temporal token quá nhỏ — bị visual feature lấn át**

```
Feature cuối = [visual_feat(2560) | temporal_token(256)] = 2816-dim
                 ↑ 91%                ↑ chỉ 9%
```

Nếu Mamba encoder chưa học được gì hữu ích (do issue #1 và #2), 256-dim temporal token chỉ thêm **noise vào 9% feature** → kéo cosine similarity matching xuống → Rank-1 giảm.

### 4. 🟡 **Best model chọn theo training loss — không có validation**

```python
# train_reid.py line 452-455
if avg_loss < best_loss:
    best_loss = avg_loss
    torch.save(checkpoint_data, ...)
```

Training loss giảm liên tục (~296 → ~15) nhưng **không có validation**. Model overfits training data, generalization kém.

### 5. 🟡 **Feature mismatch giữa backbone_only vs full model khi eval**

- `backbone_only=True`: return `v_feat` (2560-dim, **raw**, chưa normalize)
- `backbone_only=False`: return `bn_feat` (2816-dim, **qua BNNeck**)

Hai feature spaces khác nhau hoàn toàn → so sánh cùng eval script nhưng representation khác nhau. Baseline dùng 2560-dim pure visual features, Mamba v1 dùng 2816-dim mixed features.

---

## 🟢 Khuyến nghị sửa (Theo thứ tự ưu tiên)

### Fix 1: ⭐ Cài `mamba_ssm` — bỏ fallback
```bash
pip install mamba-ssm
# Hoặc trên Jetson: build from source với CUDA support
```
Đây là fix **quan trọng nhất**. `SimpleS6Block` sequential loop không tương đương và gradient flow kém hơn rất nhiều so với CUDA-optimized selective scan.

### Fix 2: ⭐ Đổi aggregation strategy — mean pooling thay vì last token
```python
# Thay vì:
x = x[:, -1, :]  # Chỉ lấy frame cuối

# Đổi thành:
x = x.mean(dim=1)  # Mean pooling qua toàn bộ sequence
```
**Thay đổi 1 dòng, impact lớn.** Mean pooling capture thông tin từ tất cả frames thay vì chỉ frame cuối.

### Fix 3: Tăng dimension temporal token
```python
# Tăng d_out từ 256 lên 512 hoặc 1024
self.temporal_encoder = TemporalMambaEncoder(d_in=2560, d_model=512, d_out=512)
# → ReIDHead in_dim = 2560 + 512 = 3072
```
Giúp temporal information có weight lớn hơn trong feature cuối.

### Fix 4: Thêm validation vào training loop
```python
# Sau mỗi N epoch, eval trên val set
if epoch % 5 == 0:
    val_rank1 = evaluate(model, val_loader)
    if val_rank1 > best_val_rank1:
        save_best_model()
```

### Fix 5: Thêm bidirectional scan
SSM mặc định chỉ scan 1 chiều. Video ReID cần aggregate thông tin cả 2 chiều thời gian.

### Fix 6: Normalize features khi eval backbone_only
```python
if backbone_only:
    return F.normalize(v_feat, p=2, dim=1)  # Thêm L2 normalize
```

---

## 🎯 Tóm tắt

| Nguyên nhân | Severity | Quick Fix? |
|---|---|---|
| SimpleS6Block fallback | 🚨 Critical | Cài mamba_ssm |
| Last hidden state only | 🚨 Critical | 1 dòng: `.mean(dim=1)` |
| Temporal token quá nhỏ (9%) | 🔴 High | Tăng d_out |
| Không có validation | 🟡 Medium | Thêm val loop |
| Feature mismatch khi eval | 🟡 Medium | Normalize |

> [!TIP]
> **Quick win**: Fix 1 + Fix 2 có thể cải thiện đáng kể chỉ với thay đổi nhỏ. Nếu không cài được `mamba_ssm` trên Jetson, tối thiểu nên áp dụng Fix 2 (mean pooling) + Fix 3 (tăng d_out) trước. 

**Note**: t hiện tại không cài được mamba_ssm, tạm bỏ qua fix 1
