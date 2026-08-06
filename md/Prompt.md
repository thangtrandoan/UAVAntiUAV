# PROMPT CHO GEMINI — UAV Anti-UAV ReID System

> **Hướng dẫn sử dụng:** Copy toàn bộ nội dung bên dưới và paste vào Gemini. Prompt đã được chia thành 4 Phase riêng biệt. Bạn có thể gửi từng Phase một hoặc gửi cả cục tuỳ theo khả năng context của Gemini.

---

## PHASE 1: DATA PIPELINE — Trích xuất cặp T_before / T_after từ video UAV-Anti-UAV

### Bối cảnh
Tôi đang xây dựng hệ thống **Air-to-Air UAV Re-Identification (ReID)** — nhận diện lại drone mục tiêu sau khi nó biến mất và xuất hiện lại trên camera. Tôi có sẵn bộ dữ liệu `UAV-Anti-UAV` gồm 1820 video sequences (1400 Train, 420 Test).

### Cấu trúc dữ liệu đầu vào
```
/thang/UAV-Anti-UAV/
├── Train/
│   ├── UAV-Anti-UAV_Train_000001/
│   │   ├── UAV-Anti-UAV_Train_000001.mp4      # Video gốc
│   │   ├── UAV-Anti-UAV_Train_000001.jpg      # Ảnh đại diện (cover)
│   │   ├── groundtruth_rect.txt               # Mỗi dòng = "x,y,w,h" cho từng frame
│   │   ├── absent.txt                         # Mỗi dòng = 0 (có mặt) hoặc 1 (biến mất)
│   │   ├── attributes.txt                     # 15 dòng cờ nhị phân (thách thức môi trường)
│   │   └── language.txt                       # Mô tả bằng ngôn ngữ tự nhiên về drone
│   └── ... (đến 001400)
└── Test/
    └── ... (đến 000420)
```

### Yêu cầu
Viết file `data_pipeline.py` đặt tại `/thang/UAVAntiUAV/data_pipeline.py` thực hiện:

1. **Đọc và parse video:** Dùng OpenCV (`cv2.VideoCapture`) đọc từng file `.mp4`. Đọc `groundtruth_rect.txt` để lấy bounding box cho mỗi frame, đọc `absent.txt` để xác định frame nào drone biến mất.

2. **Xác định các đoạn xuất hiện/biến mất (Segments):**
   - Quét file `absent.txt` để tìm các **đoạn chuyển tiếp** (transition segments): Drone có mặt → biến mất → có mặt lại.
   - Với mỗi đoạn chuyển tiếp, xác định:
     - `T_before`: **N frame cuối cùng** trước khi drone biến mất (ví dụ: N=8 hoặc N=16 frame).
     - `T_after`: **M frame đầu tiên** sau khi drone xuất hiện lại (ví dụ: M=8 hoặc M=16 frame).
   - Nếu trong 1 video không có sự kiện biến mất nào (toàn bộ `absent.txt` là 0), thì **bỏ qua video đó** cho mục đích fine-tuning (hoặc tuỳ chọn: chia đôi video làm pseudo T_before/T_after).

3. **Crop và lưu ảnh:**
   - Với mỗi frame trong `T_before` và `T_after`, dùng bounding box từ `groundtruth_rect.txt` để crop vùng chứa drone.
   - Mở rộng bounding box thêm **20% padding** mỗi chiều (để không cắt sát drone quá).
   - Resize crop về kích thước chuẩn `256×256`.
   - Lưu ra thư mục output có cấu trúc:
     ```
     /thang/UAVAntiUAV/processed/
     ├── train/
     │   ├── seq_000001_event_0/
     │   │   ├── before/  # N ảnh crop của T_before
     │   │   │   ├── frame_0145.jpg
     │   │   │   └── ...
     │   │   └── after/   # M ảnh crop của T_after
     │   │       ├── frame_0203.jpg
     │   │       └── ...
     │   └── seq_000001_event_1/  # Nếu có nhiều lần biến mất trong 1 video
     └── test/
         └── ...
     ```

4. **Tạo file metadata:**
   - Lưu file `pairs_train.json` và `pairs_test.json` chứa danh sách tất cả các cặp (T_before, T_after) kèm thông tin:
     ```json
     {
       "sequence_id": "UAV-Anti-UAV_Train_000001",
       "event_index": 0,
       "identity_id": 0,
       "before_frames": ["frame_0145.jpg", "frame_0146.jpg", ...],
       "after_frames": ["frame_0203.jpg", "frame_0204.jpg", ...],
       "disappearance_duration_frames": 58,
       "language_description": "A green fixed-wing drone with yellow accents...",
       "attributes": [0, 0, 1, 0, 0, ...]
     }
     ```
   - **Quan trọng:** `identity_id` — mỗi sequence chỉ chứa 1 drone mục tiêu, nên tất cả các event trong cùng 1 sequence chia sẻ cùng 1 `identity_id`. Gán `identity_id` tăng dần từ 0 đến N-1 (N = số sequence có ít nhất 1 event).

5. **Thống kê báo cáo:** In ra console và lưu:
   - Tổng số sequence đã xử lý / bị bỏ qua (do không có sự kiện biến mất).
   - Tổng số cặp (T_before, T_after) được tạo.
   - Thống kê thời gian biến mất: trung bình, min, max (tính bằng số frame).
   - Phân phối theo attributes (bao nhiêu video có Fast Motion, Out-of-View, v.v.).

### Lưu ý kỹ thuật
- Dùng `argparse` cho các tham số: `--data-dir`, `--output-dir`, `--num-before-frames` (default=16), `--num-after-frames` (default=16), `--bbox-padding` (default=0.2), `--crop-size` (default=256), `--num-workers` (default=4).
- Dùng multiprocessing (`concurrent.futures.ProcessPoolExecutor`) để xử lý song song nhiều video.
- Xử lý edge case: frame đầu/cuối video, bounding box nằm ngoài biên ảnh, video bị lỗi.
- Dùng tqdm cho progress bar.

---

## PHASE 2: MODEL ARCHITECTURE — Mở rộng GASNet với Temporal Memory Engine (Mamba)

### Kiến trúc GASNet hiện tại (đã có sẵn)
Tôi đã có mô hình GASNet tại `/thang/gasnet_project/train.py` với kiến trúc sau:

```
GASNet (backbone=resnet50_ibn, use_gem=True)
├── Backbone: ResNet50-IBN (Instance-Batch Norm ở layers 1-3)
│   ├── stem: conv7×7 + BN + ReLU + MaxPool → [B, 64, 56, 56]
│   ├── layer1 → [B, 256, 56, 56]
│   ├── layer2 → [B, 512, 28, 28]
│   ├── layer3 → [B, 1024, 14, 14]
│   └── layer4 → [B, 2048, 7, 7]
├── RGA Attention (sau mỗi layer):
│   ├── ga1: RGABlock(ch=256,  spatial=56×56) — Spatial + Channel attention
│   ├── ga2: RGABlock(ch=512,  spatial=28×28)
│   ├── ga3: RGABlock(ch=1024, spatial=14×14)
│   └── ga4: RGABlock(ch=2048, spatial=7×7)
├── FS Branch (Feature Synthesis từ layer3):
│   ├── fs1: OSBlockFS(1024→512, 4 parallel Lite3×3 + ChannelGate)
│   └── fs2: OSBlockFS(512→512)
├── Pooling: GeMPool(p=3.0) — generalized mean pooling
├── BNNeck: bnneck_global(2048), bnneck_fs(512)
├── Classifiers: Linear(2048→C), Linear(512→C)
│
│ Output lúc eval: cat(bn_global, bn_fs) → [B, 2560]
│ Output lúc train: (global_feat[2048], fs_feat[512]), (logits_global, logits_fs)
```

**Checkpoint đã pre-train trên VRAI:** `/thang/gasnet_project/test/gasnet.best.pth`

### Yêu cầu
Viết file `model.py` đặt tại `/thang/UAVAntiUAV/model.py` thực hiện:

1. **Import và tái sử dụng GASNet backbone:**
   - Import class `GASNet` từ `/thang/gasnet_project/train.py`.
   - Tạo class `UAVReIDNet(nn.Module)` wrap lấy GASNet làm visual backbone.
   - Load pre-trained weights từ `gasnet.best.pth` vào backbone, **freeze toàn bộ backbone trong giai đoạn đầu** (chỉ train Mamba head).

2. **Thêm Temporal Memory Engine (Mamba/S6 Block):**
   - Cài đặt một module `TemporalMambaEncoder` nhận vào chuỗi N frame features `[B, N, D]` (D=2560 — embedding dim của GASNet eval).
   - Kiến trúc bên trong:
     ```
     Input: [B, N, 2560]
       ↓ Linear projection: [B, N, 2560] → [B, N, d_model] (d_model=512)
       ↓ Positional Encoding (learnable)
       ↓ MambaBlock × L layers (L=2 hoặc 4)
       ↓ Lấy hidden state cuối cùng: [B, d_model]
       ↓ MLP Head: Linear(d_model → 256) → BN → ReLU
       ↓ Output: Temporal Token [B, 256]
     ```
   - **MambaBlock:** Nếu thư viện `mamba_ssm` không cài được trên Jetson, thì **tự implement một Simplified S6 Block** (Selective State Space):
     ```python
     class SimpleS6Block(nn.Module):
         # Input: [B, L, D]
         # 1. Linear expansion: D → 2*D (split thành x_proj, z_gate)
         # 2. 1D Depthwise Conv (kernel=4) trên x_proj
         # 3. SSM discretization: A, B, C, Δ parameters
         #    - A: [D, state_dim] (log-space, learnable)
         #    - B, C: projected from x_proj via Linear
         #    - Δ (delta/dt): projected from x_proj via Linear → Softplus
         # 4. Selective Scan (sequential hoặc parallel scan)
         # 5. Output gating: y * SiLU(z_gate)
         # 6. Linear projection: 2*D → D
         # 7. Residual + LayerNorm
     ```
   - Nếu `mamba_ssm` có sẵn, dùng `from mamba_ssm import Mamba` trực tiếp.
   - **Tự động fallback:** Thử import `mamba_ssm`, nếu fail thì dùng `SimpleS6Block`.

3. **ReID Head:**
   - Class `ReIDHead(nn.Module)`:
     ```
     Input: visual_feat [B, 2560] + temporal_token [B, 256]
       ↓ Concatenate: [B, 2816]
       ↓ BNNeck(2816)
       ↓ Classifier: Linear(2816 → num_identities) — chỉ dùng lúc train
       ↓ Output lúc eval: bn_feat [B, 2816]
     ```

4. **Full forward flow:**
   ```python
   class UAVReIDNet(nn.Module):
       def forward(self, before_clips, after_clips=None):
           """
           before_clips: [B, N, 3, 224, 224] — N frames trước khi biến mất
           after_clips:  [B, M, 3, 224, 224] — M frames sau khi xuất hiện lại (chỉ dùng lúc train)

           Lúc train:
             - Trích xuất features cho từng frame qua GASNet backbone → [B, N, 2560]
             - Đẩy qua TemporalMambaEncoder → temporal_token_before [B, 256]
             - Tương tự cho after_clips → temporal_token_after [B, 256]
             - ReID Head cho cả before và after
             - Return features + logits cho loss computation

           Lúc inference (ReID verification):
             - Chỉ cần before_clips (từ memory đã lưu trước đó)
             - Hoặc chỉ cần after_clips (candidate mới xuất hiện)
             - Return embedding vector để tính cosine similarity
           """
   ```

5. **Utility function `freeze_backbone()` và `unfreeze_backbone()`** để chuyển đổi giữa giai đoạn freeze và fine-tune.

### Lưu ý kỹ thuật
- Đảm bảo tương thích với AMP (Automatic Mixed Precision) trên Jetson AGX Orin.
- Đảm bảo `torch.channels_last` memory format cho phần CNN backbone.
- Các module mới phải hỗ trợ `torch.compile()` (tránh dynamic control flow trong forward).
- Comment rõ ràng bằng tiếng Việt hoặc tiếng Anh.

---

## PHASE 3: LOSS FUNCTIONS & TRAINING LOOP

### Yêu cầu
Viết file `train_reid.py` đặt tại `/thang/UAVAntiUAV/train_reid.py` thực hiện:

1. **Loss Functions:**
   ```python
   # L_total = L_ID + λ1 * L_HardTriplet + λ2 * L_TemporalConsistency + λ3 * L_Center

   class LabelSmoothCrossEntropy(nn.Module):
       # Cross-entropy với label smoothing (epsilon=0.1)

   class HardTripletLoss(nn.Module):
       # Online Hard Mining Triplet Loss
       # Anchor = temporal_token_before
       # Positive = temporal_token_after (cùng identity)
       # Negative = temporal_token của identity khác trong batch
       # margin = 0.3, sử dụng L2-normalized features

   class CenterLoss(nn.Module):
       # Học center cho mỗi identity
       # Pull features về phía center: ||f_i - c_yi||^2
       # Centers được update bằng moving average (lr_center = 0.5)

   class TemporalConsistencyLoss(nn.Module):
       # Cosine similarity loss giữa hidden states liên tiếp trong Mamba
       # Đảm bảo latent state không nhảy đột ngột giữa các frame
       # L_tc = 1 - mean(cos_sim(h_t, h_{t+1})) cho t trong [1, N-1]
   ```

2. **Dataset & DataLoader:**
   - `UAVReIDDataset(torch.utils.data.Dataset)`:
     - Đọc `pairs_train.json` từ Phase 1.
     - Mỗi sample trả về: `(before_clip[N,3,224,224], after_clip[M,3,224,224], identity_id)`.
     - Augmentations: Random Horizontal Flip, Color Jitter, Random Erasing (simulating occlusion), Motion Blur (Gaussian kernel).
     - Resize 256→224 với Random Crop lúc train, Center Crop lúc val.
   - `ReIDBatchSampler`: Đảm bảo mỗi batch có **P identities × K instances** (P=8, K=4 — mỗi identity lấy 4 cặp/clips) để Hard Triplet Mining hoạt động hiệu quả.

3. **Training Loop:**
   - **Stage 1 (Freeze backbone, 30 epochs):**
     - Chỉ train: `TemporalMambaEncoder` + `ReIDHead`
     - LR: 3.5e-4, Cosine Annealing, Warmup 5 epochs
     - λ1=1.0, λ2=0.5, λ3=0.005
   - **Stage 2 (Unfreeze backbone, 20 epochs):**
     - Train toàn bộ model
     - LR backbone: 1e-5 (10× thấp hơn head)
     - LR head: 1e-4
     - λ1=1.0, λ2=0.3, λ3=0.01
   - Optimizer: AdamW (weight_decay=5e-4)
   - AMP (fp16) cho toàn bộ training
   - Gradient clipping: max_norm=1.0
   - Save checkpoint mỗi epoch + save best model theo mAP trên validation split

4. **Logging:**
   - In ra mỗi N step: loss tổng, từng loss thành phần, learning rate, thời gian/step
   - Cuối mỗi epoch: chạy evaluation nhanh trên validation set (lấy 20% từ train) → báo Rank-1, mAP

### Lưu ý kỹ thuật
- Tối ưu cho Jetson AGX Orin 64GB (giới hạn GPU memory ~32GB khả dụng).
- `--gpu-jetson` flag: tự động giảm batch size, tăng gradient accumulation.
- Dùng `torch.cuda.amp.GradScaler` cho mixed precision training.
- Checkpoint phải lưu: model state_dict, optimizer state_dict, epoch, best_mAP, scaler state.

---

## PHASE 4: EVALUATION MODULE

### Yêu cầu
Viết file `evaluate_reid.py` đặt tại `/thang/UAVAntiUAV/evaluate_reid.py` thực hiện:

1. **Offline ReID Evaluation (Static Protocol):**
   - **Query Set:** Lấy clip T_after của mỗi event làm query.
   - **Gallery Set:** Lấy clip T_before của tất cả events (bao gồm cả positive match và distractors).
   - Trích xuất embedding cho toàn bộ query và gallery qua model.
   - Tính Cosine Similarity matrix.
   - Tính metrics:
     - **Rank-1 Accuracy (%)**
     - **Rank-5 Accuracy (%)**
     - **mAP (mean Average Precision)**
     - **mINP (mean Inverse Negative Penalty)** — đánh giá chất lượng hard-sample retrieval
   - Phân tích theo attributes: Tính riêng Rank-1 và mAP cho từng attribute (FM, OV, SV, LR...) để biết model yếu ở đâu.

2. **Online Sequential Evaluation (Stream Protocol):**
   - Giả lập luồng video thời gian thực:
     - Đọc video từ frame 0, chạy tracker (giả lập bằng groundtruth bbox).
     - Khi drone biến mất (absent=1), freeze temporal token.
     - Khi drone xuất hiện lại, chạy ReID verification:
       - Crop vùng bbox mới, trích xuất feature, so sánh với temporal token đã freeze.
       - Đo **Re-acquisition Latency**: Bao nhiêu frame từ lúc xuất hiện lại đến lúc model confirm match (score > threshold).
       - Đo **False Alarm Rate**: Tỉ lệ confirm sai.
   - Tính metrics trung bình trên toàn bộ test set.

3. **Visualization:**
   - Lưu ảnh so sánh (contact sheet) cho các ca lỗi: Query (T_after) bên trái, Top-5 gallery (T_before) bên phải.
   - Dùng viền xanh cho match đúng, viền đỏ cho match sai.

4. **Report:**
   - In bảng tổng hợp kết quả ra console.
   - Lưu file `evaluation_report.json` chứa tất cả metrics.

### Lệnh chạy
```bash
python evaluate_reid.py \
    --model-path /thang/UAVAntiUAV/checkpoints/best_model.pth \
    --data-dir /thang/UAVAntiUAV/processed \
    --pairs-json /thang/UAVAntiUAV/processed/pairs_test.json \
    --output-dir /thang/UAVAntiUAV/eval_results \
    --gpu-jetson \
    --batch-size 32
```

---

## TÓM TẮT CÁC FILE CẦN TẠO

| # | File | Mô tả |
|---|---|---|
| 1 | `/thang/UAVAntiUAV/data_pipeline.py` | Trích xuất cặp T_before/T_after từ video |
| 2 | `/thang/UAVAntiUAV/model.py` | UAVReIDNet = GASNet backbone + Mamba Memory + ReID Head |
| 3 | `/thang/UAVAntiUAV/train_reid.py` | Training loop với 4 loss functions, 2-stage training |
| 4 | `/thang/UAVAntiUAV/evaluate_reid.py` | Offline + Online evaluation với mAP/Rank-1/mINP |

## THÔNG TIN MÔI TRƯỜNG
- **Hardware:** Jetson AGX Orin 64GB
- **Python:** 3.10+
- **PyTorch:** 2.x
- **CUDA:** có sẵn
- **Backbone checkpoint:** `/thang/gasnet_project/test/gasnet.best.pth`
- **Dataset gốc:** `/thang/UAV-Anti-UAV/`
- **Thư mục project:** `/thang/UAVAntiUAV/`
- **GASNet source code:** `/thang/gasnet_project/train.py` (chứa class GASNet, ResNetIBN, RGABlock, OSBlockFS, BNNeck, GeMPool, AttentionLocalBranch)
