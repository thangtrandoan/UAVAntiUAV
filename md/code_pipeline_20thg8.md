# Hướng dẫn Code lại `infer_realworld.py` theo Pipeline mới
> **Ngày:** 20/08/2026  
> **Pipeline tham chiếu:** [`md/pipeline.md`](file:///home/namm/thang/UAVAntiUAV/md/pipeline.md)  
> **File cần viết lại:** [`phan_rang/infer_realworld.py`](file:///home/namm/thang/UAVAntiUAV/phan_rang/infer_realworld.py)

---

## Mục lục
1. [Phân tích khác biệt Code cũ vs Pipeline mới](#1-phân-tích-khác-biệt-code-cũ-vs-pipeline-mới)
2. [Kiến trúc Class mới](#2-kiến-trúc-class-mới)
3. [Hướng dẫn code từng phần](#3-hướng-dẫn-code-từng-phần)
4. [Thay đổi Config](#4-thay-đổi-config)
5. [Checklist hoàn thành](#5-checklist-hoàn-thành)

---

## 1. Phân tích khác biệt Code cũ vs Pipeline mới

| Khía cạnh | Code cũ (`infer_realworld.py` hiện tại) | Pipeline mới (`pipeline.md`) | Hành động |
|:---|:---|:---|:---|
| **Cửa sổ trượt & Stride** | Thu thập 16 frame **liên tiếp** (mỗi frame đều chạy GASNet). | Thu thập k frame với **stride = 2 hoặc 3** (chỉ trích xuất cách quãng), lưu vào cửa sổ trượt. | **Thêm mới**: Cơ chế stride sampling + sliding window buffer. |
| **Trọng số độ nét (Sharpness)** | Không có. Dùng `mean()` đơn thuần. | Tính trung bình **có trọng số** dựa trên **điểm độ nét** (sharpness score) của từng frame. | **Thêm mới**: Hàm tính sharpness + weighted average. |
| **Memory Bank lưu gì** | Chỉ lưu vector 3072-dim (đã qua Mamba) vào cả Anchor và Recent Bank. | Lưu **2 loại vector** cho mỗi entry: vector 2560-dim (visual, dùng cho Lọc Thô) VÀ vector 3072-dim (visual+temporal, dùng cho Lọc Tinh). | **Viết lại**: Cấu trúc Memory Bank phải lưu cặp `(visual_2560, fused_3072)`. |
| **Cập nhật Memory Bank** | Cập nhật mỗi 60 frame trong TRACKING. | Cập nhật mỗi **t giây** (configurable) HOẶC ngay trước khi mất dấu. | **Sửa lại**: Đổi trigger cập nhật từ frame count sang time-based. |
| **Lọc Thô (Coarse ReID)** | Dùng `expand_coarse_to_fine()` nhân bản 1 frame thành 16 frame ảo → chạy qua Mamba → so sánh vector 3072-dim. Tốn ~58ms (50ms GASNet + 8.5ms Mamba). | Chỉ dùng vector 2560-dim từ GASNet, so sánh trực tiếp cosine với vector 2560-dim trong Memory Bank. **Không chạy Mamba** ở bước này. Chỉ tốn ~50ms. | **Viết lại**: Bỏ `expand_coarse_to_fine()`. Lọc Thô chỉ dùng GASNet features. |
| **Spatial Penalty** | Có module riêng, trừ điểm dựa trên khoảng cách Euclidean. | **Không còn** trong pipeline mới (đã bỏ). | **Xoá**: Bỏ toàn bộ logic spatial penalty. |
| **Blacklist** | Khi Fine ReID fail → blacklist track ID vĩnh viễn. | **Không còn** trong pipeline mới. Khi fail → quay về LOST tìm lại. | **Xoá**: Bỏ cơ chế blacklist. |
| **Anti-Hijack (T3)** | Chạy verification mỗi 60 frame, check `anchor_bank` similarity < 0.40 → Break Lock. | Chỉ kiểm tra trong **x lần cập nhật recent bank đầu** sau Hard Lock. So sánh với **toàn bộ memory bank** (anchor + recent), dùng `hijack_threshold`. | **Viết lại**: Thêm biến đếm `hijack_check_count`, đổi logic kiểm tra. |
| **State Machine** | 3 trạng thái: `INITIAL_TRACKING`, `TRACKING`, `LOST`. | 4 giai đoạn rõ ràng: `T0_INIT`, `T1_LOST`, `T2_SEARCH`, `T3_VERIFIED`. | **Viết lại**: Đổi tên và logic chuyển trạng thái. |

---

## 2. Kiến trúc Class mới

Thay vì code toàn bộ logic trong hàm `main()` như hiện tại, tách ra thành các class rõ ràng:

```
infer_realworld.py
├── class SlidingWindowBuffer     # Cửa sổ trượt với stride  
├── class TwoTierMemoryBank       # Anchor Bank + Recent Bank (lưu cặp 2560+3072)
├── class ReIDPipeline            # State machine (T0 → T1 → T2 → T3)
│   ├── _extract_visual_feat()    # GASNet → 2560-dim
│   ├── _extract_temporal_feat()  # Mamba → 512-dim
│   ├── _compute_fused_vector()   # 2560 + 512 → 3072-dim (có trọng số sharpness)
│   ├── _coarse_reid()            # Lọc Thô: cosine(2560, bank_2560)
│   ├── _fine_reid()              # Lọc Tinh: cosine(3072, bank_3072)
│   └── _anti_hijack_check()      # Kiểm tra bám nhầm
└── def main()                    # Vòng lặp video chính
```

### 2.1 Class `SlidingWindowBuffer`

```python
class SlidingWindowBuffer:
    """Cửa sổ trượt quản lý k frame features với stride sampling."""
    
    def __init__(self, window_size: int = 16, stride: int = 2):
        self.window_size = window_size  # k = 16
        self.stride = stride            # Chỉ lấy mỗi frame thứ stride
        self.features = []              # List[Tensor] - mỗi tensor là 2560-dim
        self.sharpness_scores = []      # List[float] - điểm độ nét tương ứng
        self._frame_counter = 0         # Đếm frame thực tế (trước stride)
    
    def should_extract(self) -> bool:
        """Kiểm tra frame hiện tại có nên chạy GASNet không (dựa trên stride)."""
        result = (self._frame_counter % self.stride == 0)
        self._frame_counter += 1
        return result
    
    def add(self, feat: Tensor, sharpness: float):
        """Thêm feature vào cửa sổ (chỉ gọi khi should_extract() == True)."""
        self.features.append(feat)
        self.sharpness_scores.append(sharpness)
        if len(self.features) > self.window_size:
            self.features.pop(0)
            self.sharpness_scores.pop(0)
    
    def is_ready(self) -> bool:
        """Đã đủ k frame chưa?"""
        return len(self.features) >= self.window_size
    
    def get_sequence(self) -> Tensor:
        """Trả về tensor [1, k, 2560] để đưa vào Mamba."""
        return torch.stack(self.features, dim=1)  # [1, k, 2560]
    
    def get_weighted_visual_mean(self) -> Tensor:
        """Tính trung bình có trọng số theo sharpness → vector 2560-dim."""
        weights = torch.tensor(self.sharpness_scores)
        weights = weights / weights.sum()  # Normalize
        stacked = torch.stack([f.squeeze(0) for f in self.features])  # [k, 2560]
        return (stacked * weights.unsqueeze(1).to(stacked.device)).sum(dim=0, keepdim=True)  # [1, 2560]
    
    def clear(self):
        self.features.clear()
        self.sharpness_scores.clear()
        self._frame_counter = 0
```

**Hàm tính sharpness** (dùng Laplacian variance - cực nhẹ, < 0.1ms):
```python
def compute_sharpness(crop_bgr: np.ndarray) -> float:
    """Tính điểm độ nét của ảnh crop bằng Laplacian variance."""
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()
```

### 2.2 Class `TwoTierMemoryBank`

```python
class TwoTierMemoryBank:
    """Ngân Hàng Ký Ức 2 tầng: Anchor (bất biến) + Recent (cửa sổ trượt)."""
    
    def __init__(self, max_anchor: int = 10, max_recent: int = 30):
        self.max_anchor = max_anchor
        self.max_recent = max_recent
        # Mỗi entry là một dict: {"visual": Tensor[1,2560], "fused": Tensor[1,3072]}
        self.anchor_bank: List[dict] = []
        self.recent_bank: List[dict] = []
    
    def add_anchor(self, visual_feat: Tensor, fused_feat: Tensor):
        """Thêm vào Anchor Bank (chỉ khi chưa đầy)."""
        if len(self.anchor_bank) < self.max_anchor:
            self.anchor_bank.append({
                "visual": F.normalize(visual_feat, p=2, dim=1),
                "fused": F.normalize(fused_feat, p=2, dim=1)
            })
    
    def add_recent(self, visual_feat: Tensor, fused_feat: Tensor):
        """Thêm vào Recent Bank (cửa sổ trượt, pop cái cũ nhất khi đầy)."""
        self.recent_bank.append({
            "visual": F.normalize(visual_feat, p=2, dim=1),
            "fused": F.normalize(fused_feat, p=2, dim=1)
        })
        if len(self.recent_bank) > self.max_recent:
            self.recent_bank.pop(0)
    
    def coarse_score(self, query_visual: Tensor) -> float:
        """So sánh cosine giữa query 2560-dim với tất cả visual vectors trong bank."""
        query = F.normalize(query_visual, p=2, dim=1)
        max_sim = 0.0
        for entry in self.anchor_bank + self.recent_bank:
            sim = torch.mm(query, entry["visual"].t()).item()
            max_sim = max(max_sim, sim)
        return max_sim
    
    def fine_score(self, query_fused: Tensor) -> float:
        """So sánh cosine giữa query 3072-dim với tất cả fused vectors trong bank."""
        query = F.normalize(query_fused, p=2, dim=1)
        max_sim = 0.0
        for entry in self.anchor_bank + self.recent_bank:
            sim = torch.mm(query, entry["fused"].t()).item()
            max_sim = max(max_sim, sim)
        return max_sim
    
    def is_empty(self) -> bool:
        return len(self.anchor_bank) == 0 and len(self.recent_bank) == 0
    
    def freeze(self):
        """Khóa bank (gọi khi chuyển sang LOST). Thực tế chỉ cần dừng add."""
        pass  # Logic dừng cập nhật nằm ở ReIDPipeline
    
    def size_info(self) -> str:
        return f"Anchor: {len(self.anchor_bank)}/{self.max_anchor} | Recent: {len(self.recent_bank)}/{self.max_recent}"
```

### 2.3 Class `ReIDPipeline` (State Machine)

```python
class ReIDPipeline:
    """State Machine quản lý toàn bộ luồng ReID."""
    
    # Trạng thái
    T0_INIT = "T0_INIT"           # Khởi tạo, xây dựng Memory Bank
    T1_LOST = "T1_LOST"           # Mất track, đang tìm kiếm
    T2_SEARCH = "T2_SEARCH"       # Đang lọc Thô/Tinh (có Soft Lock)
    T3_VERIFIED = "T3_VERIFIED"   # Hard Lock, bám sát + chống hijack
    
    def __init__(self, model, device, cfg):
        self.model = model
        self.device = device
        self.state = self.T0_INIT
        
        # Config
        self.stride = cfg.get('stride', 2)
        self.num_frames = cfg.get('num_frames', 16)
        self.soft_lock_threshold = cfg.get('soft_lock_threshold', 0.50)
        self.reid_threshold = cfg.get('reid_threshold', 0.75)
        self.hijack_threshold = cfg.get('hijack_threshold', 0.40)
        self.hijack_check_count = cfg.get('hijack_check_count', 5)  # x lần
        self.update_interval_sec = cfg.get('update_interval_sec', 2.0)  # t giây
        self.lost_threshold = cfg.get('lost_threshold', 10)
        
        # Components
        self.memory_bank = TwoTierMemoryBank(
            max_anchor=cfg.get('max_anchor_size', 10),
            max_recent=cfg.get('max_recent_size', 30)
        )
        self.sliding_window = SlidingWindowBuffer(
            window_size=self.num_frames,
            stride=self.stride
        )
        
        # Tracking state
        self.target_track_id = None
        self.last_target_center = None
        self.lost_count = 0
        self.last_update_time = 0.0
        self._hijack_checks_remaining = 0  # Đếm ngược kiểm tra anti-hijack
        
        # Search state (T2)
        self.soft_lock_id = None
        self.soft_lock_buffer = SlidingWindowBuffer(
            window_size=self.num_frames, stride=1  # Không stride khi đang verify
        )
        self.candidate_scores = {}  # {track_id: coarse_score}
```

---

## 3. Hướng dẫn code từng phần

### Phần 0: Hàm tiện ích (Utility Functions)

Giữ nguyên các hàm sau từ code cũ (không cần thay đổi):
- `parse_args()` 
- `extract_cnn_feature(model, tensor_frame)` → trả về vector **2560-dim**
- `compute_reid_embedding(model, seq_feats)` → trả về vector **3072-dim** (đã normalize)

**Thêm mới:**
```python
def compute_sharpness(crop_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def compute_fused_vector(model, sliding_window: SlidingWindowBuffer) -> tuple:
    """
    Từ sliding window đầy đủ k frames, tính ra cặp (visual_2560, fused_3072).
    Returns:
        visual_mean: Tensor [1, 2560] - trung bình có trọng số sharpness
        fused_feat: Tensor [1, 3072] - vector hoàn chỉnh qua Mamba + BNNeck
    """
    # 1. Visual: Trung bình có trọng số theo sharpness
    visual_mean = sliding_window.get_weighted_visual_mean()  # [1, 2560]
    
    # 2. Temporal: Chạy Mamba trên toàn bộ sequence
    seq = sliding_window.get_sequence()  # [1, k, 2560]
    fused_feat = compute_reid_embedding(model, seq)  # [1, 3072]
    
    return visual_mean, fused_feat
```

**Xoá bỏ:**
- `expand_coarse_to_fine()` — không cần nữa vì Lọc Thô không chạy Mamba.

---

### Phần 1: Giai đoạn T0 — Khởi tạo và Lưu trữ Ký ức

**Kịch bản:** UAV mục tiêu đang được tracking, hệ thống liên tục trích xuất đặc trưng.

```python
# Trong vòng lặp chính, khi state == T0_INIT hoặc T3_VERIFIED:

if self.state in [self.T0_INIT, self.T3_VERIFIED]:
    target_box = find_target_box(filtered_boxes, self.target_track_id)
    
    if target_box is not None:
        self.lost_count = 0
        crop, center = crop_and_pad(frame, target_box, bbox_padding)
        self.last_target_center = center
        
        # === STRIDE SAMPLING ===
        if self.sliding_window.should_extract():
            # Chỉ chạy GASNet khi stride cho phép (tiết kiệm ~50ms mỗi frame bỏ qua)
            sharpness = compute_sharpness(crop)
            tensor_frame = transform(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device)
            feat_2560 = extract_cnn_feature(model, tensor_frame)  # ~50ms
            self.sliding_window.add(feat_2560, sharpness)
        
        # === CẬP NHẬT MEMORY BANK (mỗi t giây) ===
        current_time = time.time()
        time_elapsed = current_time - self.last_update_time
        
        if self.sliding_window.is_ready() and time_elapsed >= self.update_interval_sec:
            visual_mean, fused_feat = compute_fused_vector(model, self.sliding_window)  # ~8.5ms
            
            if self.state == self.T0_INIT:
                # Anchor Bank: Ghi nhận hình dáng gốc
                self.memory_bank.add_anchor(visual_mean, fused_feat)
                print(f"[{frame_idx}] Anchor updated. {self.memory_bank.size_info()}")
                
                # Chuyển sang T3 khi Anchor Bank đã đủ (hoặc đã có ít nhất 1 entry)
                if len(self.memory_bank.anchor_bank) >= 1:
                    self.state = self.T3_VERIFIED
                    self._hijack_checks_remaining = 0  # Không cần check hijack ở T0→T3
                    print(f"[{frame_idx}] T0 → T3_VERIFIED. Memory Bank initialized.")
            
            elif self.state == self.T3_VERIFIED:
                # Recent Bank: Cập nhật sliding window mới nhất
                self.memory_bank.add_recent(visual_mean, fused_feat)
                
                # === ANTI-HIJACK CHECK (chỉ x lần đầu sau Hard Lock) ===
                if self._hijack_checks_remaining > 0:
                    hijack_score = self.memory_bank.fine_score(fused_feat)
                    self._hijack_checks_remaining -= 1
                    print(f"[{frame_idx}] Anti-Hijack check #{self.hijack_check_count - self._hijack_checks_remaining}: score={hijack_score:.3f}")
                    
                    if hijack_score < self.hijack_threshold:
                        print(f"[{frame_idx}] ⚠️ HIJACK DETECTED! score={hijack_score:.3f} < {self.hijack_threshold}. → T1_LOST")
                        self._transition_to_lost()
                        continue
                
                print(f"[{frame_idx}] Recent updated. {self.memory_bank.size_info()}")
            
            self.last_update_time = current_time
    
    else:
        # Mục tiêu không thấy trong frame
        self.lost_count += 1
        if self.lost_count >= self.lost_threshold:
            # === CHUYỂN SANG T1: MẤT TRACK ===
            # Cập nhật memory bank lần cuối trước khi mất dấu (nếu có đủ data)
            if self.sliding_window.is_ready():
                visual_mean, fused_feat = compute_fused_vector(model, self.sliding_window)
                if self.state == self.T0_INIT:
                    self.memory_bank.add_anchor(visual_mean, fused_feat)
                else:
                    self.memory_bank.add_recent(visual_mean, fused_feat)
                print(f"[{frame_idx}] Last-moment Memory Bank update before LOST.")
            
            self._transition_to_lost()
```

**Hàm chuyển trạng thái:**
```python
def _transition_to_lost(self):
    """Chuyển sang trạng thái LOST."""
    print(f"Target {self.target_track_id} LOST! → T1_LOST")
    self.state = self.T1_LOST
    self.target_track_id = None
    self.lost_count = 0
    self.sliding_window.clear()
    self.soft_lock_id = None
    self.soft_lock_buffer.clear()
    self.candidate_scores.clear()
    # Memory Bank được giữ nguyên (freeze)
```

---

### Phần 2: Giai đoạn T1 — Mất Track

Giai đoạn này chỉ là trạng thái chờ, không có logic phức tạp. Khi có UAV xuất hiện → tự động chuyển sang T2.

```python
if self.state == self.T1_LOST:
    if len(filtered_boxes) > 0 and not self.memory_bank.is_empty():
        # Có UAV mới xuất hiện → Chuyển sang T2 để tìm kiếm
        self.state = self.T2_SEARCH
        self.candidate_scores.clear()
        print(f"[{frame_idx}] UAV detected during LOST state. → T2_SEARCH")
    # Nếu chưa có UAV nào → tiếp tục chờ ở T1
```

---

### Phần 3: Giai đoạn T2 — Tìm Kiếm (Coarse-to-Fine ReID)

Đây là phần **thay đổi nhiều nhất** so với code cũ.

```python
if self.state == self.T2_SEARCH:
    
    # ─── BƯỚC 3.1: LỌC THÔ ───
    # Với mỗi UAV mới chưa có coarse score, chạy GASNet 1 lần
    for box in filtered_boxes:
        tid = int(box.id[0])
        
        if tid in self.candidate_scores:
            continue  # Đã đánh giá rồi
        
        crop, center = crop_and_pad(frame, box, bbox_padding)
        if crop.size == 0:
            continue
        
        # Chạy GASNet → 2560-dim (~50ms)
        tensor_frame = transform(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device)
        feat_2560 = extract_cnn_feature(model, tensor_frame)
        feat_2560_norm = F.normalize(feat_2560, p=2, dim=1)
        
        # So sánh cosine với vector 2560 trong Memory Bank
        coarse_score = self.memory_bank.coarse_score(feat_2560_norm)
        self.candidate_scores[tid] = coarse_score
        
        print(f"[{frame_idx}] Coarse: ID:{tid} score={coarse_score:.3f}")
        
        if coarse_score < self.soft_lock_threshold:
            print(f"[{frame_idx}]   → LOẠI (score < {self.soft_lock_threshold})")
    
    # ─── CHỌN SOFT LOCK ───
    # Chọn UAV có coarse score cao nhất > threshold
    valid_candidates = {
        tid: score for tid, score in self.candidate_scores.items()
        if score >= self.soft_lock_threshold
    }
    
    if valid_candidates:
        best_tid = max(valid_candidates, key=valid_candidates.get)
        
        if self.soft_lock_id != best_tid:
            # Chuyển Soft Lock sang UAV mới
            self.soft_lock_id = best_tid
            self.soft_lock_buffer.clear()
            print(f"[{frame_idx}] SOFT LOCK → ID:{best_tid} (coarse={valid_candidates[best_tid]:.3f})")
    
    # ─── BƯỚC 3.3: XÁC THỰC CHUYÊN SÂU (cho Soft Lock) ───
    if self.soft_lock_id is not None:
        soft_lock_box = find_box_by_id(filtered_boxes, self.soft_lock_id)
        
        if soft_lock_box is not None:
            crop, center = crop_and_pad(frame, soft_lock_box, bbox_padding)
            if crop.size > 0:
                sharpness = compute_sharpness(crop)
                tensor_frame = transform(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device)
                feat_2560 = extract_cnn_feature(model, tensor_frame)
                
                # Thu thập vào buffer (stride=1, mỗi frame đều lấy khi đang verify)
                self.soft_lock_buffer.add(feat_2560, sharpness)
                
                print(f"[{frame_idx}] Soft Lock ID:{self.soft_lock_id} collecting: {len(self.soft_lock_buffer.features)}/{self.num_frames}")
                
                # Đã đủ k frames → Chạy Fine ReID
                if self.soft_lock_buffer.is_ready():
                    visual_mean, fused_feat = compute_fused_vector(model, self.soft_lock_buffer)  # ~8.5ms
                    fine_score = self.memory_bank.fine_score(fused_feat)
                    
                    print(f"[{frame_idx}] Fine ReID: ID:{self.soft_lock_id} score={fine_score:.3f}")
                    
                    if fine_score >= self.reid_threshold:
                        # ═══ HARD LOCK ═══
                        print(f"[{frame_idx}] ✅ HARD LOCK! ID:{self.soft_lock_id} (fine={fine_score:.3f} >= {self.reid_threshold})")
                        self.state = self.T3_VERIFIED
                        self.target_track_id = self.soft_lock_id
                        self.last_target_center = center
                        self._hijack_checks_remaining = self.hijack_check_count  # Bật anti-hijack
                        self.last_update_time = time.time()
                        
                        # Cập nhật Memory Bank với vector mới
                        self.memory_bank.add_recent(visual_mean, fused_feat)
                        
                        # Chuyển buffer Soft Lock → Sliding Window chính
                        self.sliding_window = self.soft_lock_buffer  
                        self.sliding_window.stride = self.stride  # Đặt lại stride
                        
                        # Dọn dẹp
                        self.soft_lock_id = None
                        self.soft_lock_buffer = SlidingWindowBuffer(self.num_frames, stride=1)
                        self.candidate_scores.clear()
                    else:
                        # Fine ReID thất bại → quay về LOST
                        print(f"[{frame_idx}] ❌ Fine FAILED! ID:{self.soft_lock_id} (fine={fine_score:.3f} < {self.reid_threshold}) → T1_LOST")
                        self._transition_to_lost()
        else:
            # Soft Lock UAV biến mất khỏi frame
            print(f"[{frame_idx}] Soft Lock ID:{self.soft_lock_id} lost from frame. Resetting.")
            self.soft_lock_id = None
            self.soft_lock_buffer.clear()
    
    # Nếu không có candidate nào valid → quay về T1
    if not valid_candidates and self.soft_lock_id is None:
        if len(filtered_boxes) == 0:
            self.state = self.T1_LOST
```

---

### Phần 4: Hàm tiện ích bổ sung

```python
def find_target_box(filtered_boxes, target_id):
    """Tìm bounding box của target trong danh sách detection."""
    for box in filtered_boxes:
        if box.id is not None and int(box.id[0]) == target_id:
            return box
    return None

def find_box_by_id(filtered_boxes, track_id):
    """Tìm bounding box theo track ID."""
    return find_target_box(filtered_boxes, track_id)

def crop_and_pad(frame, box, bbox_padding):
    """Crop và pad bounding box từ frame, trả về (crop_bgr, center_xy)."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
    bw, bh = x2 - x1, y2 - y1
    center = (x1 + bw / 2.0, y1 + bh / 2.0)
    
    pad_w, pad_h = int(bw * bbox_padding), int(bh * bbox_padding)
    x1_p = max(0, x1 - pad_w)
    y1_p = max(0, y1 - pad_h)
    x2_p = min(w, x2 + pad_w)
    y2_p = min(h, y2 + pad_h)
    
    crop = frame[y1_p:y2_p, x1_p:x2_p]
    return crop, center
```

---

### Phần 5: Vòng lặp chính (`main()`)

```python
def main():
    args = parse_args()
    cfg = load_config(args.config)
    
    # Khởi tạo model, YOLO, video capture (giữ nguyên code cũ)
    ...
    
    # Khởi tạo ReID Pipeline
    pipeline = ReIDPipeline(model, device, cfg['infer_realworld'])
    
    # Xử lý Query Image (nếu có)
    if query_img_path and os.path.exists(query_img_path):
        q_feat = extract_query_feature(model, query_img_path, transform, device)
        pipeline.memory_bank.add_anchor(q_feat['visual'], q_feat['fused'])
        pipeline.state = ReIDPipeline.T1_LOST  # Bắt đầu ở trạng thái tìm kiếm
    else:
        pipeline.state = ReIDPipeline.T0_INIT   # Bắt mù, auto-lock UAV đầu tiên
    
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # YOLO Detection + Tracking
        results = yolo_model.track(frame, persist=True, ...)
        filtered_boxes = filter_boxes(results)
        display_frame = frame.copy()
        
        # ════════════════════════════════════════
        # STATE MACHINE
        # ════════════════════════════════════════
        
        if pipeline.state == ReIDPipeline.T0_INIT:
            # Auto-lock UAV đầu tiên nếu chưa có target
            if pipeline.target_track_id is None and len(filtered_boxes) > 0:
                best_box = max(filtered_boxes, key=lambda b: float(b.conf[0]))
                pipeline.target_track_id = int(best_box.id[0])
            # Sau đó logic T0 như Phần 1 ở trên
            pipeline.process_tracking(frame, filtered_boxes, frame_idx, transform, device)
        
        elif pipeline.state == ReIDPipeline.T1_LOST:
            pipeline.process_lost(filtered_boxes, frame_idx)
        
        elif pipeline.state == ReIDPipeline.T2_SEARCH:
            pipeline.process_search(frame, filtered_boxes, frame_idx, transform, device)
        
        elif pipeline.state == ReIDPipeline.T3_VERIFIED:
            pipeline.process_tracking(frame, filtered_boxes, frame_idx, transform, device)
        
        # Vẽ UI (giữ nguyên logic vẽ bbox + status text từ code cũ)
        draw_ui(display_frame, pipeline, filtered_boxes)
        
        out.write(display_frame)
        frame_idx += 1
```

---

## 4. Thay đổi Config

Thêm các config mới vào [`configs/config_jetson.yaml`](file:///home/namm/thang/UAVAntiUAV/configs/config_jetson.yaml) trong section `infer_realworld`:

```yaml
infer_realworld:
  # ... (giữ nguyên config cũ) ...
  
  # ═══ CONFIG MỚI ═══
  stride: 2                    # Chỉ trích xuất GASNet mỗi frame thứ 2
  soft_lock_threshold: 0.50    # Ngưỡng Lọc Thô (đổi tên từ coarse_threshold)
  hijack_threshold: 0.40       # Ngưỡng phát hiện bám nhầm
  hijack_check_count: 5        # Số lần kiểm tra anti-hijack sau Hard Lock
  update_interval_sec: 2.0     # Cập nhật Memory Bank mỗi 2 giây
  
  # ═══ CONFIG BỊ XOÁ ═══
  # spatial_weight: 0.30       # ← XOÁ (không dùng Spatial Penalty nữa)
  # coarse_threshold: 0.50     # ← ĐỔI TÊN thành soft_lock_threshold
```

---

## 5. Checklist hoàn thành

- [ ] **Tạo class `SlidingWindowBuffer`** với stride sampling + sharpness tracking
- [ ] **Tạo hàm `compute_sharpness()`** (Laplacian variance)
- [ ] **Tạo class `TwoTierMemoryBank`** lưu cặp `(visual_2560, fused_3072)` thay vì chỉ 3072
- [ ] **Tạo class `ReIDPipeline`** với 4 trạng thái: `T0_INIT`, `T1_LOST`, `T2_SEARCH`, `T3_VERIFIED`
- [ ] **Viết lại Lọc Thô**: Chỉ dùng GASNet 2560-dim, bỏ `expand_coarse_to_fine()`
- [ ] **Viết lại Lọc Tinh**: Thu thập k frames → Mamba → so sánh 3072-dim
- [ ] **Viết lại Anti-Hijack**: Chỉ check x lần đầu sau Hard Lock, dùng `hijack_threshold`
- [ ] **Viết lại Memory Bank update**: Time-based (mỗi t giây) + trước khi mất dấu
- [ ] **Xoá Spatial Penalty** (bỏ `spatial_weight`, `last_target_center` distance calc)
- [ ] **Xoá Blacklist** (bỏ `blacklisted_ids`)
- [ ] **Thêm hàm tiện ích**: `crop_and_pad()`, `find_target_box()`, `compute_fused_vector()`
- [ ] **Cập nhật config** `config_jetson.yaml` với các tham số mới
- [ ] **Test**: Chạy thử trên video `phanrang_raptor.mp4`
