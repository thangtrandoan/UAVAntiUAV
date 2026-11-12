# Phân Tích Dataset VehicleID

> **Nguồn**: Zapletal & Herout, *"Vehicle Re-Identification for Automatic Video Traffic Surveillance"*, CVPR 2016 Workshop  
> **Đường dẫn**: `./data/VehicleID/`  
> **Tổng dung lượng**: ~9.57 GB (bao gồm video gốc)

---

## 1. Tổng Quan

Dataset VehicleID được thiết kế cho bài toán **Vehicle Re-Identification (Re-ID)** trong hệ thống giám sát giao thông nhiều camera. Mục tiêu: xác định xem hai ảnh xe chụp từ hai camera khác nhau (A và B) có phải cùng một xe hay không.

### Cấu trúc thư mục

```text
data/VehicleID/
├── img_database/              # 2,464 ảnh crop xe (150×100 px)
├── 1A/ ... 5A/                # Ảnh trích xuất từ camera A (5 shots)
├── 1B/ ... 5B/                # Ảnh trích xuất từ camera B (5 shots)
├── video_shots/               # 10 video gốc (.mov) + annotations 3D bbox
├── ground_truth_crowdsourced_avg_values.csv    # 1,232 cặp — avg score
├── ground_truth_crowdsourced_values.csv        # 24,530 bản ghi — raw votes
├── ground_truth_shot_1..5.csv                  # Mapping A↔B per shot
├── abstract.txt
├── citation.txt
└── Vehicle Re-Identification ... .pdf
```

---

## 2. Các Thành Phần Dữ Liệu

### 2.1 `img_database/` — Ảnh Crop Xe Cho Crowdsourced Evaluation

| Thuộc tính | Giá trị |
|---|---|
| Tổng ảnh | **2,464** |
| Unique pair IDs | **1,232** (mỗi pair có 1 ảnh A + 1 ảnh B) |
| Kích thước ảnh | **150 × 100 px** (đồng nhất) |
| Format | PNG, RGB |
| Quy tắc đặt tên | `{pairID}_{A|B}_{frameNumber}.png` |

> [!NOTE]
> Mỗi pair là một cặp ảnh từ camera A và camera B. Pair ID chạy từ 1 đến 1232. Ảnh đã được crop và resize thành kích thước cố định 150×100.

### 2.2 Thư mục Shot (1A–5A, 1B–5B) — Ảnh Trích Xuất Từ Video

| Folder | Ảnh Combined | Ảnh Representative | Unique Vehicles | Tổng |
|--------|-------------|-------------------|-----------------|------|
| **1A** | 4,197 | 163 | 163 | 4,360 |
| **1B** | 4,026 | 162 | 162 | 4,188 |
| **2A** | 4,660 | 149 | 149 | 4,809 |
| **2B** | 4,277 | 139 | 139 | 4,416 |
| **3A** | 4,759 | 132 | 132 | 4,891 |
| **3B** | 3,635 | 131 | 131 | 3,766 |
| **4A** | 5,290 | 142 | 142 | 5,432 |
| **4B** | 2,845 | 132 | 132 | 2,977 |
| **5A** | 6,394 | 245 | 245 | 6,639 |
| **5B** | 5,429 | 216 | 216 | 5,645 |
| **Tổng** | **45,512** | **1,611** | — | **47,123** |

| Thuộc tính | Giá trị |
|---|---|
| Kích thước ảnh | **312 × 104 px** (đồng nhất) |
| Format | PNG, RGB |
| Quy tắc đặt tên | `{vehicleID}_{frameNumber}_{combined\|representative}.png` |

- **Combined**: Ảnh ghép 3 mặt xe (3 × 104×104) từ 3D bounding box — gồm mặt dọc, mặt đầu/đuôi, và mặt ngang
- **Representative**: Một ảnh combined đại diện cho xe đó (dùng trong ground truth matching)
- Mỗi xe có trung bình **~25.7 combined images** (min 11, max 42 — ở shot 1A)

### 2.3 `video_shots/` — Video Gốc + Annotations

| Video | Kích thước | Vehicles (Annotated) |
|-------|-----------|---------------------|
| 1A.mov | 727.3 MB | 194 |
| 1B.mov | 933.5 MB | 272 |
| 2A.mov | 697.0 MB | 209 |
| 2B.mov | 907.6 MB | 293 |
| 3A.mov | 532.6 MB | 204 |
| 3B.mov | 700.5 MB | 224 |
| 4A.mov | 567.9 MB | 218 |
| 4B.mov | 746.9 MB | 200 |
| 5A.mov | 1,021.2 MB | 306 |
| 5B.mov | 1,315.3 MB | 399 |
| **Tổng** | **~8.0 GB** | — |

**Annotation format** (`{shot}_annotations.txt`):
```text
%car_id frame UpperPointShort[x y] UpperPointCorner[x y] UpperPointLong[x y] 
       CrossCorner[x y] ShortSide[x y] Corner[x y] LongSide[x y] LowerCrossCorner[x y]
```
- 8 điểm (16 tọa độ) tạo thành **3D bounding box** cho mỗi xe tại mỗi frame
- Tổng: **47,672 annotation entries** trên 10 video

---

## 3. Ground Truth Labels

### 3.1 Crowdsourced Evaluation (`ground_truth_crowdsourced_avg_values.csv`)

| Thuộc tính | Giá trị |
|---|---|
| Số cặp | **1,232** |
| Score range | **[-1.0, +1.0]** |
| Score trung bình | **-0.096** |
| Score trung vị | **-0.400** |
| Độ lệch chuẩn | **0.807** |

**Ý nghĩa score**: Trung bình các votes từ crowdsourcing workers:
- **+1**: Cùng một xe (same vehicle)
- **-1**: Khác xe (different vehicle)
- **0**: Không chắc chắn

**Phân bố score:**

```text
-1.0: ██████████████████████████████ 221  (17.9%)
-0.9: ████████████████████           150  (12.2%)
-0.8: █████████████                   95  ( 7.7%)
-0.7: ████████                        64  ( 5.2%)
-0.6: ██████                          43  ( 3.5%)
-0.5: ████                            32  ( 2.6%)
-0.4: ████                            28  ( 2.3%)
-0.3: ██                              13  ( 1.1%)
-0.2: ██                              13  ( 1.1%)
-0.1: ██                              13  ( 1.1%)
 0.0: █                               11  ( 0.9%)
+0.1: █                               11  ( 0.9%)
+0.2: █                               11  ( 0.9%)
+0.3: ███                             21  ( 1.7%)
+0.4: ████                            28  ( 2.3%)
+0.5: ████                            31  ( 2.5%)
+0.6: ████                            33  ( 2.7%)
+0.7: █████████                       67  ( 5.4%)
+0.8: █████████████                   95  ( 7.7%)
+0.9: ████████████████                120  ( 9.7%)
+1.0: █████████████████               132  (10.7%)
```

**Phân loại binary (threshold = 0)**:
- Same vehicle (score > 0): **549 cặp** (44.6%)
- Different vehicle (score ≤ 0): **683 cặp** (55.4%)

> [!IMPORTANT]
> Phân bố score có dạng **bimodal** — tập trung mạnh ở hai đầu (-1.0 và +1.0), nghĩa là đa số cặp rõ ràng giống hoặc khác xe. Tuy nhiên có ~100-200 cặp ở vùng mờ (-0.5 đến +0.5) đòi hỏi thận trọng khi chọn threshold.

### 3.2 Raw Crowdsourced Votes (`ground_truth_crowdsourced_values.csv`)

| Thuộc tính | Giá trị |
|---|---|
| Tổng votes | **24,530** |
| Unique pairs | **1,232** |
| Votes per pair | min **19**, max **27**, avg **~20** |
| Vote +1 (same) | **11,084** (45.2%) |
| Vote -1 (different) | **13,446** (54.8%) |

### 3.3 Shot Ground Truth (`ground_truth_shot_{1..5}.csv`)

Mapping **xe ở camera A ↔ xe ở camera B** cho từng shot:

| Shot | Matched Pairs |
|------|-------------|
| Shot 1 | 144 |
| Shot 2 | 127 |
| Shot 3 | 120 |
| Shot 4 | 115 |
| Shot 5 | 203 |
| **Tổng** | **709** |

Format: `"{shotA}/{vidA}_{frame}_representative.png", "{shotB}/{vidB}_{frame}_representative.png"`

> [!NOTE]
> Đây là **ground truth tuyệt đối** cho Re-ID — chỉ ra chính xác xe nào ở camera A tương ứng với xe nào ở camera B. Mỗi shot là một cặp camera cố định cùng địa điểm.

---

## 4. Dung Lượng Lưu Trữ

| Thành phần | Kích thước |
|---|---|
| `video_shots/` (10 video .mov) | ~8,153 MB |
| Shot folders (1A–5B) | ~1,589 MB |
| `img_database/` | ~58 MB |
| CSV files + docs | ~2 MB |
| **Tổng** | **~9.57 GB** |

---

## 5. Hướng Dẫn Sử Dụng Cho Training & Evaluation

### 5.1 Bài Toán Chính: Vehicle Re-Identification (Pairwise Verification)

**Task**: Cho 2 ảnh xe, xác định chúng có phải cùng 1 xe không.

#### Phương án A: Sử dụng `img_database/` + Crowdsourced Labels

```python
# Dữ liệu: 1,232 cặp, mỗi cặp 2 ảnh 150×100
# Label: score ∈ [-1, +1] — có thể dùng regression hoặc binary classification

# Binary: threshold = 0 → 549 positive, 683 negative
# Regression: sử dụng score liên tục làm soft label
```

- **Ưu điểm**: Ảnh đã crop/resize sẵn, label sạch (crowdsourced avg), dễ bắt đầu
- **Nhược điểm**: Chỉ 1,232 cặp → nhỏ, dễ overfit
- **Khuyến nghị**: Dùng cho **evaluation/validation**, không đủ lớn để train from scratch

#### Phương án B: Sử dụng Shot Folders + Shot Ground Truth

```python
# Dữ liệu: 47,123 ảnh (combined 312×104 + representative)
# Ground truth: 709 matched pairs giữa camera A↔B
# Có thể sinh thêm negative pairs từ unmatched vehicles

# Ảnh combined: 3 views ghép (312×104) → cắt thành 3 ảnh 104×104
# Hoặc dùng nguyên ảnh 312×104 như multi-view input
```

- **Ưu điểm**: Nhiều ảnh hơn, có multi-frame tracking (nhiều ảnh combined/xe), 3D structure
- **Nhược điểm**: Chỉ 709 positive pairs, cần xây dựng negative pairs
- **Khuyến nghị**: Dùng cho **training** với data augmentation + triplet/contrastive learning

### 5.2 Pipeline Đề Xuất

```text
┌─────────────────────────────────────────────────────────┐
│                    DATA PREPARATION                      │
├─────────────────────────────────────────────────────────┤
│  1. Load shot ground truth → matched pairs (A↔B)        │
│  2. Load combined/representative images per vehicle      │
│  3. Generate positive pairs (matched vehicles)           │
│  4. Generate negative pairs (unmatched vehicles)         │
│  5. Split: train/val by shot (e.g., shots 1-3 train,    │
│     shot 4 val, shot 5 test)                            │
└─────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│                    TRAINING                              │
├─────────────────────────────────────────────────────────┤
│  Model: Siamese / Triplet network                       │
│  Input: Pair of vehicle images (combined hoặc crop)     │
│  Loss: Contrastive loss / Triplet loss                  │
│  Backbone: ResNet-50 / EfficientNet (pretrained)        │
│  Data Aug: horizontal flip, color jitter, random crop   │
└─────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│                    EVALUATION                            │
├─────────────────────────────────────────────────────────┤
│  Metrics: mAP, CMC (Rank-1, Rank-5, Rank-10)           │
│  Protocol: Gallery (camera B) + Query (camera A)        │
│  Hoặc: ROC/AUC trên img_database crowdsourced pairs    │
└─────────────────────────────────────────────────────────┘
```

### 5.3 Các Lưu Ý Quan Trọng

> [!WARNING]
> - **Dataset nhỏ**: Tổng chỉ 709 matched pairs → cần pretrained backbone + strong augmentation
> - **Không có train/test split chính thức**: Cần tự chia theo shot hoặc cross-validation
> - **Ảnh combined 312×104**: Là ghép 3 views — cần quyết định dùng nguyên hay cắt ra
> - **Crowdsourced labels có noise**: Score 0 hoặc gần 0 → ambiguous, nên filter hoặc xử lý riêng

### 5.4 Chiến Lược Chia Dữ Liệu Đề Xuất

| Split | Shots | Matched Pairs | Lý do |
|-------|-------|--------------|-------|
| **Train** | 1, 2, 3 | 391 | 3 shots đa dạng nhất |
| **Val** | 4 | 115 | Nhỏ nhất, dùng để tune |
| **Test** | 5 | 203 | Lớn nhất, đánh giá cuối |

**Evaluation bổ sung**: Dùng `img_database` + `ground_truth_crowdsourced_avg_values.csv` làm independent test set (1,232 pairs).

### 5.5 Code Skeleton cho DataLoader

```python
import os
import csv
from PIL import Image
from torch.utils.data import Dataset

class VehicleReIDDataset(Dataset):
    """Vehicle Re-ID dataset from VehicleID."""
    
    def __init__(self, root_dir, shots, transform=None, mode='pair'):
        """
        Args:
            root_dir: path to data/VehicleID/
            shots: list of shot numbers, e.g. [1, 2, 3]
            transform: image transforms
            mode: 'pair' for pairwise, 'triplet' for triplet
        """
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode
        
        # Load ground truth matched pairs
        self.positive_pairs = []
        for shot in shots:
            gt_file = os.path.join(root_dir, f"ground_truth_shot_{shot}.csv")
            with open(gt_file) as f:
                reader = csv.reader(f)
                for row in reader:
                    img_a = os.path.join(root_dir, row[0].strip().strip('"'))
                    img_b = os.path.join(root_dir, row[1].strip().strip('"'))
                    self.positive_pairs.append((img_a, img_b))
        
        # Build vehicle galleries per shot/camera for negative mining
        self.vehicle_galleries = self._build_galleries(shots)
    
    def _build_galleries(self, shots):
        """Build dict of vehicleID -> list of image paths per camera."""
        galleries = {'A': {}, 'B': {}}
        for shot in shots:
            for cam in ['A', 'B']:
                folder = os.path.join(self.root_dir, f"{shot}{cam}")
                for img_name in os.listdir(folder):
                    vid = img_name.split('_')[0]
                    key = f"{shot}_{vid}"
                    if key not in galleries[cam]:
                        galleries[cam][key] = []
                    galleries[cam][key].append(
                        os.path.join(folder, img_name)
                    )
        return galleries
    
    def __len__(self):
        return len(self.positive_pairs) * 2  # pos + neg
    
    def __getitem__(self, idx):
        # Implementation: return (img_a, img_b, label)
        # label: 1 for same vehicle, 0 for different
        ...


class VehicleReIDCrowdsourcedDataset(Dataset):
    """Evaluation dataset using crowdsourced labels."""
    
    def __init__(self, root_dir, transform=None, threshold=0.0):
        self.root_dir = root_dir
        self.transform = transform
        
        self.pairs = []
        csv_path = os.path.join(root_dir, "ground_truth_crowdsourced_avg_values.csv")
        with open(csv_path) as f:
            reader = csv.reader(f)
            for row in reader:
                img_a = os.path.join(root_dir, row[1])
                img_b = os.path.join(root_dir, row[2])
                score = float(row[3])
                label = 1 if score > threshold else 0
                self.pairs.append((img_a, img_b, score, label))
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        img_a_path, img_b_path, score, label = self.pairs[idx]
        img_a = Image.open(img_a_path).convert('RGB')
        img_b = Image.open(img_b_path).convert('RGB')
        
        if self.transform:
            img_a = self.transform(img_a)
            img_b = self.transform(img_b)
        
        return img_a, img_b, score, label
```

---

## 6. Tóm Tắt Thống Kê

| Metric | Giá trị |
|---|---|
| Tổng ảnh | **49,587** |
| Ảnh trong img_database | 2,464 (1,232 cặp) |
| Ảnh trong shot folders | 47,123 (45,512 combined + 1,611 representative) |
| Số shots (cặp camera) | **5** (10 camera views) |
| Matched pairs (Re-ID GT) | **709** |
| Crowdsourced pairs | **1,232** |
| Crowdsourced votes | **24,530** |
| Kích thước img_database | 150 × 100 px |
| Kích thước shot images | 312 × 104 px |
| Unique vehicles per shot | 115–245 |
| Video gốc | 10 files, ~8 GB |
| 3D BBox annotations | 47,672 entries |
