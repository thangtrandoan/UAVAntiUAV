# PLAN V3 — KẾ HOẠCH CODE CHI TIẾT CHO GEMINI

> **Mục tiêu:** Refactor codebase hiện tại theo kiến trúc Description_v3.md.
> Modular, production-ready, train/evaluate end-to-end.
>
> **Nguyên tắc:** Tận dụng tối đa code đã có, chỉ thay đổi/bổ sung phần cần thiết.

---

## 0. PHÂN TÍCH CODEBASE HIỆN TẠI vs YÊU CẦU V3

### Kiến trúc hiện tại (đang hoạt động)

```
UAVReIDNet (model.py)
├── backbone: GASNet (ResNet50-IBN + RGA + OSBlockFS, pretrained VRU)
│   ├── Output: (global_feat=2048, fs_feat=512) → concat → 2560-d
│   └── Hỗ trợ sẵn SwinBackbone trong gasnet/train.py
├── temporal_encoder: TemporalMambaEncoder
│   ├── Input: 2560-d → proj → 512-d
│   ├── Bidirectional Mamba (2 layers) + Mean Pooling
│   └── Output: 512-d temporal token
├── head: ReIDHead
│   ├── Input: 2560 + 512 = 3072-d
│   ├── BNNeck (non-learnable shift)
│   └── Classifier → num_identities
└── Training: Siamese (before_clips, after_clips) → 2-stage freeze/unfreeze
```

### Mapping: Hiện tại → Description_v3

| Component | Hiện tại | Description_v3 yêu cầu | Gap Analysis |
|-----------|----------|------------------------|--------------|
| **Backbone** | GASNet (ResNet50-IBN) `2560-d` | HiViT-Base hoặc Swin-Transformer `768-d` | **GASNet đã có `SwinBackbone` trong `gasnet/train.py`**. Cần switch sang Swin + điều chỉnh feature dim |
| **Feature Dim** | 2560 (visual) + 512 (temporal) = 3072 | 768-d thống nhất | **Thay đổi lớn**: cần thay đổi tất cả dims |
| **Mamba Engine** | Bidirectional, 2 layers, 512-d, mean pool | Unidirectional 1D scan, O(N) | **Cần đổi về unidirectional** theo spec. Thêm return intermediate states |
| **Losses** | 4 losses inline trong `train_reid.py`. **BUG: `lam2` (temporal) không được dùng trong loss computation** | Tách ra `losses.py`. Fix temporal loss | **Phải tách file + fix bug** |
| **Data Sampling** | `_load_clip` uniform sampling. `ReIDBatchSampler` (PK) | Anchor-first boundary alignment, stride=3 | **Cần refactor sampling logic** |
| **Data Mining** | Không có | Smart Mining + Synthetic Augmentation | **Tạo mới `data_miner.py`** |
| **Augmentation** | RandomErasing, GaussianBlur, ColorJitter | Thêm MotionBlur, ScaleJitter, Synthetic Masking | **Bổ sung** |
| **Evaluate** | 2 protocols (offline + online). **BUG: online latency dùng random number** | 3 scenarios + real online simulation | **Fix + nâng cấp** |
| **Config** | YAML configs cho nhiều platform (Colab, H100, Jetson) | Config mới cho model v3 | **Thêm config mới** |

### Bugs cần fix (ưu tiên cao)

1. **`train_reid.py` line 436**: `loss = loss_id + lam1 * loss_tri + lam3 * loss_center` — **THIẾU `lam2 * loss_temporal`**. Temporal Consistency Loss được define nhưng KHÔNG BAO GIỜ được gọi trong training loop.
2. **`evaluate_reid.py` ~line 285**: Online latency dùng `np.random.randint(1, 5)` thay vì tính thực tế.
3. **`model.py` line 210**: Comment ghi `[B, 2816]` nhưng thực tế là `[B, 3072]`.
4. **`train_reid.py` line 95**: `CenterLoss` default `feat_dim=2816` nhưng instantiate với `3072` — inconsistent default.

---

## 1. CẤU TRÚC THƯ MỤC MỚI

```
UAVAntiUAV/
├── configs/
│   ├── config_v3.yaml             # ← MỚI: Config chính cho v3
│   ├── config_colab.yaml          # Giữ nguyên
│   ├── config_h100.yaml           # Giữ nguyên
│   ├── config_jetson.yaml         # Giữ nguyên
│   └── ...
├── models/
│   ├── __init__.py                # ← MỚI: Export UAVMambaReID
│   ├── backbone.py                # ← MỚI: Swin backbone + VRU weight loading
│   ├── mamba_engine.py            # ← MỚI: Tách từ model.py, thêm unidirectional + intermediates
│   └── reid_model.py             # ← MỚI: Main model (thay thế model.py)
├── data/
│   ├── __init__.py                # ← MỚI
│   ├── dataset.py                 # ← MỚI: Refactor từ train_reid.py UAVReIDDataset
│   ├── data_miner.py             # ← MỚI: Smart Mining + Synthetic Augmentation
│   ├── sampler.py                 # ← MỚI: Tách PKSampler từ train_reid.py
│   └── transforms.py             # ← MỚI: Tách transforms + thêm MotionBlur, ScaleJitter
├── losses.py                      # ← MỚI: Tách từ train_reid.py + fix bugs
├── train.py                       # ← MỚI: Refactor từ train_reid.py
├── evaluate.py                    # ← MỚI: Refactor từ evaluate_reid.py
├── process_rgbt_pipeline.py       # Giữ nguyên
├── data_pipeline.py               # Giữ nguyên (preprocessing)
├── model.py                       # GIỮ NGUYÊN (backward compat, old checkpoints)
├── train_reid.py                  # GIỮ NGUYÊN (backward compat)
├── evaluate_reid.py               # GIỮ NGUYÊN (backward compat)
├── gasnet/                        # Giữ nguyên
├── requirements.txt               # Cập nhật
├── Dockerfile                     # Cập nhật
├── entrypoint.sh                  # Cập nhật
└── ...
```

> **Lưu ý quan trọng:** KHÔNG XÓA các file cũ (`model.py`, `train_reid.py`, `evaluate_reid.py`). Chúng vẫn cần cho backward compatibility và load old checkpoints.

---

## 2. CHI TIẾT TỪNG MODULE

---

### 2.1. `models/backbone.py` — Visual Backbone (Swin-Transformer)

**Mục tiêu:** Tạo wrapper backbone mới dùng Swin-Transformer, load VRU pretrained weights. Tham khảo `SwinBackbone` đã có trong `gasnet/train.py` nhưng **viết lại gọn hơn**, output 768-d.

#### Thiết kế:

```python
import timm
import torch
import torch.nn as nn

class VRUBackbone(nn.Module):
    """
    Hierarchical Vision Transformer backbone cho UAV ReID.
    Dùng Swin-Transformer pretrained, output 768-d feature vector.
    
    Hỗ trợ:
    - swin_base_patch4_window7_224  (output 1024-d → project → 768)
    - swin_small_patch4_window7_224 (output 768-d, không cần project)
    - swin_tiny_patch4_window7_224  (output 768-d, nhẹ nhất)
    """
    
    def __init__(self, arch='swin_base_patch4_window7_224', 
                 pretrained_path=None, feat_dim=768, pretrained_imagenet=True):
        super().__init__()
        self.feat_dim = feat_dim
        
        # (1) Tạo backbone từ timm
        self.backbone = timm.create_model(arch, pretrained=pretrained_imagenet, num_classes=0)
        # num_classes=0 → bỏ classifier head, trả feature vector
        
        backbone_out_dim = self.backbone.num_features  
        # swin_base: 1024, swin_small: 768, swin_tiny: 768
        
        # (2) Projection layer nếu dim không khớp
        if backbone_out_dim != feat_dim:
            self.proj = nn.Sequential(
                nn.Linear(backbone_out_dim, feat_dim),
                nn.BatchNorm1d(feat_dim),
                nn.ReLU(inplace=True)
            )
        else:
            self.proj = nn.Identity()
        
        # (3) Load VRU pretrained weights (nếu có)
        if pretrained_path:
            self.load_vru_pretrained(pretrained_path)
    
    def forward(self, x):
        """
        Input:  (B, 3, H, W) — RGB image batch
        Output: (B, feat_dim) — 768-d feature vector
        """
        features = self.backbone(x)       # (B, backbone_out_dim)
        features = self.proj(features)     # (B, feat_dim=768)
        return features
    
    def load_vru_pretrained(self, checkpoint_path):
        """
        Load VRU dataset pretrained weights.
        - Filter out classifier/head keys
        - Partial key matching (strict=False)
        - Log missing/unexpected keys
        """
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        
        # Handle wrapped state dicts
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        elif 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        
        # Filter out classifier keys
        filtered = {k: v for k, v in state_dict.items() 
                    if not any(x in k for x in ['classifier', 'head.fc', 'head.weight', 'head.bias'])}
        
        missing, unexpected = self.backbone.load_state_dict(filtered, strict=False)
        print(f"[VRUBackbone] Loaded pretrained weights from {checkpoint_path}")
        if missing:
            print(f"  Missing keys: {len(missing)}")
        if unexpected:
            print(f"  Unexpected keys: {len(unexpected)}")
```

#### Quyết định thiết kế:
- **Dùng `timm` thay vì viết Swin từ đầu**: `timm` đã có Swin pretrained ImageNet, chỉ cần load thêm VRU weights lên trên.
- **`num_classes=0`**: Trick của timm để bỏ classifier head, trả feature pooling.
- **Projection `Linear(1024, 768)` cho Swin-Base**: Swin-Base output 1024-d, cần project xuống 768 theo spec.
- **Swin-Small/Tiny output 768 sẵn**: Nếu dùng model nhỏ hơn thì `proj = Identity()`.

---

### 2.2. `models/mamba_engine.py` — Temporal Memory Engine

**Mục tiêu:** Tách `TemporalMambaEncoder` từ `model.py`, chuyển sang **1D Unidirectional** theo spec, thêm `return_intermediate` cho Temporal Consistency Loss.

#### Thay đổi so với hiện tại:

| Hiện tại (`model.py`) | V3 mới |
|------------------------|--------|
| `d_in=2560, d_model=512, d_out=512` | `d_in=768, d_model=768` (thống nhất) |
| Bidirectional: `x_fwd + x_bwd + res` | **Unidirectional**: chỉ `x_fwd + res` |
| Mean pooling | Giữ mean pooling |
| Không trả intermediate | Thêm `return_intermediate=True` |
| MLP Head (Linear → BN → ReLU) | Bỏ MLP Head (không cần project khi dim đã khớp) |

#### Thiết kế:

```python
import torch
import torch.nn as nn

# Import mamba_ssm hoặc fallback
try:
    from mamba_ssm import Mamba
    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False
    # Import SimpleS6Block fallback (copy từ model.py cũ, sửa lại dims)

class MambaTemporalBlock(nn.Module):
    """Single Mamba block: LayerNorm → Mamba → Residual."""
    
    def __init__(self, d_model=768, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        if HAS_MAMBA:
            self.mamba = Mamba(d_model=d_model, d_state=d_state, 
                              d_conv=d_conv, expand=expand)
        else:
            self.mamba = SimpleS6Block(d_model=d_model, d_state=d_state,
                                       d_conv=d_conv, expand=expand)
    
    def forward(self, x):
        """
        Input/Output: (B, T, D)
        Pre-norm residual: x + Mamba(LayerNorm(x))
        """
        residual = x
        x = self.norm(x)
        x = self.mamba(x)
        return x + residual


class TemporalMambaEngine(nn.Module):
    """
    Vision Mamba Temporal Memory Engine.
    
    Key changes from v2:
    - 1D Unidirectional scanning (theo Description_v3 spec)
    - Return intermediate states cho Temporal Consistency Loss
    - d_model = feat_dim = 768 (no projection needed when backbone outputs 768)
    """
    
    def __init__(self, d_model=768, d_state=16, d_conv=4, expand=2, 
                 n_layers=2, max_seq_len=64):
        super().__init__()
        self.d_model = d_model
        
        # Learnable positional encoding
        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)
        
        # Mamba blocks
        self.blocks = nn.ModuleList([
            MambaTemporalBlock(d_model, d_state, d_conv, expand)
            for _ in range(n_layers)
        ])
        
        # Temporal pooling (AdaptiveAvgPool1d)
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)
    
    def forward(self, x, return_intermediate=False):
        """
        Args:
            x: (B, T, D) — sequence of frame features
            return_intermediate: if True, return hidden states for each layer
        
        Returns:
            pooled: (B, D) — temporal token / memory vector
            intermediates: list of (B, T, D) — one per Mamba layer 
                          (chỉ khi return_intermediate=True)
        """
        B, T, D = x.shape
        
        # Add positional encoding
        x = x + self.pos_embed[:, :T, :]
        
        # Process through Mamba blocks (1D UNIDIRECTIONAL)
        intermediates = []
        for block in self.blocks:
            x = block(x)                        # (B, T, D) → (B, T, D)
            if return_intermediate:
                intermediates.append(x)
        
        # Temporal mean pooling: (B, T, D) → (B, D)
        pooled = x.mean(dim=1)
        
        if return_intermediate:
            return pooled, intermediates
        return pooled
```

#### Giải thích quyết định:
- **Bỏ `in_proj` (Linear 2560→512)**: Vì backbone mới output 768 sẵn, `d_model=768`, không cần project.
- **Bỏ `out_mlp`**: Không cần project output khi dim đã thống nhất 768.
- **Bỏ bidirectional**: Description_v3 spec ghi "1D Unidirectional Scanning". Model cũ dùng bidirectional (`x_fwd + x_bwd + res`) — cần chuyển lại.
- **Giữ `SimpleS6Block` fallback**: Copy từ `model.py` cũ, chỉ đổi default dims.
- **`return_intermediate`**: Cần thiết cho `TemporalConsistencyLoss`.

---

### 2.3. `models/reid_model.py` — Main ReID Model

**Mục tiêu:** Thay thế `UAVReIDNet` từ `model.py`, dùng `VRUBackbone` + `TemporalMambaEngine` mới.

#### Thay đổi kiến trúc chính:

| Component | Hiện tại (model.py) | V3 mới |
|-----------|---------------------|--------|
| Backbone | GASNet → 2560-d (2048+512) | VRUBackbone (Swin) → 768-d |
| Temporal | TemporalMambaEncoder 2560→512 bidir | TemporalMambaEngine 768→768 unidir |
| ReIDHead | concat(2560, 512)=3072 → BNNeck → classifier | 768 → BNNeck → classifier |
| Training | Siamese (before, after) → 2 sets outputs | Giữ Siamese |

#### Thiết kế:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import VRUBackbone
from .mamba_engine import TemporalMambaEngine

def weights_init_kaiming(m):
    """Copy từ model.py cũ — giữ nguyên."""
    ...

def weights_init_classifier(m):
    """Copy từ model.py cũ — giữ nguyên."""
    ...


class ReIDHead(nn.Module):
    """
    ReID classification head.
    Input: feat_dim (768) → BNNeck → Classifier
    
    THAY ĐỔI từ model.py cũ:
    - Input dim: 3072 → 768
    - Không cần concat visual + temporal (đã gộp qua Mamba pooling)
    """
    
    def __init__(self, feat_dim=768, num_classes=1000):
        super().__init__()
        self.bnneck = nn.BatchNorm1d(feat_dim)
        self.bnneck.bias.requires_grad_(False)
        self.bnneck.apply(weights_init_kaiming)
        
        self.classifier = nn.Linear(feat_dim, num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)
    
    def forward(self, features):
        """
        Args:
            features: (B, feat_dim) — temporal pooled features
        
        Returns (training):
            raw_feat:  (B, feat_dim) — before BNNeck, cho Triplet/Center loss
            bn_feat:   (B, feat_dim) — after BNNeck
            logits:    (B, num_classes) — cho ID loss
        
        Returns (eval):
            bn_feat:   (B, feat_dim) — normalized, cho matching
        """
        bn_feat = self.bnneck(features)
        
        if not self.training:
            return F.normalize(bn_feat, p=2, dim=1)
        
        logits = self.classifier(bn_feat)
        return features, bn_feat, logits


class UAVMambaReID(nn.Module):
    """
    Complete UAV ReID model v3:
    VRU Backbone → per-frame features → Mamba Engine → temporal pooling → BNNeck → Classifier
    
    KEY DIFFERENCE từ UAVReIDNet cũ:
    1. Backbone output 768-d (không phải 2560)
    2. Mamba unidirectional (không bidirectional)
    3. ReIDHead nhận 768-d (không concat 2560+512=3072)
    4. Return intermediate states cho Temporal Consistency Loss
    """
    
    def __init__(self, backbone_arch='swin_base_patch4_window7_224',
                 pretrained_path=None, num_classes=1000, feat_dim=768,
                 n_mamba_layers=2, freeze_backbone=True):
        super().__init__()
        self.feat_dim = feat_dim
        
        # 1. Visual Backbone (Swin-Transformer)
        self.backbone = VRUBackbone(
            arch=backbone_arch,
            pretrained_path=pretrained_path,
            feat_dim=feat_dim
        )
        
        # 2. Temporal Memory Engine (Mamba)
        self.temporal_engine = TemporalMambaEngine(
            d_model=feat_dim,
            n_layers=n_mamba_layers
        )
        
        # 3. ReID Head
        self.head = ReIDHead(feat_dim=feat_dim, num_classes=num_classes)
        
        # Freeze backbone ban đầu
        if freeze_backbone:
            self.freeze_backbone()
    
    def freeze_backbone(self):
        """Freeze backbone cho warm-up epochs."""
        print("[UAVMambaReID] Freezing backbone...")
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze backbone cho end-to-end fine-tuning."""
        print("[UAVMambaReID] Unfreezing backbone...")
        for param in self.backbone.parameters():
            param.requires_grad = True
    
    def extract_features(self, clips, return_intermediate=False):
        """
        Extract features từ sequence clips.
        
        Args:
            clips: (B, T, C, H, W)
            return_intermediate: return mamba hidden states
        
        Returns:
            pooled: (B, feat_dim) — temporal token
            intermediates: list of (B, T, feat_dim) — nếu return_intermediate
        """
        B, T, C, H, W = clips.shape
        
        # (1) Per-frame backbone extraction
        # Reshape: (B*T, C, H, W) → backbone → (B*T, feat_dim)
        frames = clips.view(B * T, C, H, W)
        
        # Tối ưu memory: không track gradient nếu backbone frozen
        if not next(self.backbone.parameters()).requires_grad:
            self.backbone.eval()
        grad_enabled = next(self.backbone.parameters()).requires_grad and self.training
        with torch.set_grad_enabled(grad_enabled):
            frame_features = self.backbone(frames)   # (B*T, 768)
        
        frame_features = frame_features.view(B, T, -1)  # (B, T, 768)
        
        # (2) Temporal modeling
        if return_intermediate:
            pooled, intermediates = self.temporal_engine(frame_features, return_intermediate=True)
            return pooled, intermediates
        else:
            pooled = self.temporal_engine(frame_features)
            return pooled
    
    def forward(self, before_clips, after_clips=None, return_intermediate=False):
        """
        Forward pass.
        
        Inference mode:
            Input: before_clips (B, T, C, H, W)
            Output: bn_features (B, feat_dim) — L2-normalized
        
        Training mode (Siamese):
            Input: before_clips, after_clips
            Output: 
                ((raw_b, bn_b, logits_b, intermediates_b),
                 (raw_a, bn_a, logits_a, intermediates_a))
        """
        if not self.training:
            # Inference: chỉ extract + BNNeck + normalize
            pooled = self.extract_features(before_clips, return_intermediate=False)
            return self.head(pooled)  # L2-normalized bn_feat
        
        # Training: Siamese
        if return_intermediate:
            pooled_b, inter_b = self.extract_features(before_clips, return_intermediate=True)
            raw_b, bn_b, logits_b = self.head(pooled_b)
            
            if after_clips is not None:
                pooled_a, inter_a = self.extract_features(after_clips, return_intermediate=True)
                raw_a, bn_a, logits_a = self.head(pooled_a)
                return (raw_b, bn_b, logits_b, inter_b), (raw_a, bn_a, logits_a, inter_a)
            
            return raw_b, bn_b, logits_b, inter_b
        else:
            pooled_b = self.extract_features(before_clips, return_intermediate=False)
            raw_b, bn_b, logits_b = self.head(pooled_b)
            
            if after_clips is not None:
                pooled_a = self.extract_features(after_clips, return_intermediate=False)
                raw_a, bn_a, logits_a = self.head(pooled_a)
                return (raw_b, bn_b, logits_b), (raw_a, bn_a, logits_a)
            
            return raw_b, bn_b, logits_b
```

#### `models/__init__.py`:
```python
from .reid_model import UAVMambaReID
from .backbone import VRUBackbone
from .mamba_engine import TemporalMambaEngine
```

---

### 2.4. `losses.py` — Multi-Task Joint Loss

**Mục tiêu:** Tách 4 loss classes từ `train_reid.py`, **fix bug** Temporal Consistency Loss không được dùng, thêm `MultiTaskLoss` wrapper.

#### Những thứ cần copy từ `train_reid.py`:

1. `LabelSmoothCrossEntropy` (lines 40-52) → rename `CrossEntropyLabelSmooth`
2. `HardTripletLoss` (lines 54-89) → **giữ nguyên logic**, đã có L2-normalize
3. `CenterLoss` (lines 91-114) → **sửa default `feat_dim=768`** (thay vì 2816)
4. `TemporalConsistencyLoss` (lines 116-125) → **mở rộng**: nhận list of intermediates thay vì 1 tensor

#### Thiết kế:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLabelSmooth(nn.Module):
    """
    Copy từ train_reid.py LabelSmoothCrossEntropy.
    Label smoothing epsilon = 0.1.
    """
    def __init__(self, num_classes=None, epsilon=0.1):
        super().__init__()
        self.epsilon = epsilon
        self.log_softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, inputs, targets):
        num_classes = inputs.size(1)
        log_probs = self.log_softmax(inputs)
        targets_oh = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets_oh = (1 - self.epsilon) * targets_oh + self.epsilon / num_classes
        loss = (-targets_oh * log_probs).sum(1).mean()
        return loss


class HardTripletLoss(nn.Module):
    """
    Online Hard Mining Triplet Loss.
    Copy từ train_reid.py HardTripletLoss.
    
    ĐÃ CÓ SẴN L2-normalize trong forward.
    Dùng Euclidean distance trên L2-normalized features (tương đương cosine).
    """
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
    
    def forward(self, inputs, targets):
        # L2-normalize (đã có sẵn trong code cũ)
        inputs = F.normalize(inputs, p=2, dim=1)
        n = inputs.size(0)
        
        # Pairwise Euclidean distance
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(inputs, inputs.t(), beta=1, alpha=-2)
        dist = dist.clamp(min=1e-12).sqrt()
        
        # Hard mining
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            if len(dist[i][mask[i] == 0]) > 0:
                dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
            else:
                dist_an.append(torch.tensor([0.0], device=dist.device))
        
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        
        if torch.all(dist_an == 0):
            return torch.tensor(0.0, requires_grad=True, device=dist.device)
        
        y = torch.ones_like(dist_an)
        return self.ranking_loss(dist_an, dist_ap, y)


class CenterLoss(nn.Module):
    """
    Center Loss.
    Copy từ train_reid.py CenterLoss.
    
    THAY ĐỔI: default feat_dim=768 (thay vì 2816).
    """
    def __init__(self, num_classes=1000, feat_dim=768):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
    
    def forward(self, x, labels):
        batch_size = x.size(0)
        distmat = (torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) +
                   torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t())
        distmat.addmm_(x, self.centers.t(), beta=1, alpha=-2)
        
        classes = torch.arange(self.num_classes).long().to(x.device)
        labels_expand = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels_expand.eq(classes.expand(batch_size, self.num_classes))
        
        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size
        return loss


class TemporalConsistencyLoss(nn.Module):
    """
    Temporal Consistency Loss.
    
    THAY ĐỔI từ train_reid.py:
    - Nhận list of intermediate states (1 per Mamba layer)
    - Tính consistency trên LAYER CUỐI CÙNG
    - L_temporal = (1/(T-1)) * Σ ||h_t - h_{t+1}||^2
    
    Phiên bản cũ dùng cosine similarity: 1 - cos_sim.mean()
    Phiên bản mới: MSE giữa adjacent hidden states (smooth transitions).
    """
    def __init__(self, use_cosine=True):
        super().__init__()
        self.use_cosine = use_cosine
    
    def forward(self, intermediates):
        """
        Args:
            intermediates: list of (B, T, D) tensors — 1 per Mamba layer
                          HOẶC single (B, T, D) tensor (backward compat)
        """
        # Lấy hidden states từ layer cuối cùng
        if isinstance(intermediates, list):
            if len(intermediates) == 0:
                return torch.tensor(0.0)
            hidden = intermediates[-1]  # (B, T, D) — last layer
        else:
            hidden = intermediates
        
        if hidden.dim() == 2 or hidden.size(1) < 2:
            return torch.tensor(0.0, device=hidden.device)
        
        if self.use_cosine:
            # Cosine similarity approach (giống code cũ)
            sim = F.cosine_similarity(hidden[:, :-1, :], hidden[:, 1:, :], dim=-1)
            loss = 1.0 - sim.mean()
        else:
            # MSE approach (smoother gradients)
            loss = F.mse_loss(hidden[:, :-1, :], hidden[:, 1:, :])
        
        return loss


class MultiTaskLoss(nn.Module):
    """
    Wrapper kết hợp tất cả losses:
    
    L_total = L_ID + λ1 * L_HardTriplet + λ2 * L_TemporalConsistency + λ3 * L_Center
    
    Default (theo Description_v3):
    - λ1 = 1.0 (triplet)
    - λ2 = 0.1~0.2 (temporal consistency)
    - λ3 = 0.05 (center)
    
    FIX BUG: Code cũ (train_reid.py line 436) KHÔNG bao gồm temporal loss.
    """
    
    def __init__(self, num_classes, feat_dim=768,
                 lambda_triplet=1.0, lambda_temporal=0.15, lambda_center=0.05,
                 margin=0.3, label_smooth=0.1):
        super().__init__()
        self.lambda_triplet = lambda_triplet
        self.lambda_temporal = lambda_temporal
        self.lambda_center = lambda_center
        
        self.ce_loss = CrossEntropyLabelSmooth(epsilon=label_smooth)
        self.triplet_loss = HardTripletLoss(margin=margin)
        self.center_loss = CenterLoss(num_classes=num_classes, feat_dim=feat_dim)
        self.temporal_loss = TemporalConsistencyLoss(use_cosine=True)
    
    def forward(self, logits, raw_features, intermediates, labels):
        """
        Args:
            logits:        (B, num_classes) — from classifier
            raw_features:  (B, feat_dim) — BEFORE BNNeck, cho Triplet + Center
            intermediates: list of (B, T, D) — Mamba hidden states
            labels:        (B,) — identity labels
        
        Returns:
            total_loss: scalar
            loss_dict: dict of individual losses (for logging)
        """
        # L_ID (Cross-Entropy with Label Smoothing)
        loss_id = self.ce_loss(logits, labels)
        
        # L2-normalize raw features cho Triplet + Center
        normed_features = F.normalize(raw_features, p=2, dim=1)
        
        # L_Triplet
        loss_triplet = self.triplet_loss(normed_features, labels)
        
        # L_Center
        loss_center = self.center_loss(normed_features, labels)
        
        # L_Temporal (FIX: đây là loss bị thiếu trong code cũ!)
        loss_temporal = self.temporal_loss(intermediates)
        
        # L_total
        total = (loss_id 
                 + self.lambda_triplet * loss_triplet 
                 + self.lambda_temporal * loss_temporal 
                 + self.lambda_center * loss_center)
        
        return total, {
            'total': total.item(),
            'id': loss_id.item(),
            'triplet': loss_triplet.item(),
            'temporal': loss_temporal.item(),
            'center': loss_center.item()
        }
```

---

### 2.5. `data/dataset.py` — UAV ReID Dataset

**Mục tiêu:** Refactor `UAVReIDDataset` từ `train_reid.py`, thêm **Anchor-First Boundary Alignment** sampling theo Description_v3.

#### Thay đổi chính từ code cũ:

| Feature | Hiện tại (train_reid.py) | V3 mới |
|---------|--------------------------|--------|
| Sampling | `np.linspace` uniform | Anchor-first với stride=3 |
| Frame loading | Sequential `Image.open` | Giữ nguyên + thêm caching option |
| Crop | Đã crop sẵn bởi `data_pipeline.py` | Giữ (preprocessed data) |
| Context margin | 20% (data_pipeline.py) | 15% theo Description_v3 spec |

#### Thiết kế:

```python
import os
import json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class UAVReIDDataset(Dataset):
    """
    UAV ReID Dataset với Anchor-First Boundary Alignment.
    
    Mỗi sample = 1 cặp (before_clip, after_clip) cho Siamese training.
    Mỗi clip = N=16 frames, sampled với stride=3.
    
    Data format: Đọc từ processed data (output của data_pipeline.py).
    JSON format: list of {sequence_id, event_index, identity_id, 
                          before_frames, after_frames}
    """
    
    def __init__(self, data_dir, json_path, transform=None, 
                 num_frames=16, stride=3):
        self.data_dir = data_dir
        self.transform = transform
        self.num_frames = num_frames
        self.stride = stride
        
        # Load annotations
        with open(json_path, 'r') as f:
            self.pairs = json.load(f)
        
        # Filter valid pairs
        self.valid_pairs = [p for p in self.pairs if p.get('identity_id') is not None]
        
        # Build identity mappings
        self.identities = sorted(set(p['identity_id'] for p in self.valid_pairs))
        self.id_to_idx = {pid: i for i, pid in enumerate(self.identities)}
        self.num_identities = len(self.identities)
        
        # Build pid_to_indices for PKSampler
        self._pid_to_indices = {}
        for idx, pair in enumerate(self.valid_pairs):
            pid = self.id_to_idx[pair['identity_id']]
            self._pid_to_indices.setdefault(pid, []).append(idx)
    
    @property
    def pid_to_indices(self):
        return self._pid_to_indices
    
    def __len__(self):
        return len(self.valid_pairs)
    
    def _load_clip_strided(self, folder, frames):
        """
        Load clip với Anchor-First Boundary Alignment.
        
        Logic:
        - Nếu frames đủ (>=num_frames): lấy num_frames frames với stride
        - Nếu frames thiếu: pad bằng frame cuối
        - Anchor = frame cuối (before) hoặc frame đầu (after)
        """
        # Stride-based sampling
        if len(frames) > self.num_frames:
            # Sample với stride
            total_span = (self.num_frames - 1) * self.stride
            if total_span < len(frames):
                # Có đủ frames cho stride sampling
                # Anchor-last: lấy frames[-1] làm anchor
                start = max(0, len(frames) - 1 - total_span)
                indices = list(range(start, len(frames), self.stride))[:self.num_frames]
            else:
                # Không đủ span → uniform sample
                indices = np.linspace(0, len(frames) - 1, self.num_frames).astype(int)
            frames = [frames[i] for i in indices]
        elif len(frames) < self.num_frames:
            # Pad bằng frame cuối
            if len(frames) == 0:
                return torch.zeros((self.num_frames, 3, 256, 256))
            while len(frames) < self.num_frames:
                frames.append(frames[-1])
        
        # Load images
        clip = []
        for fn in frames:
            path = os.path.join(self.data_dir, folder, fn)
            try:
                img = Image.open(path).convert('RGB')
            except Exception:
                img = Image.new('RGB', (256, 256), (0, 0, 0))
            if self.transform:
                img = self.transform(img)
            clip.append(img)
        
        return torch.stack(clip, dim=0)  # (T, C, H, W)
    
    def __getitem__(self, idx):
        pair = self.valid_pairs[idx]
        seq_event = f"{pair['sequence_id']}_event_{pair['event_index']}"
        before_dir = os.path.join(seq_event, "before")
        after_dir = os.path.join(seq_event, "after")
        
        before_clip = self._load_clip_strided(before_dir, pair['before_frames'])
        after_clip = self._load_clip_strided(after_dir, pair['after_frames'])
        
        pid = self.id_to_idx[pair['identity_id']]
        return before_clip, after_clip, pid
```

---

### 2.6. `data/data_miner.py` — Smart Mining & Synthetic Augmentation

**Mục tiêu:** MỚI HOÀN TOÀN. Parse attribute tags, mine hard negatives, synthetic augmentation.

```python
import os
import json
import random
import numpy as np
from PIL import Image, ImageFilter
import cv2


class DataMiner:
    """
    Smart Mining cho UAV ReID.
    
    Chức năng:
    1. Parse attribute tags (OV, FO, SD, SO) từ annotations
    2. Mine hard negative distractors từ SD-tagged sequences
    3. Build distractor pool từ external datasets
    """
    
    def __init__(self, data_dir, annotation_path, external_dirs=None):
        """
        Args:
            data_dir: path to processed data
            annotation_path: path to pairs JSON
            external_dirs: list of paths to external datasets (Anti-UAV318, Anti-UAV600)
        """
        self.data_dir = data_dir
        self.external_dirs = external_dirs or []
        
        with open(annotation_path, 'r') as f:
            self.annotations = json.load(f)
        
        # Parse attribute tags
        self.sd_sequences = []  # Similar Distractor sequences
        self.ov_sequences = []  # Out-of-View sequences
        self.fo_sequences = []  # Full Occlusion sequences
        self.so_sequences = []  # Small Object sequences
        self._parse_attributes()
    
    def _parse_attributes(self):
        """Parse attribute tags từ annotations."""
        for pair in self.annotations:
            attrs = pair.get('attributes', [])
            if isinstance(attrs, str):
                attrs = [a.strip() for a in attrs.split(',')]
            
            if 'SD' in attrs or 'Similar Distractor' in attrs:
                self.sd_sequences.append(pair)
            if 'OV' in attrs or 'Out-of-View' in attrs:
                self.ov_sequences.append(pair)
            if 'FO' in attrs or 'Full Occlusion' in attrs:
                self.fo_sequences.append(pair)
            if 'SO' in attrs or 'Small Object' in attrs:
                self.so_sequences.append(pair)
    
    def build_distractor_pool(self, max_distractors=500):
        """
        Xây dựng global distractor pool.
        
        Sources:
        1. SD-tagged sequences (same video, different target)
        2. External datasets (Anti-UAV318, Anti-UAV600)
        
        Returns: list of {frames_dir, frame_files, source}
        """
        distractors = []
        
        # 1. From SD sequences
        for seq in self.sd_sequences:
            seq_event = f"{seq['sequence_id']}_event_{seq['event_index']}"
            distractors.append({
                'frames_dir': os.path.join(seq_event, "before"),
                'frame_files': seq.get('before_frames', []),
                'identity_id': seq.get('identity_id'),
                'source': 'sd_internal'
            })
        
        # 2. From external datasets
        for ext_dir in self.external_dirs:
            if not os.path.exists(ext_dir):
                continue
            ext_json = os.path.join(ext_dir, 'pairs_train.json')
            if os.path.exists(ext_json):
                with open(ext_json, 'r') as f:
                    ext_pairs = json.load(f)
                for pair in ext_pairs[:max_distractors]:
                    distractors.append({
                        'frames_dir': os.path.join(ext_dir, 
                            f"{pair['sequence_id']}_event_{pair['event_index']}", "before"),
                        'frame_files': pair.get('before_frames', []),
                        'identity_id': -1,  # Unknown ID (distractor)
                        'source': 'external'
                    })
        
        # Shuffle và limit
        random.shuffle(distractors)
        return distractors[:max_distractors]
    
    def mine_hard_negatives_for_pid(self, target_pid, num_negatives=5):
        """
        Tìm hard negatives cho 1 target ID.
        Ưu tiên: SD sequences có cùng video nhưng khác target.
        """
        negatives = []
        for seq in self.sd_sequences:
            if seq.get('identity_id') != target_pid:
                negatives.append(seq)
                if len(negatives) >= num_negatives:
                    break
        return negatives


class SyntheticAugmentor:
    """
    Synthetic Augmentation cho T_before <-> T_after pairs.
    
    Augmentations:
    1. Motion Blur (simulate fast UAV movement)
    2. Scale Jitter (simulate distance changes)
    3. Random Erasing (simulate partial occlusion)
    4. Synthetic Disappearance (progressive occlusion)
    """
    
    def __init__(self, motion_blur_prob=0.3, scale_jitter_prob=0.3,
                 scale_range=(0.7, 1.3), erase_prob=0.5):
        self.motion_blur_prob = motion_blur_prob
        self.scale_jitter_prob = scale_jitter_prob
        self.scale_range = scale_range
        self.erase_prob = erase_prob
    
    def apply_motion_blur(self, img_pil, kernel_size=None):
        """
        Directional motion blur.
        Simulate UAV camera shake / fast movement.
        """
        if random.random() > self.motion_blur_prob:
            return img_pil
        
        img_np = np.array(img_pil)
        if kernel_size is None:
            kernel_size = random.choice([5, 7, 9, 11, 15])
        
        # Random angle
        angle = random.uniform(0, 360)
        angle_rad = np.deg2rad(angle)
        
        # Create motion blur kernel
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        center = kernel_size // 2
        for i in range(kernel_size):
            offset = i - center
            x = int(center + offset * np.cos(angle_rad))
            y = int(center + offset * np.sin(angle_rad))
            if 0 <= x < kernel_size and 0 <= y < kernel_size:
                kernel[y, x] = 1.0
        kernel /= kernel.sum() + 1e-8
        
        blurred = cv2.filter2D(img_np, -1, kernel)
        return Image.fromarray(blurred)
    
    def apply_scale_jitter(self, img_pil, scale_factor=None):
        """
        Random scale jitter.
        Simulate drone approaching/receding.
        """
        if random.random() > self.scale_jitter_prob:
            return img_pil
        
        if scale_factor is None:
            scale_factor = random.uniform(*self.scale_range)
        
        w, h = img_pil.size
        new_w, new_h = int(w * scale_factor), int(h * scale_factor)
        
        # Resize then crop/pad back to original
        img_resized = img_pil.resize((new_w, new_h), Image.BILINEAR)
        
        if scale_factor > 1.0:
            # Crop center
            left = (new_w - w) // 2
            top = (new_h - h) // 2
            img_out = img_resized.crop((left, top, left + w, top + h))
        else:
            # Pad with black
            img_out = Image.new('RGB', (w, h), (0, 0, 0))
            paste_x = (w - new_w) // 2
            paste_y = (h - new_h) // 2
            img_out.paste(img_resized, (paste_x, paste_y))
        
        return img_out
    
    def __call__(self, img_pil):
        """Áp dụng random combination of augmentations."""
        img_pil = self.apply_motion_blur(img_pil)
        img_pil = self.apply_scale_jitter(img_pil)
        return img_pil
```

---

### 2.7. `data/sampler.py` — PK Sampler

**Mục tiêu:** Tách `ReIDBatchSampler` từ `train_reid.py`. Logic giữ nguyên.

```python
# Copy nguyên ReIDBatchSampler từ train_reid.py (lines 181-240)
# Rename thành PKSampler cho clarity
# Thay đổi duy nhất: nhận pid_to_indices dict thay vì access dataset.valid_pairs

class PKSampler(Sampler):
    """
    P×K Identity-Aware Batch Sampler.
    P=8 identities per batch, K=4 instances per identity.
    Batch size = P × K = 32.
    
    Copy từ train_reid.py ReIDBatchSampler, sử dụng dataset.pid_to_indices.
    """
    def __init__(self, dataset, p=8, k=4):
        ...
```

---

### 2.8. `data/transforms.py` — Transforms

**Mục tiêu:** Tách transforms từ `train_reid.py`, thêm `MotionBlur` và `ScaleJitter` như custom transforms.

```python
from torchvision import transforms


class MotionBlurTransform:
    """Custom torchvision-compatible motion blur transform."""
    def __init__(self, kernel_size_range=(5, 15), p=0.3):
        ...
    def __call__(self, img):
        ...


class ScaleJitterTransform:
    """Custom torchvision-compatible scale jitter transform."""
    def __init__(self, scale_range=(0.8, 1.2), p=0.3):
        ...
    def __call__(self, img):
        ...


def get_train_transforms(img_size=256, crop_size=224):
    """
    Training transforms (mở rộng từ train_reid.py lines 314-324).
    
    Thêm: MotionBlurTransform, ScaleJitterTransform
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomRotation(degrees=15),
        transforms.RandomCrop((crop_size, crop_size)),
        transforms.RandomHorizontalFlip(),
        MotionBlurTransform(p=0.3),           # MỚI
        ScaleJitterTransform(p=0.3),          # MỚI
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))
        ], p=0.3),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3))
    ])


def get_test_transforms(img_size=256, crop_size=224):
    """Test transforms (copy từ train_reid.py lines 337-341)."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.CenterCrop((crop_size, crop_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
```

---

### 2.9. `train.py` — Training Loop

**Mục tiêu:** Refactor `train_reid.py`, dùng modules mới, **fix temporal loss bug**.

#### Thay đổi chính so với `train_reid.py`:

| Feature | Hiện tại (`train_reid.py`) | V3 mới (`train.py`) |
|---------|---------------------------|---------------------|
| Model import | `from model import UAVReIDNet` | `from models import UAVMambaReID` |
| Loss import | Inline classes | `from losses import MultiTaskLoss` |
| Dataset import | Inline `UAVReIDDataset` | `from data.dataset import UAVReIDDataset` |
| Sampler import | Inline `ReIDBatchSampler` | `from data.sampler import PKSampler` |
| Transform import | Inline | `from data.transforms import get_train_transforms` |
| Training loop | **Không gọi temporal loss** | **Gọi temporal loss** via `MultiTaskLoss` |
| Model forward | `model(before, after)` → no intermediates | `model(before, after, return_intermediate=True)` → có intermediates |
| Feature dim | 3072 | 768 |
| 2-stage | Stage1 frozen + Stage2 unfreeze | Giữ nguyên logic |

#### Cấu trúc function chính:

```python
def main():
    # 1. Parse config (giữ logic YAML config từ train_reid.py)
    # 2. Setup logging (giữ Logger class)
    # 3. Build dataset + dataloader (dùng modules mới)
    # 4. Build model (UAVMambaReID thay vì UAVReIDNet)
    # 5. Build losses (MultiTaskLoss thay vì inline)
    # 6. Build optimizers (discriminative LR: backbone 1e-5, head 1e-4)
    # 7. Training loop
    
    ...
    
    # CRITICAL FIX: train_epoch phải gọi model với return_intermediate=True
    def train_epoch(epoch, stage_name, optim, lr_scheduler=None):
        model.train()
        for before, after, pids in dataloader:
            with torch.amp.autocast('cuda', ...):
                # V3: Gọi model với return_intermediate=True
                (raw_b, bn_b, logits_b, inter_b), \
                (raw_a, bn_a, logits_a, inter_a) = model(before, after, return_intermediate=True)
                
                # Tính loss cho cả before VÀ after
                loss_b, dict_b = multi_loss(logits_b, raw_b, inter_b, pids)
                loss_a, dict_a = multi_loss(logits_a, raw_a, inter_a, pids)
                loss = loss_b + loss_a
            
            # Gradient step (giữ logic từ train_reid.py)
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optim)
            
            # Center loss optimizer step
            scaler.step(center_optimizer)
            scaler.update()
```

#### Config YAML mới (`configs/config_v3.yaml`):

```yaml
# ========================
# UAV ReID v3 Configuration
# ========================

paths:
  data_dir: '/data/processed'
  gasnet_dir: ''                  # Không dùng GASNet nữa
  gasnet_weights: ''              # Không dùng
  backbone_pretrained: '/path/to/vru_swin_base.pth'
  checkpoint_dir: 'checkpoints_v3'
  log_dir: 'logs_v3'

device:
  gpu_jetson: false

model:
  backbone_arch: 'swin_base_patch4_window7_224'
  feat_dim: 768
  n_mamba_layers: 2

train:
  resume: ''
  batch_size: 32
  num_instances: 4              # K=4
  num_pids_per_batch: 8         # P=8
  num_frames: 16                # N=16
  stride: 3                     # Stride=3
  num_workers: 4
  use_amp: true
  pin_memory: true
  use_compile: false
  val_freq: 5
  
  stage1:
    epochs: 5                   # Warm-up (backbone frozen)
    lr: 1.0e-4                  # Head + Mamba LR
    weight_decay: 5.0e-4
  
  stage2:
    epochs: 55                  # End-to-end (total ~60 epochs)
    lr_backbone: 1.0e-5         # Discriminative LR
    lr_temporal: 1.0e-4
    lr_head: 1.0e-4
  
  loss:
    label_smooth: 0.1
    triplet_margin: 0.3
    lam1: 1.0                  # λ_triplet
    lam2: 0.15                 # λ_temporal (0.1~0.2)
    lam3: 0.05                 # λ_center
    lr_center: 0.5

scheduler:
  type: 'cosine'
  warmup_epochs: 5
  eta_min: 1.0e-7
```

---

### 2.10. `evaluate.py` — Multi-Object Evaluation

**Mục tiêu:** Refactor `evaluate_reid.py`, thêm full 3-scenario evaluation + fix online latency bug.

#### Thay đổi chính:

| Feature | Hiện tại (`evaluate_reid.py`) | V3 mới |
|---------|-------------------------------|--------|
| Model | `UAVReIDNet` | `UAVMambaReID` |
| Metrics | Rank-1, Rank-5, mAP, mINP | Giữ nguyên + thêm Rank-10 |
| 3 Scenarios | Có nhưng chưa rõ logic | Rõ ràng: Short(<5s), Long(>15s), Distractors(SD) |
| Online sim | **BUG: random latency** | **Fix: tính thực tế** |
| Gallery | Per-video | Multi-object global gallery |

#### Cấu trúc functions:

```python
def extract_features(model, dataloader, device):
    """Giữ nguyên logic từ evaluate_reid.py."""
    ...

def compute_distance_matrix(query_features, gallery_features):
    """Cosine similarity. Giữ nguyên."""
    ...

def evaluate_rank(distmat, q_pids, g_pids, max_rank=50):
    """CMC + mAP. Giữ logic từ eval_map_cmc trong evaluate_reid.py."""
    ...

def compute_minp(distmat, q_pids, g_pids):
    """mINP. Giữ nguyên logic."""
    ...

def evaluate_by_scenario(results, annotations):
    """
    MỚI: Evaluate trên 3 scenarios riêng biệt.
    
    Scenario 1 - Short Disappearance (<5s = <150 frames @30fps):
        Filter pairs có OV/FO duration < 150 frames
    
    Scenario 2 - Long Disappearance (>15s = >450 frames @30fps):
        Filter pairs có OV/FO duration > 450 frames
    
    Scenario 3 - Re-appearance with Distractors:
        Filter pairs có SD attribute tag
    """
    ...

def build_multi_object_gallery(model, dataset, video_id, device):
    """
    MỚI: Xây dựng Global Gallery cho multi-object evaluation.
    
    Gallery = [Seq_Tbefore_ID1, ..., Seq_Tbefore_IDM] + Distractor_Pool
    """
    ...


class OnlineReIDSimulator:
    """
    MỚI: 3-State Real-Time Simulation.
    
    FIX BUG: evaluate_reid.py dùng random latency.
    V3 tính thực tế bằng frame counting.
    
    State 1: Tracking (update Mamba temporal token mỗi stride=3 frames)
    State 2: Lost (freeze temporal token khi confidence < threshold)
    State 3: ReID (batch extract candidates, cosine match vs locked token)
    """
    
    def __init__(self, model, device, threshold_track=0.5, threshold_reid=0.7, stride=3):
        self.model = model
        self.device = device
        self.threshold_track = threshold_track
        self.threshold_reid = threshold_reid
        self.stride = stride
        self.state = 'tracking'
        self.locked_token = None
    
    def simulate(self, video_data, annotations):
        """
        Simulate video stream.
        
        Returns:
            latency_frames: int — frames từ reappearance đến confirmed match
            latency_ms: float — tính từ latency_frames / fps
            false_alarm_rate: float — false positive rate
            events: list of {frame, state, action, confidence}
        """
        events = []
        latencies = []
        false_alarms = 0
        total_detections = 0
        
        for frame_idx, frame_data in enumerate(video_data):
            if self.state == 'tracking':
                # Update temporal token mỗi stride frames
                if frame_idx % self.stride == 0:
                    # Feed crop vào Mamba engine
                    ...
                
                # Check if target lost
                if frame_data.get('confidence', 1.0) < self.threshold_track:
                    self.state = 'lost'
                    self.locked_token = current_token.clone()
                    events.append({'frame': frame_idx, 'state': 'lost'})
            
            elif self.state == 'lost':
                # Chờ detector tìm candidates
                candidates = frame_data.get('candidates', [])
                if len(candidates) > 0:
                    self.state = 'reid_verify'
            
            elif self.state == 'reid_verify':
                # Batch extract candidate features
                candidates = frame_data.get('candidates', [])
                if len(candidates) > 0:
                    total_detections += 1
                    # Stack candidates → batch GPU extraction
                    # Compute cosine similarity vs locked_token
                    # If max(scores) > threshold_reid → confirmed match
                    best_score = ...
                    if best_score > self.threshold_reid:
                        latency = frame_idx - lost_frame
                        latencies.append(latency)
                        self.state = 'tracking'
                        events.append({'frame': frame_idx, 'state': 'reid_confirmed',
                                      'latency_frames': latency})
                    elif best_score < 0.3:
                        false_alarms += 1
        
        return {
            'avg_latency_frames': np.mean(latencies) if latencies else -1,
            'avg_latency_ms': np.mean(latencies) / 30.0 * 1000 if latencies else -1,
            'false_alarm_rate': false_alarms / max(1, total_detections),
            'events': events
        }
```

---

## 3. THỨ TỰ THỰC HIỆN

### Phase 1: Foundation (Models + Losses)

| # | File | Mô tả | Dependencies |
|---|------|--------|--------------|
| 1.1 | `configs/config_v3.yaml` | Config mới | Không |
| 1.2 | `models/__init__.py` | Package init | Không |
| 1.3 | `models/backbone.py` | Swin backbone | `timm` |
| 1.4 | `models/mamba_engine.py` | Mamba engine + SimpleS6Block fallback | `mamba_ssm` (optional) |
| 1.5 | `models/reid_model.py` | Main model | 1.3, 1.4 |
| 1.6 | `losses.py` | Tất cả losses + MultiTaskLoss | Không |

**Test Phase 1:**
```python
# Verify shapes
from models import UAVMambaReID
model = UAVMambaReID(num_classes=100)
x = torch.randn(2, 16, 3, 224, 224)

# Training mode
model.train()
(raw_b, bn_b, logits_b, inter_b) = model(x, return_intermediate=True)
assert raw_b.shape == (2, 768)
assert logits_b.shape == (2, 100)
assert len(inter_b) == 2  # 2 mamba layers
assert inter_b[0].shape == (2, 16, 768)

# Eval mode  
model.eval()
with torch.no_grad():
    feat = model(x)
    assert feat.shape == (2, 768)
```

### Phase 2: Data Pipeline

| # | File | Mô tả | Dependencies |
|---|------|--------|--------------|
| 2.1 | `data/__init__.py` | Package init | Không |
| 2.2 | `data/transforms.py` | Transforms + MotionBlur, ScaleJitter | `opencv-python` |
| 2.3 | `data/dataset.py` | UAVReIDDataset | 2.2 |
| 2.4 | `data/sampler.py` | PKSampler | 2.3 |
| 2.5 | `data/data_miner.py` | Smart Mining + Synthetic Aug | 2.3 |

### Phase 3: Training & Evaluation

| # | File | Mô tả | Dependencies |
|---|------|--------|--------------|
| 3.1 | `train.py` | Training loop | Phase 1, Phase 2 |
| 3.2 | `evaluate.py` | Evaluation + Online sim | Phase 1, Phase 2 |

### Phase 4: Infrastructure

| # | File | Mô tả | Dependencies |
|---|------|--------|--------------|
| 4.1 | `requirements.txt` | Update deps | Không |
| 4.2 | `Dockerfile` | Update | 4.1 |
| 4.3 | `entrypoint.sh` | Update script names | 4.2 |

---

## 4. LƯU Ý QUAN TRỌNG CHO GEMINI

### 4.1. Feature Dimension — CRITICAL
- **TOÀN BỘ pipeline dùng `feat_dim=768`**: backbone output, mamba d_model, BNNeck, center loss, triplet features
- Swin-Base output 1024 → project về 768
- **KHÔNG** concat visual + temporal như code cũ (2560 + 512 = 3072)
- Temporal pooling output `(B, 768)` trực tiếp feed vào ReIDHead

### 4.2. Fix Bug Temporal Loss — CRITICAL
- **Code cũ `train_reid.py` line 436**: `loss = loss_id + lam1 * loss_tri + lam3 * loss_center`
- **THIẾU**: `+ lam2 * loss_temporal`
- `TemporalConsistencyLoss` được define nhưng **KHÔNG BAO GIỜ được gọi**
- V3 phải gọi nó qua `MultiTaskLoss.forward()` → `self.temporal_loss(intermediates)`

### 4.3. Unidirectional vs Bidirectional Mamba
- Code cũ (`model.py` line 161-163): `x_fwd + x_bwd + res` (bidirectional)
- Description_v3: "1D Unidirectional Scanning"
- **V3 phải dùng unidirectional**: chỉ forward scan, bỏ `x_bwd`

### 4.4. Giữ Backward Compatibility
- **KHÔNG XÓA** `model.py`, `train_reid.py`, `evaluate_reid.py`
- Checkpoint cũ dùng `UAVReIDNet` với dims 3072 → vẫn load được qua file cũ
- Checkpoint mới dùng `UAVMambaReID` với dims 768
- Entry point mới: `python train.py --config configs/config_v3.yaml`

### 4.5. Siamese Training
- Giữ nguyên Siamese training: model nhận `(before_clips, after_clips)`
- Loss tính riêng cho before và after, rồi cộng lại
- PK Sampler đảm bảo mỗi batch có đủ IDs cho hard mining

### 4.6. Mixed Precision
- Dùng `torch.amp.autocast('cuda')` (API mới, bỏ deprecated import)
- SimpleS6Block PHẢI chuyển sang float32 cho selective scan (giữ logic từ code cũ line 88-92)
- KHÔNG autocast khi evaluate

### 4.7. VRAM Estimation
- Swin-Base: ~88M params
- Mamba 2 layers (768-d): ~6M params
- Per-frame: (B*T, 3, 224, 224) → Swin → (B*T, 768)
- Batch 32 × 16 frames = 512 forward passes → ~25-35GB VRAM
- H100 80GB: OK
- Nếu thiếu VRAM: gradient checkpointing trên backbone

### 4.8. Dependencies cần thêm
```
timm>=0.9.0                    # Swin-Transformer
opencv-python>=4.8.0           # MotionBlur augmentation
scipy>=1.11.0                  # Distance computations (optional)
```

---

## 5. CHECKLIST KHI CODE XONG

- [ ] `models/backbone.py` tạo VRUBackbone với Swin, output 768-d
- [ ] `models/mamba_engine.py` unidirectional, return intermediates
- [ ] `models/reid_model.py` UAVMambaReID end-to-end, Siamese support
- [ ] `losses.py` tất cả 4 losses + MultiTaskLoss wrapper
- [ ] **Temporal loss ĐƯỢC GỌI** trong training loop (fix bug)
- [ ] `data/dataset.py` anchor-first strided sampling
- [ ] `data/data_miner.py` smart mining + synthetic augmentation
- [ ] `data/sampler.py` PKSampler (P=8, K=4)
- [ ] `data/transforms.py` MotionBlur + ScaleJitter
- [ ] `train.py` 2-stage training, discriminative LR
- [ ] `evaluate.py` 3 scenarios + OnlineReIDSimulator (fix random latency bug)
- [ ] `configs/config_v3.yaml` đầy đủ config
- [ ] Unit tests verify tensor shapes
- [ ] Backward compatibility: old files giữ nguyên
