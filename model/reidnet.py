"""UAVReIDNet: backbone thị giác + temporal encoder + ReID head."""

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from .components import ReIDHead, weights_init_kaiming, weights_init_classifier
from .registry import build_temporal_encoder
from .temporal_mamba import HAS_MAMBA

# Ưu tiên: ENV > relative path > hardcode fallback
gasnet_path = os.environ.get('GASNET_PATH',
    os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'gasnet')))
if gasnet_path not in sys.path:
    sys.path.append(gasnet_path)

try:
    from train import GASNet
    HAS_GASNET = True
except ImportError:
    HAS_GASNET = False
    print("Warning: Không thể import GASNet từ train.py. Sẽ sử dụng Dummy Network.")


class UAVReIDNet(nn.Module):
    def __init__(self, gasnet_weights_path=None, num_identities=1000, freeze_backbone=True,
                 backbone='resnet50_ibn', temporal_pool='attn', temporal_pe=True,
                 temporal_type='mamba', econvnext=False):
        super().__init__()
        
        self.temporal_type = temporal_type
        
        # Tự động trỏ path mặc định nếu không truyền
        if gasnet_weights_path is None:
            gasnet_weights_path = os.environ.get('GASNET_WEIGHTS',
                os.path.abspath(os.path.join(os.path.dirname(__file__), '../UAV/gasnet_project/test/gasnet.best.pth')))
            
        # 1. Visual Backbone
        if HAS_GASNET:
            self.backbone = GASNet(num_classes=num_identities, backbone=backbone, use_gem=True,
                                   econvnext=econvnext)
            if os.path.exists(gasnet_weights_path):
                state_dict = torch.load(gasnet_weights_path, map_location='cpu')
                # torch.compile lưu state_dict với prefix _orig_mod. — phải strip trước khi load
                state_dict = {
                    (k[len('_orig_mod.'):] if k.startswith('_orig_mod.') else k): v
                    for k, v in state_dict.items()
                }
                # load_state_dict(strict=False) VẪN throw RuntimeError nếu shape mismatch
                # (vd: classifier_global của VRU có num_classes khác num_identities UAV)
                # lọc trước chỉ giữ key có shape khớp chính xác.
                model_state = self.backbone.state_dict()
                filtered = {}
                for k, v in state_dict.items():
                    if k in model_state and tuple(model_state[k].shape) == tuple(v.shape):
                        filtered[k] = v
                missing = [k for k in model_state if k not in filtered]
                unexpected = [k for k in state_dict if k not in filtered]
                self.backbone.load_state_dict(filtered, strict=False)
                print(f" Đã load GASNet weights từ {gasnet_weights_path}")
                if missing:
                    print(f" ⚠️ {len(missing)} keys KHÔNG tìm thấy / shape không khớp (giữ random init):")
                    for mk in missing[:10]:
                        print(f"    - {mk}")
                if unexpected:
                    print(f" ⚠️ {len(unexpected)} keys trong checkpoint KHÔNG dùng được (shape/name không khớp):")
                    for uk in unexpected[:10]:
                        print(f"    - {uk}")
                # Cảnh báo nếu backbone convnext không được load (nguyên nhân chính gây kết quả tệ)
                if any('convnext_backbone' in m for m in missing):
                    print(" ❌ CẢNH BÁO NGHIÊM TRỌNG: convnext_backbone KHÔNG được load từ gasnet weights!")
                    print("    Kiểm tra: 1) file gasnet_weights có phải train với backbone dinov3_convnext?")
                    print("              2) config paths.gasnet_weights trỏ đúng file .pth?")
            else:
                print(f" Cảnh báo: Không tìm thấy pre-trained weights tại {gasnet_weights_path}")
            self.backbone.return_raw_features_eval = True
        else:
            # Dummy cho mục đích debugging nếu mất mã nguồn GASNet
            class DummyGASNet(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.pool = nn.AdaptiveAvgPool2d(1)
                    self.fc = nn.Linear(3, 2560)
                def forward(self, x):
                    x = self.pool(x).view(x.size(0), -1)
                    return self.fc(x)
            self.backbone = DummyGASNet()
            
        # Freeze backbone ban đầu (train temporal head)
        if freeze_backbone:
            self.freeze_backbone()
            
        # 2. Temporal Memory Engine
        # Kích thước vector đầu ra của GASNet phụ thuộc vào Backbone
        if backbone == "dinov3_convnext":
            visual_dim = 960  # 768 (Global) + 192 (FS)
        else:
            visual_dim = 2560 # 2048 (Global) + 512 (FS)
        
        # Chọn temporal encoder qua registry (model/registry.py).
        # Thêm encoder mới: tạo file trong model/, gắn @register_temporal, rồi đặt
        # `train.temporal_type` trong config bằng tên đã đăng ký.
        self.temporal_encoder = build_temporal_encoder(
            temporal_type, d_in=visual_dim, d_model=512, d_out=512, max_seq_len=64,
            num_layers=2, num_heads=8, dropout=0.1, pool=temporal_pool, use_pe=temporal_pe,
        )
        print(f"  Temporal encoder: {type(self.temporal_encoder).__name__}"
              + ("" if temporal_type != 'mamba' else
                 f" ({'mamba_ssm' if HAS_MAMBA else 'SimpleS6Block fallback'})"))
        
        # 3. ReID Head
        self.head = ReIDHead(in_dim=visual_dim + 512, num_identities=num_identities)
        
    def freeze_backbone(self):
        """Đóng băng trọng số của Visual Backbone."""
        print("  Freezing Visual Backbone...")
        for param in self.backbone.parameters():
            param.requires_grad = False
            
    def unfreeze_backbone(self):
        """Mở băng trọng số của Visual Backbone (để end-to-end fine-tuning)."""
        print(" Unfreezing Visual Backbone...")
        for param in self.backbone.parameters():
            param.requires_grad = True
            
    def extract_features(self, clips):
        # clips: [B, N, C, H, W]
        B, N, C, H, W = clips.shape
        
        # Chuyển batch format sang memory_format channels_last để tận dụng TensorCores (Jetson) & AMP
        clips = clips.view(B * N, C, H, W).contiguous(memory_format=torch.channels_last)
        
        # Switch mode dựa vào tình trạng frozen để BN layer xử lý cho đúng
        if not next(self.backbone.parameters()).requires_grad:
            self.backbone.eval()
            
        # Tiết kiệm memory bằng cách không track gradient nếu đang freeze
        grad_enabled = next(self.backbone.parameters()).requires_grad and self.training
        with torch.set_grad_enabled(grad_enabled):
            feats = self.backbone(clips) 
            
        # GASNet trả về tuple ((global, fs), (logits)) lúc train, hoặc (bn_global, bn_fs) lúc eval
        if isinstance(feats, tuple):
            if isinstance(feats[0], tuple):
                global_feat = feats[0][0] # 2048
                fs_feat = feats[0][1]     # 512
            else:
                global_feat = feats[0]
                fs_feat = feats[1]
            feats = torch.cat([global_feat, fs_feat], dim=-1)
            
        feats = feats.view(B, N, -1) # [B, N, 2560]
        
        # Visual Feature: Sử dụng trung bình pooling theo thời gian để đại diện ngoại hình chung của drone
        visual_feat = feats.mean(dim=1) # [B, 2560]
        
        # Temporal Token: Qua Mamba Block
        temporal_token, temporal_seq = self.temporal_encoder(feats) # temporal_token [B, 256], temporal_seq [B, N, d_model]
        
        return visual_feat, temporal_token, temporal_seq
        
    def forward(self, before_clips, after_clips=None, backbone_only=False):
        """
        Flow Forward của UAVReIDNet
        """
        # Inference mode (Matching/Tracking)
        if not self.training:
            v_feat, t_token, _ = self.extract_features(before_clips)
            if backbone_only:
                return v_feat
            return self.head(v_feat, t_token) # Return bn_feat [B, 2816]
            
        # Training mode
        v_feat_before, t_token_before, t_seq_before = self.extract_features(before_clips)
        feat_b, bn_feat_b, logits_b = self.head(v_feat_before, t_token_before)
        
        # Nếu có provide both clips cho việc train pair (Siamese training)
        if after_clips is not None:
            v_feat_after, t_token_after, t_seq_after = self.extract_features(after_clips)
            feat_a, bn_feat_a, logits_a = self.head(v_feat_after, t_token_after)
            
            # Trả về features, bn_feats, logits phục vụ tính toán Loss (ID Loss + Triplet Loss)
            return (feat_b, bn_feat_b, logits_b), (feat_a, bn_feat_a, logits_a), (t_seq_before, t_seq_after)
            
        return feat_b, bn_feat_b, logits_b
