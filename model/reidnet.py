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
    from train import GASNet, infer_backbone_name, infer_econvnext
    from bcgg import infer_gate_config
    HAS_GASNET = True
except ImportError:
    HAS_GASNET = False
    infer_gate_config = None
    infer_backbone_name = None
    infer_econvnext = None
    print("Warning: Không thể import GASNet từ train.py. Sẽ sử dụng Dummy Network.")


class UAVReIDNet(nn.Module):
    def __init__(self, gasnet_weights_path=None, num_identities=1000, freeze_backbone=True,
                 backbone='resnet50_ibn', temporal_pool='attn', temporal_pe=True,
                 temporal_type='mamba', econvnext=False, gate=None,
                 gate_hidden=None, gate_groups=None, gate_kernel=None,
                 gate_channel_kernel=None, gate_inject_value=None):
        super().__init__()
        
        self.temporal_type = (temporal_type or 'mamba').lower()
        
        # Tự động trỏ path mặc định nếu không truyền
        if gasnet_weights_path is None:
            gasnet_weights_path = os.environ.get('GASNET_WEIGHTS',
                os.path.abspath(os.path.join(os.path.dirname(__file__), '../UAV/gasnet_project/test/gasnet.best.pth')))
            
        # 1. Visual Backbone
        if HAS_GASNET:
            state_dict = None
            has_gasnet_ckpt = bool(gasnet_weights_path and os.path.exists(gasnet_weights_path))
            if has_gasnet_ckpt:
                raw_ckpt = torch.load(gasnet_weights_path, map_location='cpu')
                if isinstance(raw_ckpt, dict) and 'model_state_dict' in raw_ckpt:
                    raw_ckpt = raw_ckpt['model_state_dict']
                # torch.compile lưu state_dict với prefix _orig_mod. — phải strip trước khi load
                state_dict = {
                    (k[len('_orig_mod.'):] if k.startswith('_orig_mod.') else k): v
                    for k, v in raw_ckpt.items()
                }

            inferred_gate = (
                infer_gate_config(state_dict or {}, default_kind=(gate or "rga").lower())
                if infer_gate_config is not None
                else {
                    "gate": (gate or "rga").lower(),
                    "gate_hidden": 16,
                    "gate_groups": 4,
                    "gate_kernel": 7,
                    "gate_channel_kernel": 5,
                    "gate_inject_value": False,
                }
            )
            effective_gate = (gate or inferred_gate["gate"] or "rga").lower()
            if state_dict is not None and gate is not None and inferred_gate["gate"] != effective_gate:
                print(
                    f" ⚠️ Config đặt gate={effective_gate!r} nhưng gasnet_weights ({os.path.basename(gasnet_weights_path)}) "
                    f"được train với gate={inferred_gate['gate']!r}. Các tầng ga1..ga4 sẽ khởi tạo mới theo {effective_gate!r}."
                )

            eff_hidden = int(gate_hidden if gate_hidden is not None else inferred_gate["gate_hidden"])
            eff_groups = int(gate_groups if gate_groups is not None else inferred_gate["gate_groups"])
            eff_kernel = int(gate_kernel if gate_kernel is not None else inferred_gate["gate_kernel"])
            eff_ch_kernel = int(
                gate_channel_kernel if gate_channel_kernel is not None else inferred_gate["gate_channel_kernel"]
            )
            eff_inject = bool(
                gate_inject_value if gate_inject_value is not None else inferred_gate["gate_inject_value"]
            )
            self.gate = effective_gate

            print(
                f"  GASNet stage gate: {effective_gate.upper()}"
                + (
                    f" (d={eff_hidden}, G={eff_groups}, k={eff_kernel}, kc={eff_ch_kernel}, value={eff_inject})"
                    if effective_gate == "bcgg"
                    else ""
                )
            )
            try:
                self.backbone = GASNet(
                    num_classes=num_identities,
                    use_pretrained=(not has_gasnet_ckpt),
                    backbone=backbone,
                    use_gem=True,
                    econvnext=econvnext,
                    gate=effective_gate,
                    gate_hidden=eff_hidden,
                    gate_groups=eff_groups,
                    gate_kernel=eff_kernel,
                    gate_channel_kernel=eff_ch_kernel,
                    gate_inject_value=eff_inject,
                )
            except Exception as e:
                if not has_gasnet_ckpt:
                    print(f" ⚠️ Không tải được pretrained backbone online ({e}); khởi tạo kiến trúc với use_pretrained=False.")
                    self.backbone = GASNet(
                        num_classes=num_identities,
                        use_pretrained=False,
                        backbone=backbone,
                        use_gem=True,
                        econvnext=econvnext,
                        gate=effective_gate,
                        gate_hidden=eff_hidden,
                        gate_groups=eff_groups,
                        gate_kernel=eff_kernel,
                        gate_channel_kernel=eff_ch_kernel,
                        gate_inject_value=eff_inject,
                    )
                else:
                    raise

            if state_dict is not None:
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
            self.gate = (gate or 'rga').lower()
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

        is_no_temporal = self.temporal_type in ('none', 'identity', 'visual_only')
        temporal_d_out = 0 if is_no_temporal else 512
        
        # Chọn temporal encoder qua registry (model/registry.py).
        # Thêm encoder mới: tạo file trong model/, gắn @register_temporal, rồi đặt
        # `train.temporal_type` trong config bằng tên đã đăng ký.
        self.temporal_encoder = build_temporal_encoder(
            self.temporal_type, d_in=visual_dim, d_model=temporal_d_out, d_out=temporal_d_out, max_seq_len=64,
            num_layers=2, num_heads=8, dropout=0.1, pool=temporal_pool, use_pe=temporal_pe,
        )
        print(f"  Temporal encoder: {type(self.temporal_encoder).__name__}"
              + (" (Visual-Only / Không dùng nhánh thời gian)" if is_no_temporal else
                 ("" if self.temporal_type != 'mamba' else
                  f" ({'mamba_ssm' if HAS_MAMBA else 'SimpleS6Block fallback'})")))
        
        # 3. ReID Head
        self.visual_dim = visual_dim
        self.temporal_d_out = temporal_d_out
        self.feat_dim = visual_dim + temporal_d_out
        self.head = ReIDHead(in_dim=self.feat_dim, num_identities=num_identities)
        
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
