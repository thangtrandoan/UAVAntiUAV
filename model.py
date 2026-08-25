import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ưu tiên: ENV > relative path > hardcode fallback
gasnet_path = os.environ.get('GASNET_PATH', 
    os.path.abspath(os.path.join(os.path.dirname(__file__), 'gasnet')))
if gasnet_path not in sys.path:
    sys.path.append(gasnet_path)

try:
    from train import GASNet
    HAS_GASNET = True
except ImportError:
    HAS_GASNET = False
    print("Warning: Không thể import GASNet từ train.py. Sẽ sử dụng Dummy Network.")

# Thử import mamba_ssm (tối ưu cuda), fallback về tự implement
try:
    from mamba_ssm import Mamba
    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False
    print("Warning: Không tìm thấy mamba_ssm. Sử dụng Simplified S6 Block (fallback).")


class SimpleS6Block(nn.Module):
    """
    Simplified S6 (Selective State Space) Block fallback khi không cài được mamba_ssm.
    Phù hợp cho Jetson AGX Orin, hỗ trợ torch.compile.
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        
        # 1. Linear expansion
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        
        # 2. 1D Depthwise Conv
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            bias=True
        )
        self.act = nn.SiLU()
        
        # 3. Discretization parameters
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)
        
        # A matrix (đường chéo - log space)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(self.d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # 6. Linear projection output
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        
    def forward(self, x):
        # x: [B, L, D]
        B, L, D = x.shape
        xz = self.in_proj(x)
        x_proj, z_gate = xz.chunk(2, dim=-1) # [B, L, d_inner]
        
        # Depthwise Conv1d (hỗ trợ channel-last memory layout bằng transpose)
        x_conv = x_proj.transpose(1, 2) # [B, d_inner, L]
        x_conv = self.conv1d(x_conv)[:, :, :L]
        x_conv = x_conv.transpose(1, 2) # [B, L, d_inner]
        x_conv = self.act(x_conv)
        
        # Tính toán SSM parameters
        x_dbl = self.x_proj(x_conv) # [B, L, d_state*2 + 1]
        dt, B_mat, C_mat = torch.split(x_dbl, [1, self.d_state, self.d_state], dim=-1)
        
        dt = F.softplus(self.dt_proj(dt)) # [B, L, d_inner]
        
        A = -torch.exp(self.A_log.float()) # [d_inner, d_state]
        
        # 4. Selective Scan (Vectorized Parallel Scan)
        # Thay vì viết C++, ta sử dụng thủ thuật Toán học (Parallel Cumsum) 
        # để biến vòng lặp tuần tự thành tính toán ma trận song song 100% trên GPU.
        dt = dt.float()
        B_mat = B_mat.float()
        C_mat = C_mat.float()
        x_conv_f32 = x_conv.float()
        
        # W = dt * A
        W = dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0) # [B, L, d_inner, d_state]
        
        # V = (dt * B) * x
        dB = dt.unsqueeze(-1) * B_mat.unsqueeze(2) # [B, L, d_inner, d_state]
        V = dB * x_conv_f32.unsqueeze(-1) # [B, L, d_inner, d_state]
        
        # Giải quyết NaN bằng ma trận (Attention-like formulation)
        # Thay vì exp(-P) gây tràn số dương, ta tính trực tiếp P_i - P_j <= 0
        W_t = W.permute(0, 2, 3, 1) # [B, d_inner, d_state, L]
        V_t = V.permute(0, 2, 3, 1) # [B, d_inner, d_state, L]
        
        P = torch.cumsum(W_t, dim=-1) # [B, d_inner, d_state, L]
        
        # Tạo ma trận mask tam giác dưới (j <= i)
        mask = torch.tril(torch.ones(L, L, device=x.device)).view(1, 1, 1, L, L)
        
        # Tính M_{i,j} = P_i - P_j
        # P.unsqueeze(-1) shape: [..., L, 1] (ứng với i)
        # P.unsqueeze(-2) shape: [..., 1, L] (ứng với j)
        M = P.unsqueeze(-1) - P.unsqueeze(-2)
        
        # Dùng masked_fill để tránh tính exp(số dương) gây inf, sau đó inf * 0 = nan
        M = M.masked_fill(mask == 0, float('-inf'))
        weights = torch.exp(M) # Các vị trí j > i sẽ thành exp(-inf) = 0
        
        # Tính h_i = sum_j weights_{i,j} V_j
        h_t = (weights * V_t.unsqueeze(-2)).sum(dim=-1) # [B, d_inner, d_state, L]
        
        h = h_t.permute(0, 3, 1, 2) # Trả về [B, L, d_inner, d_state]
        
        # Output y_i = (h_i * C_i)
        y = (h * C_mat.unsqueeze(2)).sum(dim=-1) # [B, L, d_inner]
        
        y = y.to(x.dtype) # [B, L, d_inner]
        y = y + x_conv * self.D
        
        # 5. Output gating
        y = y * self.act(z_gate)
        
        # Linear projection
        out = self.out_proj(y)
        return out


class TemporalMambaEncoder(nn.Module):
    """
    Temporal Memory Engine: Xử lý chuỗi frame N chiều thời gian
    """
    def __init__(self, d_in=2560, d_model=512, d_out=512, max_seq_len=64, num_layers=2):
        super().__init__()
        self.d_model = d_model
        
        # Linear projection
        self.in_proj = nn.Linear(d_in, d_model)
        
        # Positional Encoding (Learnable)
        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, d_model))
        
        # Mamba Blocks
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            if HAS_MAMBA:
                self.layers.append(Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2))
            else:
                self.layers.append(SimpleS6Block(d_model=d_model))
                
        self.norm_layers = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        
        # MLP Head
        self.out_mlp = nn.Sequential(
            nn.Linear(d_model, d_out),
            nn.BatchNorm1d(d_out),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        # x: [B, N, d_in]
        B, N, _ = x.shape
        x = self.in_proj(x)
        
        # Thêm positional encoding
        x = x + self.pos_embed[:, :N, :]
        
        for mamba_layer, norm in zip(self.layers, self.norm_layers):
            res = x
            # Bidirectional processing
            x_fwd = mamba_layer(x)
            x_bwd = mamba_layer(x.flip(dims=[1])).flip(dims=[1])
            x = norm(x_fwd + x_bwd + res)
            
        # Mean pooling thay vì chỉ lấy frame cuối cùng
        x = x.mean(dim=1) # [B, d_model]
        
        # MLP Head
        x = self.out_mlp(x) # [B, d_out]
        return x


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


class ReIDHead(nn.Module):
    """
    Classifer cho UAV ReID.
    Input = Visual (2560) + Temporal (512) = 3072
    """
    def __init__(self, in_dim=3072, num_identities=1000):
        super().__init__()
        self.bnneck = nn.BatchNorm1d(in_dim)
        self.bnneck.bias.requires_grad_(False)  # no shift
        self.bnneck.apply(weights_init_kaiming)
        
        self.classifier = nn.Linear(in_dim, num_identities, bias=False)
        self.classifier.apply(weights_init_classifier)
        
    def forward(self, visual_feat, temporal_token):
        feat = torch.cat([visual_feat, temporal_token], dim=-1) # [B, 2816]
        bn_feat = self.bnneck(feat)
        
        # Eval mode: return normalization feature cho matching
        if not self.training:
            return bn_feat
            
        # Train mode
        logits = self.classifier(bn_feat)
        return feat, bn_feat, logits


class UAVReIDNet(nn.Module):
    def __init__(self, gasnet_weights_path=None, num_identities=1000, freeze_backbone=True, backbone='resnet50_ibn'):
        super().__init__()
        
        # Tự động trỏ path mặc định nếu không truyền
        if gasnet_weights_path is None:
            gasnet_weights_path = os.environ.get('GASNET_WEIGHTS',
                os.path.abspath(os.path.join(os.path.dirname(__file__), '../UAV/gasnet_project/test/gasnet.best.pth')))
            
        # 1. Visual Backbone
        if HAS_GASNET:
            self.backbone = GASNet(num_classes=num_identities, backbone=backbone, use_gem=True)
            if os.path.exists(gasnet_weights_path):
                state_dict = torch.load(gasnet_weights_path, map_location='cpu')
                self.backbone.load_state_dict(state_dict, strict=False)
                print(f" Đã load GASNet weights từ {gasnet_weights_path}")
            else:
                print(f" Cảnh báo: Không tìm thấy pre-trained weights tại {gasnet_weights_path}")
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
            
        # Freeze backbone ban đầu (train Mamba head)
        if freeze_backbone:
            self.freeze_backbone()
            
        # 2. Temporal Memory Engine
        # Kích thước vector đầu ra của GASNet phụ thuộc vào Backbone
        if backbone == "convnext_small":
            visual_dim = 960  # 768 (Global) + 192 (FS)
        else:
            visual_dim = 2560 # 2048 (Global) + 512 (FS)
            
        self.temporal_encoder = TemporalMambaEncoder(
            d_in=visual_dim, d_model=512, d_out=512, max_seq_len=64, num_layers=2
        )
        
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
        temporal_token = self.temporal_encoder(feats) # [B, 256]
        
        return visual_feat, temporal_token
        
    def forward(self, before_clips, after_clips=None, backbone_only=False):
        """
        Flow Forward của UAVReIDNet
        """
        # Inference mode (Matching/Tracking)
        if not self.training:
            v_feat, t_token = self.extract_features(before_clips)
            if backbone_only:
                return v_feat
            return self.head(v_feat, t_token) # Return bn_feat [B, 2816]
            
        # Training mode
        v_feat_before, t_token_before = self.extract_features(before_clips)
        feat_b, bn_feat_b, logits_b = self.head(v_feat_before, t_token_before)
        
        # Nếu có provide both clips cho việc train pair (Siamese training)
        if after_clips is not None:
            v_feat_after, t_token_after = self.extract_features(after_clips)
            feat_a, bn_feat_a, logits_a = self.head(v_feat_after, t_token_after)
            
            # Trả về features, bn_feats, logits phục vụ tính toán Loss (ID Loss + Triplet Loss)
            return (feat_b, bn_feat_b, logits_b), (feat_a, bn_feat_a, logits_a)
            
        return feat_b, bn_feat_b, logits_b
