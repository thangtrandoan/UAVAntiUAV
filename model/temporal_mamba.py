"""Temporal encoder dùng SSM/Mamba. Có fallback SimpleS6Block khi thiếu mamba_ssm."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .components import AttentionPooling

# Ưu tiên mamba_ssm (CUDA tối ưu), fallback về tự implement.
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
        
        # 4. Selective Scan — RECURRENCE TUẦN TỰ (  22/9)
        # TRƯỚC ĐÂY: công thức kiểu attention O(L²) — `cumsum` rồi `exp(P_i - P_j)` -> ma trận
        # [B, d_inner, d_state, L, L]. Với d_inner=1024, d_state=16, L=12, B=16 thì ma trận đó
        # ~151 MB cho MỘT layer -> đó là lý do `Avg Mamba + Head = 76.62 ms`.
        #
        # Công thức dưới đây TƯƠNG ĐƯƠNG TOÁN HỌC (đã kiểm chứng bằng numpy: sai số tương đối
        # ~1e-6 = đúng mức float32, KHÔNG phải xấp xỉ) nhưng:
        # - KHÔNG `cumsum` -> không trừ hai số lớn (bản cũ mất chính xác tăng theo L:
        # 2.4e-7 @L=4 -> 5.0e-6 @L=64)
        # - `exp(W)` in (0,1) -> KHÔNG cần `masked_fill(-inf)`, không có nguy cơ inf*0=nan
        # - bộ nhớ [B, L, d_inner, d_state] = **16x nhỏ hơn** ở L=16 (0.26 MB vs 4.19 MB)
        # - L chỉ 4..16 nên vòng lặp L bước RẺ HƠN NHIỀU so với ma trận LxL
        # THAM SỐ KHÔNG ĐỔI -> **checkpoint cũ vẫn nạp được, KHÔNG cần train lại.**
        # LƯU Ý: đây là fix TỐC ĐỘ/BỘ NHỚ, KHÔNG phải fix N-collapse. N-collapse nằm ở
        # kiến trúc (`mean(dim=1)` + `pos_embed` + BN + biên conv), không nằm ở `cumsum`.
        dt = dt.float()
        B_mat = B_mat.float()
        C_mat = C_mat.float()
        x_conv_f32 = x_conv.float()

        # W = dt * A  (W < 0 vì A = -exp(A_log) < 0 và dt > 0)
        W = dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0)   # [B, L, d_inner, d_state]

        # V = (dt * B) * x
        dB = dt.unsqueeze(-1) * B_mat.unsqueeze(2)           # [B, L, d_inner, d_state]
        V = dB * x_conv_f32.unsqueeze(-1)                    # [B, L, d_inner, d_state]

        # h_t = exp(W_t) * h_{t-1} + V_t   <=>   h_i = sum_{j<=i} exp(P_i - P_j) V_j
        a = torch.exp(W)                                     # in (0,1) -> luôn ổn định
        h = torch.zeros_like(V[:, 0])                        # [B, d_inner, d_state]
        h_list = []
        for t in range(L):
            h = a[:, t] * h + V[:, t]
            h_list.append(h)
        h = torch.stack(h_list, dim=1)                       # [B, L, d_inner, d_state]
        
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
    def __init__(self, d_in=2560, d_model=512, d_out=512, max_seq_len=64, num_layers=2,
                 pool='attn', use_pe=True):
        super().__init__()
        self.d_model = d_model
        self.pool = pool
        self.use_pe = use_pe
        
        # Linear projection
        self.in_proj = nn.Linear(d_in, d_model)
        
        # (22/9) PE TRỞ LẠI (md/22thg9.md §31). Trước đó tôi bỏ PE vì với
        # `mean(dim=1)` nó chỉ đóng góp `mean(PE[0:N])` — offset CHỈ phụ thuộc N, không
        # mang thông tin (§30.3: lệch 5.55 giữa N=8 và N=16 ≈ 25% độ lớn vector đặc trưng).
        # NHƯNG với **attention pooling** thì PE trở nên CÓ ÍCH: attention CHỌN được vị trí
        # (frame cuối của gallery ≈ sát t1; frame đầu của query ≈ sát t2), thay vì bị trung
        # bình hoá như `mean`. Đây đúng là công thức chuẩn của Transformer (PE + attention).
        self.max_seq_len = max_seq_len
        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, d_model)) if use_pe else None

        # Pooling: 'attn' (mặc định) hoặc 'mean' (hành vi cũ)
        self.attn_pool = AttentionPooling(d_model) if pool == 'attn' else None
        
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
        
        # (22/9) PE — có ích khi pooling là ATTENTION (xem __init__).
        if self.pos_embed is not None:
            x = x + self.pos_embed[:, :N, :]
        
        for mamba_layer, norm in zip(self.layers, self.norm_layers):
            res = x
            # Bidirectional processing
            x_fwd = mamba_layer(x)
            x_bwd = mamba_layer(x.flip(dims=[1])).flip(dims=[1])
            x = norm(x_fwd + x_bwd + res)
            
        # Lưu sequence features trước mean pooling (cho temporal consistency loss)
        temporal_seq = x  # [B, N, d_model]
        
        # (22/9) POOLING (md/22thg9.md §31).
        if self.attn_pool is not None:
            # TỔ HỢP LỒI (trọng số softmax tổng = 1) -> scale BẤT BIẾN THEO N.
            # KHÔNG nhân sqrt(N) ở đây: nhân vào tổ hợp lồi sẽ TÁI TẠO phụ thuộc N.
            x = self.attn_pool(x)                          # [B, d_model]
        else:
            # (29/9) BỎ `* sqrt(N)` — đây là lỗi toán, không phải lựa chọn.
            # `mean` chỉ có phương sai ~sigma^2/N khi các frame ĐỘC LẬP (iid).
            # Frame UAV trong 1 cửa sổ tương quan rất cao (rho ~ 0.9):
            # Var(mean) ~ sigma^2 * (rho + (1-rho)/N)   -> gần như KHÔNG giảm theo N
            # Nên nhan sqrt(N) làm SCALE TĂNG theo N, không phải bất biến.
            # Thêm nữa N ngẫu nhiên {8,12,16} => hệ số 2.83/3.46/4.00 theo TỪNG SAMPLE
            # => BN không thể bù. (md/report_25thg9.md §1.1, md/29thg9.md §3 việc #10)
            # Pipeline đóng băng dùng `temporal_pool: attn` (tổ hợp lồi, đã bất biến theo N);
            # nhánh `mean` ở đây chỉ là BASELINE SẠCH để so.
            x = x.mean(dim=1)                              # [B, d_model]
        
        # MLP Head
        x = self.out_mlp(x) # [B, d_out]
        return x, temporal_seq
