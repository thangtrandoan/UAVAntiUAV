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
        
        # 4. Selective Scan — RECURRENCE TUẦN TỰ (🛠️ 22/9)
        # TRƯỚC ĐÂY: công thức kiểu attention O(L²) — `cumsum` rồi `exp(P_i - P_j)` -> ma trận
        # [B, d_inner, d_state, L, L]. Với d_inner=1024, d_state=16, L=12, B=16 thì ma trận đó
        # ~151 MB cho MỘT layer -> đó là lý do `Avg Mamba + Head = 76.62 ms`.
        #
        # Công thức dưới đây TƯƠNG ĐƯƠNG TOÁN HỌC (đã kiểm chứng bằng numpy: sai số tương đối
        # ~1e-6 = đúng mức float32, KHÔNG phải xấp xỉ) nhưng:
        #   - KHÔNG `cumsum` -> không trừ hai số lớn (bản cũ mất chính xác tăng theo L:
        #     2.4e-7 @L=4 -> 5.0e-6 @L=64)
        #   - `exp(W)` in (0,1) -> KHÔNG cần `masked_fill(-inf)`, không có nguy cơ inf*0=nan
        #   - bộ nhớ [B, L, d_inner, d_state] = **16x nhỏ hơn** ở L=16 (0.26 MB vs 4.19 MB)
        #   - L chỉ 4..16 nên vòng lặp L bước RẺ HƠN NHIỀU so với ma trận LxL
        # ⚠️ THAM SỐ KHÔNG ĐỔI -> **checkpoint cũ vẫn nạp được, KHÔNG cần train lại.**
        # ⚠️ LƯU Ý: đây là fix TỐC ĐỘ/BỘ NHỚ, KHÔNG phải fix N-collapse. N-collapse nằm ở
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


class AttentionPooling(nn.Module):
    """🛠️ (22/9) Attention pooling theo THỜI GIAN — thay `mean(dim=1)`.

    Vì sao tốt hơn `mean` cho bài toán này (md/22thg9.md §31):
      1. Trọng số softmax TỔNG = 1 theo N ⇒ đầu ra là **TỔ HỢP LỒI** của các frame ⇒
         **scale BẤT BIẾN THEO N**. `mean` có phương sai ~1/N nên phải bù bằng `sqrt(N)`;
         tổ hợp lồi thì KHÔNG cần — và nhân `sqrt(N)` vào tổ hợp lồi sẽ TÁI TẠO phụ thuộc N.
      2. **HỌC được trọng số** ⇒ có thể GIẢM trọng số frame bị PAD. `_load_clip` pad clip
         ngắn bằng cách LẶP frame cuối; `mean` cho các frame lặp đó trọng số ĐẦY ĐỦ ⇒
         feature bị kéo lệch về frame cuối. Attention có thể học để bỏ qua chúng.
      3. **Kết hợp được với PE**: attention có thể CHỌN vị trí (vd frame CUỐI của clip
         gallery = sát `t1`; frame ĐẦU của clip query = sát `t2`). Với `mean` thì PE vô
         dụng vì bị trung bình hoá (md/22thg9.md §30.5).
    """
    def __init__(self, d_model, hidden=None):
        super().__init__()
        hidden = hidden if hidden is not None else max(d_model // 2, 8)
        self.score = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        # 🛠️ (22/9) ZERO-INIT lớp CUỐI -> điểm attention = 0 -> softmax ĐỀU -> lúc khởi
        # tạo pooling ĐÚNG BẰNG `mean`. Model học LỆCH DẦN từ đó.
        # Vì sao cần: init mặc định của `nn.Linear(256,1)` cho std(score) ~ 0.4 -> attention
        # hơi lệch NGAY từ đầu (max(a) ~ 0.14-0.25 so với đều 0.06-0.12). Không phải lỗi,
        # nhưng zero-init làm thay đổi này KHÔNG PHÁ gì ở bước 0 rồi mới cải thiện.
        # (An toàn: `weights_init_kaiming` KHÔNG áp lên `temporal_encoder` — chỉ `bnneck`
        #  và `classifier` ở `ReIDHead` — nên zero-init này KHÔNG bị ghi đè.)
        nn.init.zeros_(self.score[-1].weight)
        nn.init.zeros_(self.score[-1].bias)

    def forward(self, x):
        # x: [B, N, D] -> [B, D]
        w = self.score(x).squeeze(-1)             # [B, N]
        a = torch.softmax(w, dim=1)               # [B, N], tổng = 1 theo N
        return (a.unsqueeze(-1) * x).sum(dim=1)   # [B, D] tổ hợp lồi


class TemporalAttentionEncoder(nn.Module):
    """
    Temporal encoder dùng standard Multi-Head Self-Attention thay vì Mamba/SSM.
    Drop-in replacement cho TemporalMambaEncoder — cùng input/output interface.

    So sánh với Mamba:
      - Mamba: O(L) sequential scan, implicit memory, unidirectional (ta chạy bidirectional thủ công)
      - Attention: O(L²) nhưng L chỉ 4~16 nên không đáng kể, bidirectional tự nhiên,
        dễ hiểu hơn, không cần mamba_ssm dependency.
    """

    def __init__(self, d_in=2560, d_model=512, d_out=512, max_seq_len=64, num_layers=2,
                 num_heads=8, dropout=0.1, pool='attn', use_pe=True):
        super().__init__()
        self.d_model = d_model
        self.pool = pool
        self.use_pe = use_pe

        # Linear projection
        self.in_proj = nn.Linear(d_in, d_model)

        # Positional Embedding
        self.max_seq_len = max_seq_len
        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, d_model)) if use_pe else None

        # Pooling
        self.attn_pool = AttentionPooling(d_model) if pool == 'attn' else None

        # Transformer Encoder layers (Self-Attention + FFN)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-LN (ổn định hơn Post-LN)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # MLP Head (giống TemporalMambaEncoder)
        self.out_mlp = nn.Sequential(
            nn.Linear(d_model, d_out),
            nn.BatchNorm1d(d_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # x: [B, N, d_in]
        B, N, _ = x.shape
        x = self.in_proj(x)

        # Positional Embedding
        if self.pos_embed is not None:
            x = x + self.pos_embed[:, :N, :]

        # Self-Attention layers (bidirectional tự nhiên, không cần flip)
        x = self.transformer(x)  # [B, N, d_model]

        # Lưu sequence features trước pooling (cho temporal consistency loss)
        temporal_seq = x  # [B, N, d_model]

        # Pooling
        if self.attn_pool is not None:
            x = self.attn_pool(x)                          # [B, d_model]
        else:
            x = x.mean(dim=1) * (N ** 0.5)                 # [B, d_model]

        # MLP Head
        x = self.out_mlp(x)  # [B, d_out]
        return x, temporal_seq


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
        
        # 🛠️ (22/9) PE TRỞ LẠI (md/22thg9.md §31). Trước đó tôi bỏ PE vì với
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
        
        # 🛠️ (22/9) PE — có ích khi pooling là ATTENTION (xem __init__).
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
        
        # 🛠️ (22/9) POOLING (md/22thg9.md §31).
        if self.attn_pool is not None:
            # TỔ HỢP LỒI (trọng số softmax tổng = 1) -> scale BẤT BIẾN THEO N.
            # ⚠️ KHÔNG nhân sqrt(N) ở đây: nhân vào tổ hợp lồi sẽ TÁI TẠO phụ thuộc N.
            x = self.attn_pool(x)                          # [B, d_model]
        else:
            # `mean` có phương sai ~1/N -> bù bằng sqrt(N) để scale bất biến theo N (§16.2).
            # Trong TRAIN mode BN ở `out_mlp` hấp thụ luôn scale này; giá trị thật là làm
            # `running_mean/var` ĐÚNG CHO MỌI N ở EVAL.
            x = x.mean(dim=1) * (N ** 0.5)                 # [B, d_model]
        
        # MLP Head
        x = self.out_mlp(x) # [B, d_out]
        return x, temporal_seq


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


def _is_classifier_key(name):
    """
    Key của các đầu CLASSIFIER (không tham gia trích feature khi eval).

    Vì sao cần phân biệt: checkpoint train với `num_identities` khác model khởi tạo
    (vd: 502 danh tính lúc train vs 1000 default) sẽ khiến các key này lệch shape.
    Đó là chuyện BÌNH THƯỜNG, không phải lỗi:
      - `backbone.classifier_global` / `classifier_fs`: đầu phân loại pretrain của DINOv3/GASNet
        (chỉ dùng cho pretraining objective)
      - `head.classifier`: đầu ID classifier trong ReIDHead — `ReIDHead.forward` chỉ gọi nó khi
        `self.training == True`; eval trả về `bn_feat` TRƯỚC đó (xem ReIDHead.forward).
    Nên thiếu/lệch các key này KHÔNG ảnh hưởng feature dùng để matching.
    """
    n = name.lower()
    return ('classifier' in n) or n.endswith('.fc.weight') or n.endswith('.fc.bias')


def load_checkpoint_verbose(model, checkpoint_path, tag="checkpoint", log=print):
    """
    Load checkpoint vào model, ĐỒNG THỜI báo cáo đầy đủ:
      - missing        : key có trong model nhưng checkpoint không có → giữ init hiện tại
      - unexpected     : key có trong checkpoint nhưng model không có → bị bỏ
      - shape mismatch : key cùng tên nhưng khác shape → bị bỏ (trước đây lặng im)

    Lý do: mọi chỗ load đều dùng `strict=False`, nghĩa là checkpoint thiếu key
    (ví dụ TOÀN BỘ `backbone.*`) vẫn in ra "Loaded" như thành công. Nếu backbone
    không được nạp, visual feature là feature pretrain chung chứ không phải feature
    đã train → mọi kết luận eval/infer đều nhiễu.

    🛠️ (14/9) FIX: các key CLASSIFIER (xem `_is_classifier_key`) được tách riêng và KHÔNG
    kích hoạt cảnh báo nghiêm trọng — lệch `num_identities` giữa checkpoint và model khởi tạo
    là chuyện bình thường và chúng không nằm trên đường trích feature. Cảnh báo nghiêm trọng
    chỉ bật khi thiếu key `backbone.*` KHÔNG phải classifier.

    Trả về dict summary để caller log/lưu.
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model_state = model.state_dict()

    new_state_dict = {}
    skipped_shape = []
    for k, v in state_dict.items():
        new_k = k[len('_orig_mod.'):] if k.startswith('_orig_mod.') else k
        if new_k in model_state and tuple(v.shape) != tuple(model_state[new_k].shape):
            skipped_shape.append((new_k, tuple(v.shape), tuple(model_state[new_k].shape)))
            continue
        new_state_dict[new_k] = v

    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)

    n_model = len(model_state)
    n_loaded = sum(1 for k in model_state if k in new_state_dict)
    # Chỉ tính các key ẢNH HƯỞNG ĐẶC TRƯNG là "nghiêm trọng"
    backbone_missing = [k for k in missing
                        if k.startswith('backbone.') and not _is_classifier_key(k)]
    head_feature_missing = [k for k in missing
                            if k.startswith('head.') and not _is_classifier_key(k)
                            and 'bnneck' not in k]
    # 🛠️ (24/9) `temporal_encoder.*` CŨNG là trọng số ĐẶC TRƯNG. Thiếu key ở đây nghĩa là
    # kiến trúc lúc DỰNG MODEL khác lúc TRAIN (`temporal_type` hoặc `temporal_pool` truyền sai)
    # -> encoder chạy random init. Trước đây nhóm này chỉ nằm trong ⚠️ MISSING chung nên RẤT DỄ
    # bỏ sót (đã xảy ra thật: eval dựng `pool='attn'` cho checkpoint train `pool='mean'`).
    # Loại `pos_embed` ra vì nó là thành phần TÙY CHỌN đã có khối ℹ️ riêng giải thích bên dưới.
    temporal_missing = [k for k in missing
                        if k.startswith('temporal_encoder.') and 'pos_embed' not in k]
    # 🛠️ (24/9) Lỗ hổng ĐỐI XỨNG với `temporal_missing`: nếu config nói `pool='mean'` nhưng
    # checkpoint train bằng `pool='attn'` thì `attn_pool.*` là UNEXPECTED (KHÔNG phải MISSING)
    # -> trọng số attention pooling bị VỨT ĐI âm thầm và pooling rơi về `mean*sqrt(N)`.
    temporal_unexpected = [k for k in unexpected
                           if k.startswith('temporal_encoder.') and 'pos_embed' not in k]
    classifier_missing = [k for k in missing if _is_classifier_key(k)]
    classifier_mismatch = [s for s in skipped_shape if _is_classifier_key(s[0])]

    log(f"  [{tag}] {os.path.basename(str(checkpoint_path))}: "
        f"{n_loaded}/{n_model} keys của model được nạp "
        f"({len(state_dict)} keys trong file)")

    def _group(keys):
        groups = {}
        for k in keys:
            groups.setdefault(k.split('.')[0], []).append(k)
        return groups

    # 🛠️ (24/9) TÁCH key CLASSIFIER ra khỏi khối ⚠️. `_is_classifier_key` đã có sẵn và khối ℹ️
    # bên dưới giải thích chúng là BÌNH THƯỜNG, NHƯNG trước đây chúng vẫn bị liệt kê dưới ⚠️ và
    # in RA TRƯỚC khối ℹ️ đó -> người đọc thấy "⚠️ MISSING 3 keys" rồi tưởng hỏng, dù 3 key đó
    # chỉ là đầu classifier (lệch `num_identities`), KHÔNG nằm trên đường trích feature.
    missing_feature = [k for k in missing if not _is_classifier_key(k)]
    skipped_shape_feature = [s for s in skipped_shape if not _is_classifier_key(s[0])]

    if missing_feature:
        log(f"  [{tag}] ⚠️ MISSING {len(missing_feature)} keys (giữ init hiện tại):")
        for prefix, keys in sorted(_group(missing_feature).items(), key=lambda kv: -len(kv[1])):
            log(f"      - {prefix}.* : {len(keys)} keys (vd: {keys[0]})")
    # 🛠️ (22/9) `pos_embed` bị BỎ khỏi kiến trúc (§16) nên key này thành "unexpected".
    # Đây là thay đổi CHỦ Ý, không phải lỗi -> báo riêng để không gây hoang mang.
    _pe = [k for k in unexpected if 'pos_embed' in k]
    _te_set = set(temporal_unexpected)
    _other_unexp = [k for k in unexpected if 'pos_embed' not in k and k not in _te_set]
    if _pe:
        log(f"  [{tag}] ℹ️ {len(_pe)} key `pos_embed` bị bỏ — CHỦ Ý, không phải lỗi "
            f"(md/22thg9.md §16): {_pe}")
    if _other_unexp:
        log(f"  [{tag}] ⚠️ UNEXPECTED {len(_other_unexp)} keys trong checkpoint (bị bỏ):")
        for prefix, keys in sorted(_group(_other_unexp).items(), key=lambda kv: -len(kv[1])):
            log(f"      - {prefix}.* : {len(keys)} keys (vd: {keys[0]})")
    if skipped_shape_feature:
        log(f"  [{tag}] ⚠️ SHAPE MISMATCH {len(skipped_shape_feature)} keys (bị bỏ):")
        for name, ck_shape, md_shape in skipped_shape_feature[:10]:
            log(f"      - {name}: checkpoint{ck_shape} vs model{md_shape}")

    # --- Chẩn đoán implementation của TEMPORAL ENCODER (Mamba thật vs SimpleS6Block fallback) ---
    # Vì sao quan trọng: `TemporalMambaEncoder.__init__` chọn module theo `HAS_MAMBA`. Nếu lúc TRAIN
    # có `mamba_ssm` mà lúc EVAL không (hoặc ngược lại) thì trọng số temporal KHÔNG khớp và nhánh
    # temporal chạy random — một confound bậc một khi đánh giá "temporal có ý nghĩa không".
    # Phân biệt bằng shape (hai implementation KHÁC shape):
    #   SimpleS6Block : x_proj = Linear(d_inner, d_state*2 + 1) = (33, d_inner), dt_proj = (d_inner, 1)
    #   mamba_ssm     : x_proj = Linear(d_inner, dt_rank + 2*d_state) = (64, d_inner) với d_model=512,
    #                   dt_rank = ceil(512/16) = 32, dt_proj = (d_inner, dt_rank)
    def _find_shape(sd, target):
        for k, v in sd.items():
            nk = k[len('_orig_mod.'):] if k.startswith('_orig_mod.') else k
            if nk == target:
                return tuple(v.shape)
        return None

    _xk = 'temporal_encoder.layers.0.x_proj.weight'
    _dk = 'temporal_encoder.layers.0.dt_proj.weight'
    ck_x = _find_shape(state_dict, _xk)
    ck_d = _find_shape(state_dict, _dk)
    ms_x = _find_shape(model_state, _xk)
    active_impl = 'mamba_ssm.Mamba (thật)' if HAS_MAMBA else 'SimpleS6Block (fallback)'
    temporal_arch_ok = True

    # 🛠️ (24/9) `x_proj`/`dt_proj` là key RIÊNG CỦA MAMBA. Với encoder ATTENTION thì việc thiếu
    # chúng là BÌNH THƯỜNG — trước đây khối này báo ℹ️ kèm câu "checkpoint có thể thiếu cả nhánh
    # temporal" (gây hoang mang SAI) và đặt `temporal_arch_ok = False`.
    # Với ATTENTION, tiêu chí đúng là: `temporal_encoder.*` có được nạp ĐỦ hay không.
    _attn_enc = type(getattr(model, 'temporal_encoder', None)).__name__ == 'TemporalAttentionEncoder'

    if _attn_enc:
        temporal_arch_ok = (len(temporal_missing) == 0 and len(temporal_unexpected) == 0)
        if temporal_arch_ok:
            _n_tc = len([k for k in model_state if k.startswith('temporal_encoder.')])
            log(f"  [{tag}] ℹ️ Temporal encoder: **ATTENTION** — nạp đủ {_n_tc} keys "
                f"`temporal_encoder.*`. Bỏ qua kiểm tra `x_proj`/`dt_proj` (key riêng của Mamba).")
        else:
            _bad = []
            if temporal_missing:
                _bad.append(f"THIẾU {len(temporal_missing)} keys")
            if temporal_unexpected:
                _bad.append(f"THỪA {len(temporal_unexpected)} keys (có trong checkpoint, "
                            f"không có trong model)")
            log(f"  [{tag}] ❌ Temporal encoder: {' + '.join(_bad)} `temporal_encoder.*` → "
                f"trọng số temporal KHÔNG khớp kiến trúc (xem chi tiết bên dưới).")
    elif ck_x is None:
        # 🛠️ (24/9) Câu cũ — "checkpoint có thể thiếu cả nhánh temporal" — bị in CẢ KHI model là
        # ATTENTION, lúc đó nó VÔ NGHĨA (attention không có `x_proj`) và gây hoang mang.
        # Nay chỉ in trong nhánh MAMBA này, và nói đúng nguyên nhân khả dĩ nhất.
        log(f"  [{tag}] ℹ️ Model là MAMBA nhưng checkpoint KHÔNG có `{_xk}` → không xác định "
            f"được implementation temporal. Nhiều khả năng checkpoint được train bằng "
            f"`temporal_type` khác (ATTENTION?) hoặc thiếu hẳn nhánh temporal.")
        temporal_arch_ok = False
    else:
        if ck_x[0] == 33:
            ck_impl = 'SimpleS6Block (fallback)'
        elif ms_x is not None and ck_x[0] == ms_x[0]:
            ck_impl = active_impl
        else:
            ck_impl = f'KHÁC (x_proj={ck_x}, model={ms_x})'
        if ms_x is not None and ck_x != ms_x:
            temporal_arch_ok = False
            log(f"  [{tag}] ❌ CẢNH BÁO NGHIÊM TRỌNG: trọng số TEMPORAL KHÔNG khớp implementation!")
            log(f"      checkpoint: x_proj{ck_x}, dt_proj{ck_d}  ({ck_impl})")
            log(f"      model     : x_proj{ms_x}, dt_proj{_find_shape(model_state, _dk)}  ({active_impl})")
            log(f"      → `x_proj`/`dt_proj` bị bỏ do lệch shape → nhánh temporal chạy RANDOM.")
            log(f"      → Mọi kết luận về 'temporal/Mamba' đều vô hiệu. Phải cài đúng mamba_ssm "
                f"hoặc TRAIN LẠI với implementation đang dùng.")
        else:
            log(f"  [{tag}] ℹ️ Temporal encoder: cả checkpoint và model đều dùng **{ck_impl}** "
                f"(x_proj{ck_x}) → trọng số temporal khớp.")
            if not HAS_MAMBA and ck_impl == 'SimpleS6Block (fallback)':
                log(f"      ⚠️ Đây KHÔNG phải Mamba thật (`mamba_ssm` chưa cài). Muốn dùng Mamba thật:")
                log(f"         1) `pip install mamba-ssm causal-conv1d`; 2) **TRAIN LẠI** — không chuyển")
                log(f"         được trọng số vì `x_proj`/`dt_proj` khác shape. Trọng số còn lại (conv1d,")
                log(f"         A_log, D, in_proj, out_proj) thì tương thích.")

    # --- Giải thích các key classifier (BÌNH THƯỜNG, không phải lỗi) ---
    if classifier_missing or classifier_mismatch:
        ck_cls = [s[1][0] for s in classifier_mismatch if s[0].endswith('.weight')]
        md_cls = [s[2][0] for s in classifier_mismatch if s[0].endswith('.weight')]
        detail = ""
        if ck_cls and md_cls:
            detail = (f" (số lớp: checkpoint={ck_cls[0]} vs model={md_cls[0]})"
                      f" → model khởi tạo `num_identities` khác lúc train")
        # 🛠️ (24/9) Một key lệch shape nằm ở CẢ `missing` lẫn `skipped_shape`, nên câu cũ
        # "3 missing + 3 lệch shape" đọc như 6 vấn đề trong khi thật ra chỉ là 3 KEY. Tách rõ.
        _cls_mm = {s[0] for s in classifier_mismatch}
        _cls_only_missing = [k for k in classifier_missing if k not in _cls_mm]
        _parts = []
        if classifier_mismatch:
            _parts.append(f"{len(classifier_mismatch)} key lệch shape (bị bỏ, không nạp)")
        if _cls_only_missing:
            _parts.append(f"{len(_cls_only_missing)} key thiếu hẳn")
        log(f"  [{tag}] ℹ️ Đầu CLASSIFIER: " + " + ".join(_parts) + detail)
        log(f"      → KHÔNG phải lỗi: `head.classifier` chỉ dùng khi training "
            f"(eval trả `bn_feat` trước đó); `backbone.classifier_*` là đầu phân loại pretrain.")
        log(f"      → Đường trích feature (backbone trunk + temporal + bnneck) KHÔNG bị ảnh hưởng.")
        # Nếu muốn eval với đúng num_classes: truyền num_identities=<số lớp của checkpoint>.

    if backbone_missing or head_feature_missing or temporal_missing or temporal_unexpected:
        log(f"  [{tag}] ❌ CẢNH BÁO NGHIÊM TRỌNG: "
            f"{len(backbone_missing)} keys `backbone.*` + {len(head_feature_missing)} keys "
            f"feature của `head.*` + {len(temporal_missing)} keys `temporal_encoder.*` THIẾU "
            f"+ {len(temporal_unexpected)} keys `temporal_encoder.*` THỪA "
            f"→ trọng số ĐẶC TRƯNG không khớp checkpoint.")
        for k in (backbone_missing + head_feature_missing + temporal_missing
                  + temporal_unexpected)[:5]:
            log(f"      - {k}")
        log(f"      → Trọng số ĐẶC TRƯNG đang là init/pretrain, KHÔNG phải trọng số đã train.")
        log(f"      → Kết quả eval/infer sẽ nhiễu (visual feature không khớp temporal/head).")
        if temporal_missing or temporal_unexpected:
            log(f"      → RIÊNG `temporal_encoder.*`: kiến trúc dựng model KHÁC lúc train.")
            log(f"        Kiểm tra `train.temporal_type` và `train.temporal_pool` trong config "
                f"có khớp checkpoint không.")
        log(f"      → Kiểm tra: checkpoint có chứa các key này không? Có lệch tên/shape không?")

    return {
        'tag': tag,
        'path': str(checkpoint_path),
        'n_model_keys': n_model,
        'n_loaded_keys': n_loaded,
        'missing': list(missing),
        'unexpected': list(unexpected),
        'skipped_shape': [s[0] for s in skipped_shape],
        'backbone_missing': backbone_missing,
        'head_feature_missing': head_feature_missing,
        'temporal_missing': temporal_missing,
        'temporal_unexpected': temporal_unexpected,
        'classifier_missing': classifier_missing,
        'classifier_shape_mismatch': [s[0] for s in classifier_mismatch],
        'temporal_impl_checkpoint': ck_impl if ck_x is not None else None,
        'temporal_impl_active': active_impl,
        'temporal_arch_ok': temporal_arch_ok,
    }


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
    def __init__(self, gasnet_weights_path=None, num_identities=1000, freeze_backbone=True,
                 backbone='resnet50_ibn', temporal_pool='attn', temporal_pe=True,
                 temporal_type='mamba'):
        super().__init__()
        
        self.temporal_type = temporal_type
        
        # Tự động trỏ path mặc định nếu không truyền
        if gasnet_weights_path is None:
            gasnet_weights_path = os.environ.get('GASNET_WEIGHTS',
                os.path.abspath(os.path.join(os.path.dirname(__file__), '../UAV/gasnet_project/test/gasnet.best.pth')))
            
        # 1. Visual Backbone
        if HAS_GASNET:
            self.backbone = GASNet(num_classes=num_identities, backbone=backbone, use_gem=True)
            if os.path.exists(gasnet_weights_path):
                state_dict = torch.load(gasnet_weights_path, map_location='cpu')
                # torch.compile lưu state_dict với prefix _orig_mod. — phải strip trước khi load
                state_dict = {
                    (k[len('_orig_mod.'):] if k.startswith('_orig_mod.') else k): v
                    for k, v in state_dict.items()
                }
                # load_state_dict(strict=False) VẪN throw RuntimeError nếu shape mismatch
                # (vd: classifier_global của VRU có num_classes khác num_identities UAV)
                # → lọc trước chỉ giữ key có shape khớp chính xác.
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
        
        if temporal_type == 'attention':
            print(f"  ⚡ Temporal encoder: ATTENTION (Self-Attention Transformer)")
            self.temporal_encoder = TemporalAttentionEncoder(
                d_in=visual_dim, d_model=512, d_out=512, max_seq_len=64, num_layers=2,
                num_heads=8, dropout=0.1, pool=temporal_pool, use_pe=temporal_pe
            )
        else:
            print(f"  ⚡ Temporal encoder: MAMBA ({'mamba_ssm' if HAS_MAMBA else 'SimpleS6Block fallback'})")
            self.temporal_encoder = TemporalMambaEncoder(
                d_in=visual_dim, d_model=512, d_out=512, max_seq_len=64, num_layers=2,
                pool=temporal_pool, use_pe=temporal_pe
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
