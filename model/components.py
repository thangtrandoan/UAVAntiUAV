"""Khối dùng chung: khởi tạo trọng số, temporal pooling, ReID head."""

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class AttentionPooling(nn.Module):
    """🛠️ (22/9) Attention pooling theo THỜI GIAN — thay `mean(dim=1)`.

    Vì sao tốt hơn `mean` cho bài toán này (md/22thg9.md §31):
      1. Trọng số softmax TỔNG = 1 theo N ⇒ đầu ra là **TỔ HỢP LỒI** của các frame ⇒
         **scale BẤT BIẾN THEO N**. `mean` chỉ bất biến nếu frame iid; với frame UAV
         tương quan cao thì không, và nhân `sqrt(N)` vào tổ hợp lồi sẽ TÁI TẠO phụ thuộc N.
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
        # (22/9) ZERO-INIT lớp CUỐI -> điểm attention = 0 -> softmax ĐỀU -> lúc khởi
        # tạo pooling ĐÚNG BẰNG `mean`. Model học LỆCH DẦN từ đó.
        # Vì sao cần: init mặc định của `nn.Linear(256,1)` cho std(score) ~ 0.4 -> attention
        # hơi lệch NGAY từ đầu (max(a) ~ 0.14-0.25 so với đều 0.06-0.12). Không phải lỗi,
        # nhưng zero-init làm thay đổi này KHÔNG PHÁ gì ở bước 0 rồi mới cải thiện.
        # (An toàn: `weights_init_kaiming` KHÔNG áp lên `temporal_encoder` — chỉ `bnneck`
        # và `classifier` ở `ReIDHead` — nên zero-init này KHÔNG bị ghi đè.)
        nn.init.zeros_(self.score[-1].weight)
        nn.init.zeros_(self.score[-1].bias)

    def forward(self, x):
        # x: [B, N, D] -> [B, D]
        w = self.score(x).squeeze(-1)             # [B, N]
        a = torch.softmax(w, dim=1)               # [B, N], tổng = 1 theo N
        return (a.unsqueeze(-1) * x).sum(dim=1)   # [B, D] tổ hợp lồi


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
