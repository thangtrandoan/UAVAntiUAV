"""Kiến trúc model của UAVAntiUAV.

Đây là package DUY NHẤT được sửa khi thử nghiệm model. Mọi script khác
(`train_reid.py`, `infer.py`, `evaluate_reid.py`, ...) chỉ gọi
`from model import UAVReIDNet, load_checkpoint_verbose` và không cần đổi.

Cấu trúc
    components.py           khởi tạo trọng số, AttentionPooling, ReIDHead
    temporal_mamba.py       SimpleS6Block + TemporalMambaEncoder
    temporal_attention.py   TemporalAttentionEncoder
    registry.py             TEMPORAL_ENCODERS + build_temporal_encoder()
    reidnet.py              UAVReIDNet (backbone + temporal encoder + head)
    checkpoint.py           load_checkpoint_verbose()

Hai trục thử nghiệm
    1. TEMPORAL ENCODER — thêm file mới, gắn `@register_temporal('ten')`, rồi đặt
       `train.temporal_type: "ten"` trong config. Không sửa file nào khác.
    2. PHẦN CÒN LẠI (backbone, pooling, head, cách ghép) — sửa trực tiếp
       `components.py` / `reidnet.py`.

Ràng buộc khi sửa
    - `UAVReIDNet` phải giữ TÊN thuộc tính `self.backbone`, `self.temporal_encoder`,
      `self.head`: key của state_dict sinh từ tên này, đổi tên là checkpoint cũ
      không nạp được.
    - Encoder phải giữ interface `[B, N, d_in] -> ([B, d_out], [B, N, d_model])`;
      `reidnet.py` và mọi script đều dựa vào đó.
    - Temporal pooling nên là tổ hợp lồi (softmax) để scale bất biến theo N.
      Xem md/pipeline.md mục "Pipeline đóng băng".
"""

from .checkpoint import _is_classifier_key, load_checkpoint_verbose
from .components import (
    AttentionPooling,
    ReIDHead,
    weights_init_classifier,
    weights_init_kaiming,
)
from .registry import TEMPORAL_ENCODERS, build_temporal_encoder, register_temporal
from .reidnet import HAS_GASNET, UAVReIDNet
from .temporal_attention import TemporalAttentionEncoder
from .temporal_mamba import HAS_MAMBA, SimpleS6Block, TemporalMambaEncoder

__all__ = [
    # model chính
    'UAVReIDNet',
    # temporal encoder
    'TemporalAttentionEncoder',
    'TemporalMambaEncoder',
    'SimpleS6Block',
    'TEMPORAL_ENCODERS',
    'build_temporal_encoder',
    'register_temporal',
    # khối dùng chung
    'AttentionPooling',
    'ReIDHead',
    'weights_init_kaiming',
    'weights_init_classifier',
    # checkpoint
    'load_checkpoint_verbose',
    '_is_classifier_key',
    # cờ môi trường
    'HAS_GASNET',
    'HAS_MAMBA',
]
