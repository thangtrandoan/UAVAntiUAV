"""Đăng ký temporal encoder để chọn qua `train.temporal_type`.

Thêm encoder mới:
    1. Tạo file trong `model/`, ví dụ `model/temporal_my.py`
    2. Khai báo lớp với interface `[B, N, d_in] -> ([B, d_out], [B, N, d_model])`
    3. Gắn decorator:

        from model.registry import register_temporal

        @register_temporal('my_encoder')
        class MyEncoder(nn.Module):
            def __init__(self, d_in=2560, d_model=512, d_out=512, max_seq_len=64,
                         num_layers=2, pool='attn', use_pe=True, **kwargs): ...

    4. Import nó trong `model/__init__.py`
    5. Đặt `train.temporal_type: "my_encoder"` trong config

Không cần sửa `reidnet.py`.
"""

import inspect

from .temporal_attention import TemporalAttentionEncoder
from .temporal_mamba import TemporalMambaEncoder


TEMPORAL_ENCODERS = {
    'mamba':     TemporalMambaEncoder,
    'attention': TemporalAttentionEncoder,
}


def register_temporal(name):
    """Decorator đăng ký một temporal encoder dưới tên `name`."""
    def deco(cls):
        TEMPORAL_ENCODERS[name.lower()] = cls
        return cls
    return deco


def build_temporal_encoder(temporal_type, **kwargs):
    """Tạo encoder theo tên trong registry.

    Chỉ truyền những kwarg mà lớp đó thực sự nhận, nên các encoder không cần
    cùng chữ ký (ví dụ `TemporalMambaEncoder` không có `num_heads`/`dropout`).
    Lớp nào có `**kwargs` thì nhận tất cả.
    """
    key = (temporal_type or 'mamba').lower()
    if key not in TEMPORAL_ENCODERS:
        raise KeyError(f"temporal_type={temporal_type!r} chưa đăng ký. "
                       f"Đang có: {sorted(TEMPORAL_ENCODERS)}")
    cls = TEMPORAL_ENCODERS[key]
    params = inspect.signature(cls.__init__).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return cls(**kwargs)
    return cls(**{k: v for k, v in kwargs.items() if k in params})
