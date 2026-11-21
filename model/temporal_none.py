"""Temporal encoder đối chứng không dùng nhánh thời gian (`temporal_type: "none"`).

Mục đích:
    - Tạo baseline thuần thị giác (Visual-Only Control) trong cùng pipeline huấn luyện
      và đánh giá của `UAVReIDNet` để kiểm chứng trực tiếp đóng góp của nhánh thời gian
      (`temporal_type: "mamba"` hoặc `temporal_type: "attention"`).
    - Giữ nguyên hoàn toàn giao diện `[B, N, d_in] -> (temporal_token, temporal_seq)`:
      `temporal_token` có kích thước `[B, 0]` (tensor rỗng 0 chiều), nhờ đó
      `torch.cat([visual_feat, temporal_token], dim=-1)` trong `ReIDHead`,
      `evaluate_reid.py`, `calibrate_threshold.py`, `infer.py` trả về chính xác
      `visual_feat` (`[B, visual_dim]`, tức 960 chiều cho `dinov3_convnext` hoặc
      2560 chiều cho `resnet50_ibn`) mà không có bất kỳ tham số hay nhiễu nào từ
      nhánh temporal.
"""

import torch
import torch.nn as nn


class TemporalNoneEncoder(nn.Module):
    """Encoder rỗng cho chế độ `temporal_type: "none"` (Visual-Only baseline)."""

    def __init__(self, d_in=2560, d_model=0, d_out=0, **kwargs):
        super().__init__()
        self.d_in = int(d_in)
        self.d_model = 0
        self.d_out = 0

    def forward(self, x: torch.Tensor):
        # x: [B, N, d_in]
        b = x.size(0)
        empty_token = x.new_zeros((b, 0))
        return empty_token, x
