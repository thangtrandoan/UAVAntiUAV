"""Temporal encoder dùng Multi-Head Self-Attention."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .components import AttentionPooling


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
            x = x.mean(dim=1)                              # [B, d_model]

        # MLP Head
        x = self.out_mlp(x)  # [B, d_out]
        return x, temporal_seq
