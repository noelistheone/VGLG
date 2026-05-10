"""Attention TokenMixer: variate-axis self-attention (iTransformer style)."""
from __future__ import annotations

import torch
import torch.nn as nn


class AttnMixer(nn.Module):
    """Self-attention across variates. Each variate's full time series of length
    `seq_len` is treated as one token of dimension `seq_len`.

    Input  : (B, L, N)
    Output : (B, L, N)

    Note: nn.MultiheadAttention requires embed_dim divisible by num_heads,
    so seq_len % n_heads must be 0. We auto-adjust n_heads down if needed.
    """

    def __init__(self, seq_len: int, n_vars: int, n_heads: int = 4,
                 dropout: float = 0.1, **_unused):
        super().__init__()
        # adjust n_heads so seq_len is divisible
        while n_heads > 1 and seq_len % n_heads != 0:
            n_heads //= 2
        self.attn = nn.MultiheadAttention(
            embed_dim=seq_len,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, N) -> (B, N, L)  (each variate is a token)
        h = x.transpose(1, 2)
        out, _ = self.attn(h, h, h, need_weights=False)
        return out.transpose(1, 2)           # (B, L, N)
