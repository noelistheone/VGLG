"""Conv TokenMixer: large-kernel depthwise + pointwise conv (CNN family rep)."""
from __future__ import annotations

import torch
import torch.nn as nn


class ConvMixer(nn.Module):
    """ModernTCN-flavoured token mixer: depthwise large kernel followed by 1x1.

    Input  : (B, L, N)
    Output : (B, L, N)
    """

    def __init__(self, seq_len: int, n_vars: int, kernel_size: int = 31,
                 dropout: float = 0.1, **_unused):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        self.dw_conv = nn.Conv1d(
            in_channels=n_vars, out_channels=n_vars,
            kernel_size=kernel_size, padding=kernel_size // 2,
            groups=n_vars,
        )
        self.pw_conv = nn.Conv1d(n_vars, n_vars, kernel_size=1)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.transpose(1, 2)                # (B, N, L)
        h = self.dw_conv(h)
        h = self.act(h)
        h = self.pw_conv(h)
        h = self.drop(h)
        return h.transpose(1, 2)             # (B, L, N)
