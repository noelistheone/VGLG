"""MLP TokenMixer: time-axis MLP (sole representative of MLP family)."""
from __future__ import annotations

import torch
import torch.nn as nn


class MLPMixer(nn.Module):
    """Two-layer MLP applied along the time axis (TSMixer / MLP-Mixer style).

    Input  : (B, L, N)
    Output : (B, L, N)
    """

    def __init__(self, seq_len: int, n_vars: int, hidden_mult: int = 2,
                 dropout: float = 0.1, **_unused):
        super().__init__()
        hidden = seq_len * hidden_mult
        self.net = nn.Sequential(
            nn.Linear(seq_len, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, seq_len),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, L, N) -> (B, N, L) -> MLP -> (B, N, L) -> (B, L, N)
        return self.net(x.transpose(1, 2)).transpose(1, 2)
