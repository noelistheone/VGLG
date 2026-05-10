"""MetaTSF block: Norm -> TokenMixer -> Norm -> ChannelMLP. Uniform across mixers."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .mixers import build_mixer


class ChannelMLP(nn.Module):
    """Variate-mixing FFN. Same module shared across all mixer variants."""

    def __init__(self, n_vars: int, hidden_mult: int = 2, dropout: float = 0.1):
        super().__init__()
        hidden = max(n_vars * hidden_mult, 8)
        self.net = nn.Sequential(
            nn.Linear(n_vars, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_vars),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MetaTSFBlock(nn.Module):
    """Norm -> TokenMixer -> Norm -> ChannelMLP. All baselines and VGLG share this."""

    def __init__(self, seq_len: int, n_vars: int, mixer_cfg: Any,
                 channel_mlp_mult: int = 2, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(n_vars)
        self.mixer = build_mixer(mixer_cfg, seq_len=seq_len, n_vars=n_vars)
        self.norm2 = nn.LayerNorm(n_vars)
        self.channel_mlp = ChannelMLP(n_vars, hidden_mult=channel_mlp_mult, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, N)
        x = x + self.mixer(self.norm1(x))
        x = x + self.channel_mlp(self.norm2(x))
        return x
