"""VGLG-CNN backbone.

Same outer skeleton as VGLG_MLP but the variate-mixing block is a 1x1 ConvFFN
(viewing the variate axis as channels). This gives a small but distinct
inductive bias compared to the MLP variant.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..layers.revin import RevIN
from .block import VGLGBlock


class _ConvFFN(nn.Module):
    """Channel-wise FFN implemented as 1x1 convs along the time axis."""

    def __init__(self, n_vars: int, ffn_ratio: int = 2, dropout: float = 0.0):
        super().__init__()
        hidden = ffn_ratio * n_vars
        self.norm = nn.LayerNorm(n_vars)
        # Operate on (B, N, T): conv1d with kernel=1 mixes variates.
        self.fc1 = nn.Conv1d(n_vars, hidden, kernel_size=1)
        self.fc2 = nn.Conv1d(hidden, n_vars, kernel_size=1)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, N)
        h = self.norm(x).transpose(1, 2)            # (B, N, T)
        h = self.fc2(self.act(self.fc1(h)))         # (B, N, T)
        return x + self.drop(h.transpose(1, 2))


class VGLG_CNN(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        n_vars: int,
        d_model: int = 128,
        n_layers: int = 2,
        kernel_size: int = 31,
        rank: int = 8,
        ffn_ratio: int = 2,
        dropout: float = 0.1,
        revin: bool = True,
        affine: bool = True,
        gate_hidden: int = 16,
        **_unused,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_vars = n_vars
        self.use_revin = revin
        if revin:
            self.revin = RevIN(n_vars, affine=affine)

        self.input_proj = nn.Linear(seq_len, d_model)
        self.blocks = nn.ModuleList(
            [
                nn.ModuleDict({
                    "vglg": VGLGBlock(
                        time_dim=d_model,
                        n_vars=n_vars,
                        kernel_size=kernel_size,
                        rank=rank,
                        gate_hidden=gate_hidden,
                        dropout=dropout,
                    ),
                    "channel": _ConvFFN(
                        n_vars=n_vars,
                        ffn_ratio=ffn_ratio,
                        dropout=dropout,
                    ),
                })
                for _ in range(n_layers)
            ]
        )
        self.output_proj = nn.Linear(d_model, pred_len)
        self._last_gates: list[torch.Tensor] = []

    def forward(self, x_enc: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if self.use_revin:
            x = self.revin(x_enc, mode="norm")
        else:
            x = x_enc

        h = self.input_proj(x.transpose(1, 2)).transpose(1, 2)
        self._last_gates = []
        for layer in self.blocks:
            h, g = layer["vglg"](h)
            self._last_gates.append(g)
            h = layer["channel"](h)

        out = self.output_proj(h.transpose(1, 2)).transpose(1, 2)
        if self.use_revin:
            out = self.revin(out, mode="denorm")
        return out
