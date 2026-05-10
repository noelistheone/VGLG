"""VGLG-MLP backbone.

Pipeline (over a batch of shape (B, L, N)):
    RevIN(norm) -> input projection L -> d_model (per variate)
    -> N_layers x [VGLG block (time mixing) + Channel MLP (variate mixing)]
    -> output projection d_model -> pred_len
    -> RevIN(denorm)
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..layers.revin import RevIN
from .block import VGLGBlock


class _ChannelMLP(nn.Module):
    """Variate-mixing FFN. Operates on the last dim (n_vars)."""

    def __init__(self, n_vars: int, hidden: int, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(n_vars)
        self.fc1 = nn.Linear(n_vars, hidden)
        self.fc2 = nn.Linear(hidden, n_vars)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.drop(self.fc2(self.act(self.fc1(self.norm(x)))))


class VGLG_MLP(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        n_vars: int,
        d_model: int = 128,
        n_layers: int = 2,
        kernel_size: int = 31,
        rank: int = 8,
        channel_mlp_ratio: int = 2,
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
                    "channel": _ChannelMLP(
                        n_vars=n_vars,
                        hidden=channel_mlp_ratio * n_vars,
                        dropout=dropout,
                    ),
                })
                for _ in range(n_layers)
            ]
        )
        self.output_proj = nn.Linear(d_model, pred_len)
        self._last_gates: list[torch.Tensor] = []

    def forward(self, x_enc: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        # x_enc: (B, L, N)
        if self.use_revin:
            x = self.revin(x_enc, mode="norm")
        else:
            x = x_enc

        # project time axis L -> d_model: act on (B, N, L) -> (B, N, d_model)
        h = self.input_proj(x.transpose(1, 2)).transpose(1, 2)  # (B, d_model, N)

        self._last_gates = []
        for layer in self.blocks:
            h, g = layer["vglg"](h)
            self._last_gates.append(g)
            h = layer["channel"](h)

        out = self.output_proj(h.transpose(1, 2)).transpose(1, 2)  # (B, pred_len, N)
        if self.use_revin:
            out = self.revin(out, mode="denorm")
        return out
