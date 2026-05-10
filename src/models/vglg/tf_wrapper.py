"""VGLG-Transformer backbone.

Replaces the variate-mixing MLP with iTransformer-style self-attention along
the variate axis. Each variate is treated as a token of length d_model, and
attention learns cross-variate dependencies.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..layers.revin import RevIN
from .block import VGLGBlock


class _VariateAttention(nn.Module):
    """Self-attention along the variate (channel) axis."""

    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, N) where T is treated as feature dim and N as token count.
        # Permute to (B, N, T) so we attend across variates.
        h = x.transpose(1, 2)                       # (B, N, T)
        h_norm = self.norm(h)
        attn_out, _ = self.attn(h_norm, h_norm, h_norm, need_weights=False)
        h = h + self.drop(attn_out)
        h = h + self.drop(self.ffn(self.ffn_norm(h)))
        return h.transpose(1, 2)                    # back to (B, T, N)


class VGLG_Transformer(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        n_vars: int,
        d_model: int = 128,
        n_layers: int = 2,
        n_heads: int = 8,
        kernel_size: int = 31,
        rank: int = 8,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
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
                    "attn": _VariateAttention(
                        d_model=d_model,
                        n_heads=n_heads,
                        dropout=attn_dropout,
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
            h = layer["attn"](h)

        out = self.output_proj(h.transpose(1, 2)).transpose(1, 2)
        if self.use_revin:
            out = self.revin(out, mode="denorm")
        return out
