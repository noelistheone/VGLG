"""Variate-Gated Local-Global (VGLG) block.

Operates on a hidden representation of shape (B, T, N) where T is a
*time-axis* dimension (could be the original seq_len or a re-projected
d_model along that same axis) and N is the variate (channel) count.

  local path  = depthwise 1D conv along T (variate-independent)
  global path = low-rank linear projection along T
                  T -> rank -> T (acts as a long-range mixer along T)
  gate        = per-variate scalar in (0,1) computed from the input window
                  out = g * local + (1 - g) * global
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .gate import VariateGate


class VGLGBlock(nn.Module):
    def __init__(
        self,
        time_dim: int,
        n_vars: int,
        kernel_size: int = 31,
        rank: int = 8,
        gate_hidden: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd for symmetric padding"
        self.time_dim = time_dim
        self.n_vars = n_vars

        # Local path: depthwise conv over the time axis (one filter per variate)
        self.local_conv = nn.Conv1d(
            in_channels=n_vars,
            out_channels=n_vars,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=n_vars,
        )

        # Global path: low-rank linear over the time axis
        self.W1 = nn.Linear(time_dim, rank, bias=False)
        self.W2 = nn.Linear(rank, time_dim, bias=False)

        # Per-variate adaptive gate
        self.gate = VariateGate(hidden=gate_hidden)

        self.norm = nn.LayerNorm(n_vars)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """`x` shape (B, T, N). Returns (out, g) where g has shape (B, 1, N)."""
        residual = x
        h = self.norm(x)

        # Local path: conv expects (B, N, T); swap, conv, swap back
        h_local = self.local_conv(h.transpose(1, 2)).transpose(1, 2)

        # Global path: low-rank linear acting on the time axis
        h_perm = h.transpose(1, 2)                  # (B, N, T)
        h_global = self.W2(self.W1(h_perm)).transpose(1, 2)

        # Variate gate computed on raw input (more informative than normed)
        g = self.gate(x)                            # (B, 1, N)

        out = g * h_local + (1.0 - g) * h_global
        out = self.dropout(out)
        return residual + out, g
