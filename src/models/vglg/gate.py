"""Variate-aware gate.

For each variate n, compute a small set of statistics over the *current input
window* (mean, std, lag-1 autocorrelation, dominant frequency magnitude), then
map them through a tiny MLP to a scalar g_n in (0, 1). The gate output is then
broadcast to mix the local and global mixing paths.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class VariateGate(nn.Module):
    def __init__(self, n_stats: int = 4, hidden: int = 16):
        super().__init__()
        self.n_stats = n_stats
        # Per-feature LayerNorm gives the MLP a stable scale without depending
        # on the batch dimension (avoids NaN when B == 1 at eval time).
        self.norm = nn.LayerNorm(n_stats)
        self.mlp = nn.Sequential(
            nn.Linear(n_stats, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    @staticmethod
    def compute_stats(x: torch.Tensor) -> torch.Tensor:
        """`x` shape (B, L, N) -> stats shape (B, N, 4)."""
        mean = x.mean(dim=1, keepdim=True)                          # (B, 1, N)
        std = x.std(dim=1, keepdim=True, unbiased=False)            # (B, 1, N)

        x_centered = x - mean
        # lag-1 autocorrelation
        num = (x_centered[:, 1:] * x_centered[:, :-1]).sum(dim=1, keepdim=True)
        den = x_centered.pow(2).sum(dim=1, keepdim=True) + 1e-8
        autocorr = num / den

        # dominant non-DC frequency mag, normalized by total mag
        # Force fp32 to avoid ComplexHalf when running under autocast.
        fft = torch.fft.rfft(x_centered.float(), dim=1)
        mag = fft.abs().to(x.dtype)
        if mag.size(1) > 1:
            dom = mag[:, 1:].max(dim=1, keepdim=True).values
        else:
            dom = mag[:, :1]
        dom = dom / (mag.sum(dim=1, keepdim=True) + 1e-8)

        stats = torch.cat([mean, std, autocorr, dom], dim=1)        # (B, 4, N)
        return stats.transpose(1, 2)                                # (B, N, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns gate of shape (B, 1, N) suitable for broadcasting along time."""
        stats = self.compute_stats(x)                               # (B, N, 4)
        stats = self.norm(stats)
        g = torch.sigmoid(self.mlp(stats))                          # (B, N, 1)
        return g.transpose(1, 2)                                    # (B, 1, N)


def gate_entropy(g: torch.Tensor) -> torch.Tensor:
    """Bernoulli entropy of gate values, averaged over batch and variates.

    Useful as an auxiliary regularizer to prevent the gate from collapsing
    to 0 or 1 across all variates.
    """
    g = g.clamp(1e-6, 1.0 - 1e-6)
    h = -(g * torch.log(g) + (1 - g) * torch.log(1 - g))
    return h.mean()
