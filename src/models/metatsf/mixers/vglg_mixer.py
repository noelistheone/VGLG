"""VGLG TokenMixer (ours): variate-gated local-global fusion.

Local path : depthwise large-kernel conv along the time axis (variate-independent)
Global path: low-rank linear projection along the time axis (long-range mixer)
Gate       : per-variate scalar in (0, 1) computed from input statistics

Input  : (B, L, N)
Output : (B, L, N)
"""
from __future__ import annotations

import torch
import torch.nn as nn


class VariateGate(nn.Module):
    """Per-variate gate from 4 input statistics: mean, std, lag-1 autocorr, dom freq."""

    def __init__(self, n_stats: int = 4, hidden: int = 16):
        super().__init__()
        # Per-feature LayerNorm gives the MLP a stable scale (no batch dependence).
        self.norm = nn.LayerNorm(n_stats)
        self.mlp = nn.Sequential(
            nn.Linear(n_stats, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    @staticmethod
    def compute_stats(x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, N) -> (B, N, 4)
        mean = x.mean(dim=1, keepdim=True)                            # (B, 1, N)
        std = x.std(dim=1, keepdim=True, unbiased=False)              # (B, 1, N)
        x_c = x - mean
        num = (x_c[:, 1:] * x_c[:, :-1]).sum(dim=1, keepdim=True)
        den = x_c.pow(2).sum(dim=1, keepdim=True) + 1e-8
        autocorr = num / den
        # Force fp32 to avoid ComplexHalf under autocast.
        fft = torch.fft.rfft(x_c.float(), dim=1)
        mag = fft.abs().to(x.dtype)
        if mag.size(1) > 1:
            dom = mag[:, 1:].max(dim=1, keepdim=True).values
        else:
            dom = mag[:, :1]
        dom = dom / (mag.sum(dim=1, keepdim=True) + 1e-8)
        stats = torch.cat([mean, std, autocorr, dom], dim=1)          # (B, 4, N)
        return stats.transpose(1, 2)                                  # (B, N, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stats = self.compute_stats(x)
        stats = self.norm(stats)
        g = torch.sigmoid(self.mlp(stats))                            # (B, N, 1)
        return g.transpose(1, 2)                                      # (B, 1, N)


class VGLGMixer(nn.Module):
    """VGLG TokenMixer.

    `gate_mode` controls how the local/global mixing weight is determined:
      - "learned"  : the default. A small MLP over per-variate statistics.
      - "fixed_0.5": frozen 50/50 mix (ablates the gate's adaptivity).
      - "fixed_1.0": local-only (ablates the global path).
      - "fixed_0.0": global-only (ablates the local path).
    """

    def __init__(self, seq_len: int, n_vars: int, kernel_size: int = 31,
                 rank: int = 8, dropout: float = 0.1,
                 gate_hidden: int = 16, gate_entropy_reg: float = 0.0,
                 gate_mode: str = "learned", **_unused):
        super().__init__()
        assert kernel_size % 2 == 1
        valid_modes = {"learned", "fixed_0.5", "fixed_1.0", "fixed_0.0"}
        assert gate_mode in valid_modes, f"gate_mode must be one of {valid_modes}"
        self.seq_len = seq_len
        self.n_vars = n_vars
        self.gate_mode = gate_mode

        self.local_conv = nn.Conv1d(
            in_channels=n_vars, out_channels=n_vars,
            kernel_size=kernel_size, padding=kernel_size // 2,
            groups=n_vars,
        )
        self.W1 = nn.Linear(seq_len, rank, bias=False)
        self.W2 = nn.Linear(rank, seq_len, bias=False)

        # Always instantiate the gate so checkpoint shapes are stable,
        # but skip its forward pass when frozen.
        self.gate = VariateGate(hidden=gate_hidden)
        if gate_mode != "learned":
            fixed = float(gate_mode.split("_")[1])
            self.register_buffer("_fixed_g", torch.tensor(fixed))
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.gate_entropy_reg = gate_entropy_reg
        self._last_gate: torch.Tensor | None = None  # hook for visualisation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, N)
        h_local = self.local_conv(x.transpose(1, 2)).transpose(1, 2)
        h_global = self.W2(self.W1(x.transpose(1, 2))).transpose(1, 2)
        if self.gate_mode == "learned":
            g = self.gate(x)                                          # (B, 1, N)
        else:
            g = self._fixed_g.expand(x.size(0), 1, x.size(-1))
        self._last_gate = g.detach()
        out = g * h_local + (1.0 - g) * h_global
        return self.drop(out)

    def gate_entropy_loss(self) -> torch.Tensor:
        """Optional regulariser to discourage gate collapse to 0/1.

        Returns 0 if `gate_entropy_reg` is 0 or no forward has run yet.
        """
        if self.gate_entropy_reg == 0.0 or self._last_gate is None:
            return torch.tensor(0.0)
        g = self._last_gate.clamp(1e-6, 1 - 1e-6)
        H = -(g * g.log() + (1 - g) * (1 - g).log())
        return -self.gate_entropy_reg * H.mean()
