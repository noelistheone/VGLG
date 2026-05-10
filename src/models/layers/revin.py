"""Reversible Instance Normalization (Kim et al., ICLR 2022).

Normalize each (instance, variate) over the time axis, then de-normalize the
prediction with the same statistics. Optional learnable affine.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))
        self._mean: torch.Tensor | None = None
        self._stdev: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        """`x` shape: (B, L, N). `mode` in {'norm', 'denorm'}."""
        if mode == "norm":
            self._mean = x.mean(dim=1, keepdim=True).detach()
            self._stdev = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + self.eps).detach()
            x = (x - self._mean) / self._stdev
            if self.affine:
                x = x * self.affine_weight + self.affine_bias
            return x
        elif mode == "denorm":
            assert self._mean is not None and self._stdev is not None, (
                "RevIN: must call forward(mode='norm') before 'denorm'."
            )
            if self.affine:
                x = (x - self.affine_bias) / (self.affine_weight + self.eps * self.eps)
            x = x * self._stdev + self._mean
            return x
        else:
            raise ValueError(f"Unknown RevIN mode: {mode}")
