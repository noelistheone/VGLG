"""DLinear (Zeng et al., AAAI 2023).

Trend/seasonal decomposition by moving average, then a linear projection per
component. The two flavors are individual=True (one linear per variate) and
the simpler shared variant.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _MovingAvg(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        # AvgPool1d expects (B, C, L)
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C). Symmetric edge padding to keep length L.
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        back = x[:, -1:, :].repeat(1, self.kernel_size - 1 - (self.kernel_size - 1) // 2, 1)
        x_padded = torch.cat([front, x, back], dim=1)
        x_avg = self.avg(x_padded.permute(0, 2, 1)).permute(0, 2, 1)
        return x_avg


class _SeriesDecomp(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        self.moving_avg = _MovingAvg(kernel_size)

    def forward(self, x: torch.Tensor):
        trend = self.moving_avg(x)
        seasonal = x - trend
        return seasonal, trend


class DLinear(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        n_vars: int,
        moving_avg: int = 25,
        individual: bool = False,
        **_unused,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_vars = n_vars
        self.individual = individual
        self.decomp = _SeriesDecomp(moving_avg)

        if individual:
            self.linear_seasonal = nn.ModuleList(
                [nn.Linear(seq_len, pred_len) for _ in range(n_vars)]
            )
            self.linear_trend = nn.ModuleList(
                [nn.Linear(seq_len, pred_len) for _ in range(n_vars)]
            )
        else:
            self.linear_seasonal = nn.Linear(seq_len, pred_len)
            self.linear_trend = nn.Linear(seq_len, pred_len)

    def forward(self, x_enc: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        # x_enc: (B, L, N) -> (B, pred_len, N)
        seasonal, trend = self.decomp(x_enc)
        # operate on the time dim
        seasonal = seasonal.permute(0, 2, 1)  # (B, N, L)
        trend = trend.permute(0, 2, 1)

        if self.individual:
            seasonal_out = torch.zeros(
                seasonal.size(0), self.n_vars, self.pred_len,
                device=x_enc.device, dtype=x_enc.dtype,
            )
            trend_out = torch.zeros_like(seasonal_out)
            for i in range(self.n_vars):
                seasonal_out[:, i, :] = self.linear_seasonal[i](seasonal[:, i, :])
                trend_out[:, i, :] = self.linear_trend[i](trend[:, i, :])
        else:
            seasonal_out = self.linear_seasonal(seasonal)
            trend_out = self.linear_trend(trend)

        out = (seasonal_out + trend_out).permute(0, 2, 1)  # (B, pred_len, N)
        return out
