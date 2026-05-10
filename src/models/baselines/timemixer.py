"""TimeMixer (Wang et al., ICLR 2024).

Decomposable multiscale mixing of past time series. We default to
channel_independence=True (matches the strongest reported configuration) and
drop the temporal-mark embedding to keep the implementation self-contained.

Adapted from THUML TSlib (models/TimeMixer.py + layers/Autoformer_EncDec.py
+ layers/StandardNorm.py).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class _MovingAvg(nn.Module):
    def __init__(self, kernel_size: int, stride: int = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        return self.avg(x.permute(0, 2, 1)).permute(0, 2, 1)


class _SeriesDecomp(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        self.moving_avg = _MovingAvg(kernel_size)

    def forward(self, x: torch.Tensor):
        m = self.moving_avg(x)
        return x - m, m


class _DftDecomp(nn.Module):
    def __init__(self, top_k: int = 5):
        super().__init__()
        self.top_k = top_k

    def forward(self, x: torch.Tensor):
        xf = torch.fft.rfft(x, dim=1)
        freq = xf.abs()
        freq[:, 0, :] = 0
        top_k_freq, _ = torch.topk(freq, k=min(self.top_k, freq.size(1)), dim=1)
        xf = torch.where(freq <= top_k_freq.min(dim=1, keepdim=True).values, torch.zeros_like(xf), xf)
        x_season = torch.fft.irfft(xf, n=x.size(1), dim=1)
        return x_season, x - x_season


class _Normalize(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True, non_norm: bool = False):
        super().__init__()
        self.eps = eps
        self.affine = affine
        self.non_norm = non_norm
        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == "norm":
            self.mean = x.mean(dim=1, keepdim=True).detach()
            self.stdev = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + self.eps).detach()
            if self.non_norm:
                return x
            x = (x - self.mean) / self.stdev
            if self.affine:
                x = x * self.weight + self.bias
            return x
        if self.non_norm:
            return x
        if self.affine:
            x = (x - self.bias) / (self.weight + self.eps * self.eps)
        return x * self.stdev + self.mean


class _MultiScaleSeasonMixing(nn.Module):
    """Bottom-up: mix high-resolution season patterns into lower scales."""

    def __init__(self, seq_len: int, down_sampling_window: int, down_sampling_layers: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(
                        seq_len // (down_sampling_window ** i),
                        seq_len // (down_sampling_window ** (i + 1)),
                    ),
                    nn.GELU(),
                    nn.Linear(
                        seq_len // (down_sampling_window ** (i + 1)),
                        seq_len // (down_sampling_window ** (i + 1)),
                    ),
                )
                for i in range(down_sampling_layers)
            ]
        )

    def forward(self, season_list: list[torch.Tensor]) -> list[torch.Tensor]:
        out_high = season_list[0]
        out_low = season_list[1]
        out = [out_high.permute(0, 2, 1)]
        for i in range(len(season_list) - 1):
            out_low_res = self.layers[i](out_high)
            out_low = out_low + out_low_res
            out_high = out_low
            if i + 2 <= len(season_list) - 1:
                out_low = season_list[i + 2]
            out.append(out_high.permute(0, 2, 1))
        return out


class _MultiScaleTrendMixing(nn.Module):
    """Top-down: mix low-resolution trend patterns into higher scales."""

    def __init__(self, seq_len: int, down_sampling_window: int, down_sampling_layers: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(
                        seq_len // (down_sampling_window ** (i + 1)),
                        seq_len // (down_sampling_window ** i),
                    ),
                    nn.GELU(),
                    nn.Linear(
                        seq_len // (down_sampling_window ** i),
                        seq_len // (down_sampling_window ** i),
                    ),
                )
                for i in reversed(range(down_sampling_layers))
            ]
        )

    def forward(self, trend_list: list[torch.Tensor]) -> list[torch.Tensor]:
        rev = list(reversed(trend_list))
        out_low = rev[0]
        out_high = rev[1]
        out = [out_low.permute(0, 2, 1)]
        for i in range(len(rev) - 1):
            out_high_res = self.layers[i](out_low)
            out_high = out_high + out_high_res
            out_low = out_high
            if i + 2 <= len(rev) - 1:
                out_high = rev[i + 2]
            out.append(out_low.permute(0, 2, 1))
        out.reverse()
        return out


class _PastDecomposableMixing(nn.Module):
    def __init__(
        self,
        seq_len: int,
        d_model: int,
        d_ff: int,
        dropout: float,
        decomp_method: str,
        moving_avg: int,
        top_k: int,
        down_sampling_window: int,
        down_sampling_layers: int,
    ):
        super().__init__()
        if decomp_method == "moving_avg":
            self.decomp = _SeriesDecomp(moving_avg)
        elif decomp_method == "dft_decomp":
            self.decomp = _DftDecomp(top_k)
        else:
            raise ValueError(f"Unknown decomp_method: {decomp_method}")

        self.season_mix = _MultiScaleSeasonMixing(seq_len, down_sampling_window, down_sampling_layers)
        self.trend_mix = _MultiScaleTrendMixing(seq_len, down_sampling_window, down_sampling_layers)
        self.out_cross = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x_list: list[torch.Tensor]) -> list[torch.Tensor]:
        lengths = [x.size(1) for x in x_list]
        seasons, trends = [], []
        for x in x_list:
            s, t = self.decomp(x)
            seasons.append(s.permute(0, 2, 1))
            trends.append(t.permute(0, 2, 1))
        out_seasons = self.season_mix(seasons)
        out_trends = self.trend_mix(trends)
        out = []
        for ori, s, t, L in zip(x_list, out_seasons, out_trends, lengths):
            mixed = ori + self.out_cross(s + t)
            out.append(mixed[:, :L, :])
        return out


class TimeMixer(nn.Module):
    """Channel-independent TimeMixer (no temporal-mark embedding).

    Matches the configuration that gives TSlib's reported numbers on most
    long-horizon datasets.
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        n_vars: int,
        d_model: int = 16,
        d_ff: int = 32,
        e_layers: int = 2,
        down_sampling_layers: int = 3,
        down_sampling_window: int = 2,
        down_sampling_method: str = "avg",
        decomp_method: str = "moving_avg",
        moving_avg: int = 25,
        top_k: int = 5,
        dropout: float = 0.1,
        use_norm: int = 1,
        **_unused,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_vars = n_vars
        self.down_sampling_layers = down_sampling_layers
        self.down_sampling_window = down_sampling_window
        self.down_sampling_method = down_sampling_method

        self.normalizers = nn.ModuleList(
            [_Normalize(n_vars, affine=True, non_norm=(use_norm == 0)) for _ in range(down_sampling_layers + 1)]
        )

        # value-only embedding (we ignore the temporal mark for simplicity)
        self.embedding = nn.Sequential(nn.Linear(1, d_model), nn.Dropout(dropout))

        self.pdm_blocks = nn.ModuleList(
            [
                _PastDecomposableMixing(
                    seq_len=seq_len,
                    d_model=d_model,
                    d_ff=d_ff,
                    dropout=dropout,
                    decomp_method=decomp_method,
                    moving_avg=moving_avg,
                    top_k=top_k,
                    down_sampling_window=down_sampling_window,
                    down_sampling_layers=down_sampling_layers,
                )
                for _ in range(e_layers)
            ]
        )

        self.predict_layers = nn.ModuleList(
            [
                nn.Linear(seq_len // (down_sampling_window ** i), pred_len)
                for i in range(down_sampling_layers + 1)
            ]
        )
        self.projection = nn.Linear(d_model, 1, bias=True)

    def _multi_scale_inputs(self, x_enc: torch.Tensor) -> list[torch.Tensor]:
        if self.down_sampling_method == "max":
            pool = nn.MaxPool1d(self.down_sampling_window)
        elif self.down_sampling_method == "avg":
            pool = nn.AvgPool1d(self.down_sampling_window)
        elif self.down_sampling_method == "conv":
            pool = nn.Conv1d(self.n_vars, self.n_vars, 3, padding=1, stride=self.down_sampling_window,
                             padding_mode="circular", bias=False).to(x_enc.device)
        else:
            return [x_enc]

        x = x_enc.permute(0, 2, 1)  # (B, N, T)
        x_list = [x.permute(0, 2, 1)]
        cur = x
        for _ in range(self.down_sampling_layers):
            cur = pool(cur)
            x_list.append(cur.permute(0, 2, 1))
        return x_list

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None) -> torch.Tensor:
        # Build a multi-scale pyramid of the input
        x_list_orig = self._multi_scale_inputs(x_enc)

        # Per-scale normalization, then channel-independent reshape
        x_list = []
        B = x_enc.size(0)
        for i, x in enumerate(x_list_orig):
            T, N = x.size(1), x.size(2)
            xn = self.normalizers[i](x, "norm")
            xn = xn.permute(0, 2, 1).reshape(B * N, T, 1)
            x_list.append(self.embedding(xn))

        # Past-decomposable mixing
        for block in self.pdm_blocks:
            x_list = block(x_list)

        # Future multi-mixing decoder: predict per-scale, sum
        dec_out_list = []
        for i, enc_out in enumerate(x_list):
            dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(0, 2, 1)
            dec_out = self.projection(dec_out)
            dec_out = dec_out.reshape(B, self.n_vars, self.pred_len).permute(0, 2, 1).contiguous()
            dec_out_list.append(dec_out)

        out = torch.stack(dec_out_list, dim=-1).sum(-1)
        out = self.normalizers[0](out, "denorm")
        return out
