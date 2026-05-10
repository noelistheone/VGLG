"""ModernTCN (Luo & Wang, ICLR 2024 Spotlight).

Simplified single-stage variant of the official ModernTCN. Each variate is
patched, then run through `num_blocks` residual blocks. Each block applies
one large depthwise conv along the time (patch) axis followed by two grouped
ConvFFNs that mix along the feature dim and the variate axis respectively.

We omit the multi-stage downsampling pyramid, time-feature embedding, and
structural reparameterisation tricks from the official implementation. With
a single stage these match what the published configs use on most ETT /
Weather settings.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..layers.revin import RevIN


class _LargeKernelConv(nn.Module):
    """Depthwise large-kernel conv with an optional auxiliary small-kernel branch."""

    def __init__(self, channels: int, large: int, small: int | None):
        super().__init__()
        self.large_conv = nn.Conv1d(
            channels, channels, kernel_size=large,
            padding=large // 2, groups=channels, bias=False,
        )
        self.bn_large = nn.BatchNorm1d(channels)
        self.small_conv = None
        if small is not None and small < large:
            self.small_conv = nn.Conv1d(
                channels, channels, kernel_size=small,
                padding=small // 2, groups=channels, bias=False,
            )
            self.bn_small = nn.BatchNorm1d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.bn_large(self.large_conv(x))
        if self.small_conv is not None:
            out = out + self.bn_small(self.small_conv(x))
        return out


class _Block(nn.Module):
    """One ModernTCN residual block.

    Layout (with M = n_vars, D = dmodel):
        x: (B, M, D, P)
        -> reshape (B, M*D, P), large-kernel depthwise conv (groups=M*D)
        -> BatchNorm over D
        -> reshape (B, M*D, P), grouped ConvFFN with groups=M (mixes D within each variate)
        -> reshape (B, D*M, P), grouped ConvFFN with groups=D (mixes M within each feature)
    """

    def __init__(self, n_vars: int, dmodel: int, dff: int, large: int, small: int | None,
                 drop: float = 0.1):
        super().__init__()
        self.n_vars = n_vars
        self.dmodel = dmodel
        c = n_vars * dmodel
        self.dw = _LargeKernelConv(c, large, small)
        self.norm = nn.BatchNorm1d(dmodel)

        # ConvFFN 1: groups=n_vars  (mix the D axis within each variate)
        self.ffn1 = nn.Sequential(
            nn.Conv1d(n_vars * dmodel, n_vars * dff, 1, groups=n_vars),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Conv1d(n_vars * dff, n_vars * dmodel, 1, groups=n_vars),
            nn.Dropout(drop),
        )
        # ConvFFN 2: groups=dmodel  (mix the M axis within each feature)
        self.ffn2 = nn.Sequential(
            nn.Conv1d(dmodel * n_vars, dmodel * dff, 1, groups=dmodel),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Conv1d(dmodel * dff, dmodel * n_vars, 1, groups=dmodel),
            nn.Dropout(drop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, M, D, P)
        residual = x
        B, M, D, P = x.shape
        z = x.reshape(B, M * D, P)
        z = self.dw(z)
        z = z.reshape(B * M, D, P)
        z = self.norm(z).reshape(B, M, D, P)

        z = z.reshape(B, M * D, P)
        z = self.ffn1(z).reshape(B, M, D, P)

        z = z.permute(0, 2, 1, 3).reshape(B, D * M, P)
        z = self.ffn2(z).reshape(B, D, M, P).permute(0, 2, 1, 3)

        return residual + z


class ModernTCN(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        n_vars: int,
        d_model: int = 64,
        ffn_ratio: int = 1,
        num_blocks: int = 1,
        large_size: int = 31,
        small_size: int = 5,
        patch_size: int = 16,
        patch_stride: int = 8,
        backbone_dropout: float = 0.1,
        head_dropout: float = 0.1,
        revin: bool = True,
        affine: bool = True,
        **_unused,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_vars = n_vars
        self.use_revin = revin
        if revin:
            self.revin = RevIN(n_vars, affine=affine)

        # Patch / stem: per-variate Conv1d 1 -> d_model
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.stem_conv = nn.Conv1d(1, d_model, kernel_size=patch_size, stride=patch_stride)
        self.stem_bn = nn.BatchNorm1d(d_model)

        # number of patches after stem (with right-padding to keep multiples of stride)
        self.patch_num = (seq_len + patch_stride - 1) // patch_stride
        # (matches the official "stem layer padding" path, see forward)

        d_ff = d_model * ffn_ratio
        self.blocks = nn.ModuleList(
            [
                _Block(n_vars=n_vars, dmodel=d_model, dff=d_ff,
                       large=large_size, small=small_size, drop=backbone_dropout)
                for _ in range(num_blocks)
            ]
        )

        head_nf = d_model * self.patch_num
        self.head = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Linear(head_nf, pred_len),
            nn.Dropout(head_dropout),
        )

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None) -> torch.Tensor:
        # x_enc: (B, L, N)
        if self.use_revin:
            x = self.revin(x_enc, mode="norm")
        else:
            x = x_enc

        # to (B, N, L), then patch each variate independently
        x = x.permute(0, 2, 1)
        B, M, L = x.shape

        # right-pad so that L is a multiple of patch_stride
        if self.patch_size != self.patch_stride:
            pad_len = self.patch_size - self.patch_stride
            x = torch.cat([x, x[:, :, -1:].repeat(1, 1, pad_len)], dim=-1)

        # collapse variate into batch for the stem conv
        x = x.reshape(B * M, 1, x.size(-1))
        x = self.stem_conv(x)
        x = self.stem_bn(x)
        # back to (B, M, D, P)
        D, P = x.size(1), x.size(2)
        x = x.reshape(B, M, D, P)

        # match the head's expected patch_num (in case our right-pad is off by 1)
        if P != self.patch_num:
            if P > self.patch_num:
                x = x[:, :, :, : self.patch_num]
            else:
                pad = self.patch_num - P
                x = torch.cat([x, x[:, :, :, -1:].repeat(1, 1, 1, pad)], dim=-1)

        for blk in self.blocks:
            x = blk(x)

        # head: (B, M, D, P) -> (B, M, pred_len)
        out = self.head(x)
        # to (B, pred_len, N)
        out = out.permute(0, 2, 1)

        if self.use_revin:
            out = self.revin(out, mode="denorm")
        return out
