"""PatchTST (Nie et al., ICLR 2023).

Channel-independent: each variate is patched into overlapping windows, fed
through a shared Transformer encoder, then a flattening head produces the
forecast for that variate. Adapted from THUML TSlib.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..layers.transformer import (
    AttentionLayer,
    Encoder,
    EncoderLayer,
    FullAttention,
    PositionalEmbedding,
)


class _PatchEmbedding(nn.Module):
    def __init__(self, d_model: int, patch_len: int, stride: int, padding: int, dropout: float):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.pad = nn.ReplicationPad1d((0, padding))
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int]:
        # x: (B, N, L) -> (B, N, L+pad) -> unfold -> (B, N, P, patch_len)
        n_vars = x.size(1)
        x = self.pad(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = x.reshape(x.size(0) * x.size(1), x.size(2), x.size(3))  # (B*N, P, patch_len)
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x), n_vars


class _FlattenHead(nn.Module):
    def __init__(self, n_vars: int, nf: int, target_window: int, head_dropout: float = 0.0):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.linear(self.flatten(x)))


class PatchTST(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        n_vars: int,
        d_model: int = 128,
        d_ff: int = 256,
        e_layers: int = 3,
        n_heads: int = 16,
        patch_len: int = 16,
        stride: int = 8,
        dropout: float = 0.2,
        head_dropout: float = 0.0,
        activation: str = "gelu",
        use_norm: bool = True,
        **_unused,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_vars = n_vars
        self.use_norm = use_norm
        padding = stride

        self.patch_embedding = _PatchEmbedding(d_model, patch_len, stride, padding, dropout)

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(mask_flag=False, attention_dropout=dropout),
                        d_model=d_model, n_heads=n_heads,
                    ),
                    d_model=d_model, d_ff=d_ff, dropout=dropout, activation=activation,
                )
                for _ in range(e_layers)
            ],
            norm_layer=nn.LayerNorm(d_model),
        )

        head_nf = d_model * (int((seq_len - patch_len) / stride + 2))
        self.head = _FlattenHead(n_vars, head_nf, pred_len, head_dropout=head_dropout)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None) -> torch.Tensor:
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(x_enc.var(dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc = x_enc / stdev

        # patch + embed: (B, N, L) -> (B*N, P, d_model)
        x = x_enc.permute(0, 2, 1)
        enc_out, n_vars = self.patch_embedding(x)

        enc_out, _ = self.encoder(enc_out)                                   # (B*N, P, d_model)
        enc_out = enc_out.reshape(-1, n_vars, enc_out.size(-2), enc_out.size(-1))
        enc_out = enc_out.permute(0, 1, 3, 2)                                # (B, N, d_model, P)

        out = self.head(enc_out).permute(0, 2, 1)                            # (B, pred_len, N)

        if self.use_norm:
            out = out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            out = out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        return out
