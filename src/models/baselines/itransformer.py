"""iTransformer (Liu et al., ICLR 2024 Spotlight).

The "inverted" view: each variate is a token of length seq_len; attention
operates over variates rather than time steps. Adapted from THUML TSlib.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..layers.transformer import AttentionLayer, Encoder, EncoderLayer, FullAttention


class _DataEmbeddingInverted(nn.Module):
    """Project each variate's full window to d_model. Optionally concat time marks."""

    def __init__(self, seq_len: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.value_embedding = nn.Linear(seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, x_mark: torch.Tensor | None) -> torch.Tensor:
        # x: (B, L, N) -> (B, N, L)
        x = x.permute(0, 2, 1)
        if x_mark is not None:
            # mark broadcast as additional 'variates' so the encoder can attend to them
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], dim=1))
        else:
            x = self.value_embedding(x)
        return self.dropout(x)  # (B, N (+marks), d_model)


class iTransformer(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        n_vars: int,
        d_model: int = 256,
        d_ff: int = 256,
        e_layers: int = 2,
        n_heads: int = 8,
        dropout: float = 0.1,
        activation: str = "gelu",
        use_norm: bool = True,
        **_unused,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_vars = n_vars
        self.use_norm = use_norm

        self.enc_embedding = _DataEmbeddingInverted(seq_len, d_model, dropout)
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
        self.projection = nn.Linear(d_model, pred_len, bias=True)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None) -> torch.Tensor:
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(x_enc.var(dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc = x_enc / stdev

        N = x_enc.size(-1)
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)

        # (B, N+marks, d_model) -> (B, pred_len, N)
        out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]

        if self.use_norm:
            out = out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            out = out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        return out
