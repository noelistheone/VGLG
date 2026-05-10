"""GRUForecaster: same skeleton as LSTMForecaster, with GRU cells."""
from __future__ import annotations

import torch
import torch.nn as nn

from ..layers.revin import RevIN


class GRUForecaster(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        n_vars: int,
        d_model: int = 128,
        e_layers: int = 2,
        dropout: float = 0.1,
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
        self.gru = nn.GRU(
            input_size=1, hidden_size=d_model,
            num_layers=e_layers, batch_first=True,
            dropout=dropout if e_layers > 1 else 0.0,
        )
        self.head = nn.Linear(d_model, pred_len)

    def forward(self, x_enc: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if self.use_revin:
            x = self.revin(x_enc, mode="norm")
        else:
            x = x_enc
        B, L, N = x.shape
        x = x.permute(0, 2, 1).reshape(B * N, L, 1)
        out, _ = self.gru(x)
        last = out[:, -1, :]
        pred = self.head(last).reshape(B, N, self.pred_len).permute(0, 2, 1)
        if self.use_revin:
            pred = self.revin(pred, mode="denorm")
        return pred
