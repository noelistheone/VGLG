"""SegRNN (Lin et al., arXiv:2308.11200).

Modern-RNN baseline that splits the input sequence into segments of length
`seg_len`, encodes each segment with a shared GRU, then auto-regressively
decodes the future segments using positional + channel embeddings.

Adapted from THUML TSlib (models/SegRNN.py); only the forecasting branch
is kept and the task_name dispatch is dropped.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SegRNN(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        n_vars: int,
        d_model: int = 512,
        seg_len: int = 48,
        dropout: float = 0.5,
        **_unused,
    ):
        super().__init__()
        # SegRNN requires seq_len and pred_len divisible by seg_len.
        # Round seg_len down to a divisor that works for both.
        if seq_len % seg_len != 0 or pred_len % seg_len != 0:
            for s in [seg_len, 24, 16, 12, 8, 4, 2, 1]:
                if seq_len % s == 0 and pred_len % s == 0:
                    seg_len = s
                    break
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_vars = n_vars
        self.d_model = d_model
        self.seg_len = seg_len
        self.seg_num_x = seq_len // seg_len
        self.seg_num_y = pred_len // seg_len

        self.value_embedding = nn.Sequential(
            nn.Linear(seg_len, d_model),
            nn.ReLU(),
        )
        self.rnn = nn.GRU(
            input_size=d_model, hidden_size=d_model,
            num_layers=1, batch_first=True, bidirectional=False,
        )
        # Positional + channel embedding for each future segment / variate.
        self.pos_emb = nn.Parameter(torch.randn(self.seg_num_y, d_model // 2))
        self.channel_emb = nn.Parameter(torch.randn(n_vars, d_model // 2))

        self.predict = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, seg_len),
        )

    def forward(self, x_enc: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        # x_enc: (B, L, N)
        B = x_enc.size(0)

        # last-value subtraction normalisation (SegRNN-original)
        seq_last = x_enc[:, -1:, :].detach()
        x = (x_enc - seq_last).permute(0, 2, 1)             # (B, N, L)

        # segment + embed: (B, N, L) -> (B*N, seg_num_x, seg_len) -> (B*N, seg_num_x, d_model)
        x = self.value_embedding(x.reshape(-1, self.seg_num_x, self.seg_len))

        # encode
        _, hn = self.rnn(x)                                 # hn: (1, B*N, d_model)

        # build per-future-segment positional+channel embedding,
        # shape (B * N * seg_num_y, 1, d_model)
        pos = torch.cat(
            [
                self.pos_emb.unsqueeze(0).repeat(self.n_vars, 1, 1),
                self.channel_emb.unsqueeze(1).repeat(1, self.seg_num_y, 1),
            ],
            dim=-1,
        ).view(-1, 1, self.d_model).repeat(B, 1, 1)

        # decode each future segment, using the encoder hidden state, expanded
        h0 = hn.repeat(1, 1, self.seg_num_y).view(1, -1, self.d_model)
        _, hy = self.rnn(pos, h0)                           # hy: (1, B*N*seg_num_y, d_model)

        # project back to seg_len, reshape to (B, N, pred_len), permute to (B, pred_len, N)
        y = self.predict(hy).view(-1, self.n_vars, self.pred_len)
        y = y.permute(0, 2, 1) + seq_last
        return y
