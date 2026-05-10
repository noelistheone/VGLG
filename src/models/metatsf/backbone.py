"""MetaTSF backbone: shared across all four TokenMixer variants.

Pipeline (over input of shape (B, L, N)):
    RevIN(norm)
    -> input projection L -> d_model along the time axis
    -> n_layers x MetaTSFBlock (Norm -> TokenMixer -> Norm -> ChannelMLP)
    -> output projection d_model -> pred_len
    -> RevIN(denorm)

Only the `mixer.type` field of the config changes between variants. Everything
else (depth, width, optimiser, schedule) is held constant for fair comparison.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from ..layers.revin import RevIN
from .block import MetaTSFBlock
from .mixers.vglg_mixer import VGLGMixer


class MetaTSF(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        n_vars: int,
        d_model: int = 128,
        n_layers: int = 2,
        dropout: float = 0.1,
        channel_mlp_mult: int = 2,
        revin: bool = True,
        affine: bool = True,
        mixer: Any = None,
        **_unused,
    ):
        super().__init__()
        assert mixer is not None, "MetaTSF requires a mixer config (e.g. {type: vglg, ...})"
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_vars = n_vars
        self.d_model = d_model
        self.n_layers = n_layers
        self.use_revin = revin
        if revin:
            self.revin = RevIN(n_vars, affine=affine)

        self.input_proj = nn.Linear(seq_len, d_model)
        self.blocks = nn.ModuleList([
            MetaTSFBlock(
                seq_len=d_model,            # within-block "time" dim is d_model
                n_vars=n_vars,
                mixer_cfg=mixer,
                channel_mlp_mult=channel_mlp_mult,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])
        self.output_proj = nn.Linear(d_model, pred_len)

    def forward(self, x_enc: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if self.use_revin:
            x = self.revin(x_enc, mode="norm")
        else:
            x = x_enc
        # (B, L, N) -> (B, N, L) -> Linear -> (B, N, d_model) -> (B, d_model, N)
        h = self.input_proj(x.transpose(1, 2)).transpose(1, 2)
        for block in self.blocks:
            h = block(h)
        out = self.output_proj(h.transpose(1, 2)).transpose(1, 2)
        if self.use_revin:
            out = self.revin(out, mode="denorm")
        return out

    def gate_entropy_loss(self) -> torch.Tensor:
        """Sum of gate-entropy regularisers from any VGLG mixers in the stack.

        Returns a 0-d tensor; safe to add to the main loss unconditionally.
        """
        total = torch.tensor(0.0)
        for blk in self.blocks:
            mx = blk.mixer
            if isinstance(mx, VGLGMixer) and mx.gate_entropy_reg != 0.0:
                term = mx.gate_entropy_loss()
                if isinstance(term, torch.Tensor) and term.numel() > 0:
                    total = total.to(term.device) + term
        return total
