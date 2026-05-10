"""End-to-end forward/backward tests for the MetaTSF backbone with each mixer
and for the eight registered baselines on a small synthetic input.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from src.models import build_model

# Each entry: (model name, model_cfg fields)
BASELINE_CFGS = [
    ("dlinear", {"name": "dlinear", "type": "linear", "individual": False, "moving_avg": 25}),
    ("lstm", {"name": "lstm", "type": "rnn", "d_model": 32, "e_layers": 1, "dropout": 0.0}),
    ("gru", {"name": "gru", "type": "rnn", "d_model": 32, "e_layers": 1, "dropout": 0.0}),
    ("segrnn", {"name": "segrnn", "type": "rnn", "d_model": 64, "seg_len": 24, "dropout": 0.0}),
    ("itransformer", {
        "name": "itransformer", "type": "transformer",
        "d_model": 64, "d_ff": 64, "e_layers": 1, "n_heads": 4,
        "dropout": 0.0, "activation": "gelu", "use_norm": True,
    }),
    ("patchtst", {
        "name": "patchtst", "type": "transformer",
        "d_model": 64, "d_ff": 64, "e_layers": 1, "n_heads": 4,
        "patch_len": 16, "stride": 8,
        "dropout": 0.0, "head_dropout": 0.0, "activation": "gelu", "use_norm": True,
    }),
    ("timemixer", {
        "name": "timemixer", "type": "mlp",
        "d_model": 16, "d_ff": 32, "e_layers": 1,
        "down_sampling_layers": 2, "down_sampling_window": 2, "down_sampling_method": "avg",
        "decomp_method": "moving_avg", "moving_avg": 25, "top_k": 5,
        "dropout": 0.0, "use_norm": 1,
    }),
    ("moderntcn", {
        "name": "moderntcn", "type": "cnn",
        "patch_size": 16, "patch_stride": 8, "ffn_ratio": 1, "num_blocks": 1,
        "large_size": 15, "small_size": 5, "d_model": 32,
        "backbone_dropout": 0.0, "head_dropout": 0.0,
        "revin": True, "affine": False,
    }),
]


def _metatsf_cfg(name: str, mixer_type: str, **mixer_extras) -> dict:
    return {
        "name": name,
        "type": mixer_type,
        "d_model": 64, "n_layers": 2, "dropout": 0.0,
        "channel_mlp_mult": 2, "revin": True, "affine": True,
        "mixer": {"type": mixer_type, **mixer_extras},
    }


METATSF_CFGS = [
    ("metatsf_mlp", _metatsf_cfg("metatsf_mlp", "mlp", hidden_mult=2, dropout=0.0)),
    ("metatsf_conv", _metatsf_cfg("metatsf_conv", "conv", kernel_size=15, dropout=0.0)),
    ("metatsf_attn", _metatsf_cfg("metatsf_attn", "attn", n_heads=4, dropout=0.0)),
    ("metatsf_vglg", _metatsf_cfg("metatsf_vglg", "vglg", kernel_size=15, rank=4, dropout=0.0)),
]

ALL_CFGS = BASELINE_CFGS + METATSF_CFGS


@pytest.mark.parametrize("name,cfg", ALL_CFGS)
def test_model_forward_and_backward(name: str, cfg: dict):
    data_cfg = SimpleNamespace(n_vars=7)
    train_cfg = SimpleNamespace(seq_len=96, pred_len=24)
    # convert nested mixer dict to SimpleNamespace for the duck-typed builder
    if "mixer" in cfg:
        cfg = {**cfg, "mixer": SimpleNamespace(**cfg["mixer"])}
    model = build_model(SimpleNamespace(**cfg), data_cfg, train_cfg)
    x = torch.randn(2, 96, 7)
    y = model(x)
    assert y.shape == (2, 24, 7), f"{name}: got {y.shape}"

    y.mean().backward()
    for pname, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"{name}: no grad for {pname}"
