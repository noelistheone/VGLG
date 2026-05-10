"""Smoke tests across all registered models on synthetic data."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from src.models import build_model

MODELS = [
    ("dlinear", {"name": "dlinear", "type": "linear", "individual": False, "moving_avg": 25}),
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
    ("vglg_mlp", {
        "name": "vglg_mlp", "type": "mlp",
        "d_model": 64, "n_layers": 2, "kernel_size": 15, "rank": 4,
        "channel_mlp_ratio": 2, "dropout": 0.0,
        "revin": True, "affine": True, "gate_hidden": 8, "gate_entropy_lambda": 0.0,
    }),
    ("vglg_cnn", {
        "name": "vglg_cnn", "type": "cnn",
        "d_model": 64, "n_layers": 2, "kernel_size": 15, "rank": 4,
        "ffn_ratio": 2, "dropout": 0.0,
        "revin": True, "affine": True, "gate_hidden": 8, "gate_entropy_lambda": 0.0,
    }),
    ("vglg_transformer", {
        "name": "vglg_transformer", "type": "transformer",
        "d_model": 64, "n_layers": 2, "n_heads": 4, "kernel_size": 15, "rank": 4,
        "dropout": 0.0, "attn_dropout": 0.0,
        "revin": True, "affine": True, "gate_hidden": 8, "gate_entropy_lambda": 0.0,
    }),
]


@pytest.mark.parametrize("name,model_cfg", MODELS)
def test_model_forward(name, model_cfg):
    data_cfg = SimpleNamespace(n_vars=7)
    train_cfg = SimpleNamespace(seq_len=96, pred_len=24)
    model = build_model(SimpleNamespace(**model_cfg), data_cfg, train_cfg)
    x = torch.randn(2, 96, 7)
    y = model(x)
    assert y.shape == (2, 24, 7), f"{name}: got {y.shape}"

    loss = y.mean()
    loss.backward()
    for pname, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"{name}: no grad for {pname}"
