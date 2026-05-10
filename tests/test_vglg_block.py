"""Unit tests for the VGLG block: shapes, gradient flow, gate properties."""
from __future__ import annotations

import pytest
import torch

from src.models.vglg.block import VGLGBlock
from src.models.vglg.gate import VariateGate, gate_entropy
from src.models.vglg import VGLG_CNN, VGLG_MLP, VGLG_Transformer


def test_block_shape():
    block = VGLGBlock(time_dim=96, n_vars=21, kernel_size=31, rank=8)
    x = torch.randn(4, 96, 21)
    y, g = block(x)
    assert y.shape == x.shape, f"got {y.shape}, expected {x.shape}"
    assert g.shape == (4, 1, 21)


def test_block_gradient_flow():
    block = VGLGBlock(time_dim=96, n_vars=7, kernel_size=15, rank=4)
    x = torch.randn(2, 96, 7, requires_grad=True)
    y, _ = block(x)
    loss = y.sum()
    loss.backward()
    for name, p in block.named_parameters():
        assert p.grad is not None, f"No grad for {name}"
        assert not torch.isnan(p.grad).any(), f"NaN grad in {name}"


def test_gate_in_unit_interval():
    gate = VariateGate()
    x = torch.randn(8, 96, 21) * 10
    g = gate(x)
    assert g.shape == (8, 1, 21)
    assert (g > 0).all() and (g < 1).all()


def test_gate_entropy_in_range():
    g = torch.full((4, 1, 7), 0.5)
    h = gate_entropy(g)
    # Bernoulli(0.5) entropy in nats is ln(2) ~ 0.693
    assert torch.isclose(h, torch.tensor(0.693147), atol=1e-3)


@pytest.mark.parametrize("cls", [VGLG_MLP, VGLG_CNN, VGLG_Transformer])
def test_wrapper_forward(cls):
    model = cls(seq_len=96, pred_len=24, n_vars=7, d_model=64, n_layers=2)
    x = torch.randn(3, 96, 7)
    y = model(x)
    assert y.shape == (3, 24, 7), f"{cls.__name__}: got {y.shape}"
    # last gates recorded
    assert len(model._last_gates) == 2
