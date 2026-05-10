"""Unit tests for the four MetaTSF TokenMixers.

Each mixer must:
  - preserve input shape (B, L, N) exactly
  - have all parameters receive non-NaN gradients
  - VGLG additionally must keep the gate strictly inside (0, 1)
"""
from __future__ import annotations

import pytest
import torch

from src.models.metatsf.mixers import MIXER_REGISTRY


@pytest.mark.parametrize("name", sorted(MIXER_REGISTRY))
def test_mixer_shape_preserved(name: str):
    cls = MIXER_REGISTRY[name]
    mixer = cls(seq_len=128, n_vars=21)
    x = torch.randn(4, 128, 21)
    y = mixer(x)
    assert y.shape == x.shape, f"{name}: {y.shape} != {x.shape}"


@pytest.mark.parametrize("name", sorted(MIXER_REGISTRY))
def test_mixer_gradients(name: str):
    cls = MIXER_REGISTRY[name]
    mixer = cls(seq_len=64, n_vars=7)
    x = torch.randn(2, 64, 7, requires_grad=True)
    mixer(x).sum().backward()
    for pname, p in mixer.named_parameters():
        assert p.grad is not None, f"{name}/{pname}: no grad"
        assert not torch.isnan(p.grad).any(), f"{name}/{pname}: NaN grad"


def test_vglg_gate_in_unit_interval():
    from src.models.metatsf.mixers.vglg_mixer import VariateGate
    gate = VariateGate()
    x = torch.randn(8, 96, 21) * 10.0
    g = gate(x)
    assert g.shape == (8, 1, 21)
    assert (g > 0).all() and (g < 1).all()


def test_attn_mixer_handles_odd_seq_len():
    """seq_len=126 isn't divisible by default n_heads=4 — mixer must auto-adjust."""
    cls = MIXER_REGISTRY["attn"]
    mixer = cls(seq_len=126, n_vars=7, n_heads=4)
    x = torch.randn(2, 126, 7)
    y = mixer(x)
    assert y.shape == x.shape
