"""TokenMixer registry. Adding a new mixer = drop a file + add to MIXER_REGISTRY."""
from __future__ import annotations

from typing import Any

import torch.nn as nn

from .attn_mixer import AttnMixer
from .conv_mixer import ConvMixer
from .mlp_mixer import MLPMixer
from .vglg_mixer import VariateGate, VGLGMixer

MIXER_REGISTRY: dict[str, type[nn.Module]] = {
    "mlp": MLPMixer,
    "conv": ConvMixer,
    "attn": AttnMixer,
    "vglg": VGLGMixer,
}


def build_mixer(cfg: Any, seq_len: int, n_vars: int) -> nn.Module:
    """`cfg` may be a DictConfig, dict, or SimpleNamespace with a `type` field."""
    if hasattr(cfg, "items"):
        items = dict(cfg.items())
    elif hasattr(cfg, "__dict__"):
        items = vars(cfg).copy()
    else:
        raise TypeError(f"Cannot build mixer from cfg of type {type(cfg)}")
    mixer_type = items.pop("type")
    if mixer_type not in MIXER_REGISTRY:
        raise ValueError(f"Unknown mixer type: {mixer_type}. Known: {sorted(MIXER_REGISTRY)}")
    cls = MIXER_REGISTRY[mixer_type]
    return cls(seq_len=seq_len, n_vars=n_vars, **items)


__all__ = [
    "MIXER_REGISTRY", "build_mixer",
    "MLPMixer", "ConvMixer", "AttnMixer", "VGLGMixer", "VariateGate",
]
