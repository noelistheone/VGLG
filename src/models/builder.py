"""Build a model from a Hydra config dict.

Adding a new baseline:
  1. Drop a class into src/models/baselines/<name>.py
  2. Register it in MODEL_REGISTRY below
  3. Add a config at configs/model/<name>.yaml

Adding a new MetaTSF mixer:
  1. Drop a class into src/models/metatsf/mixers/<name>_mixer.py
  2. Register it in src/models/metatsf/mixers/__init__.py:MIXER_REGISTRY
  3. Add a config at configs/model/metatsf_<name>.yaml that sets `mixer.type: <name>`
"""
from __future__ import annotations

from typing import Any

import torch.nn as nn

from .baselines import (
    DLinear,
    GRUForecaster,
    LSTMForecaster,
    ModernTCN,
    PatchTST,
    SegRNN,
    TimeMixer,
    iTransformer,
)
from .metatsf import MetaTSF

# Models that take a flat kwargs interface
MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    # baselines
    "dlinear": DLinear,
    "lstm": LSTMForecaster,
    "gru": GRUForecaster,
    "segrnn": SegRNN,
    "timemixer": TimeMixer,
    "moderntcn": ModernTCN,
    "itransformer": iTransformer,
    "patchtst": PatchTST,
    # ours: every metatsf_* config maps to MetaTSF with a different mixer.type
    "metatsf_mlp": MetaTSF,
    "metatsf_conv": MetaTSF,
    "metatsf_attn": MetaTSF,
    "metatsf_vglg": MetaTSF,
}


def _to_dict(cfg: Any) -> dict:
    if hasattr(cfg, "items"):
        return dict(cfg)
    if hasattr(cfg, "__dict__"):
        return vars(cfg).copy()
    raise TypeError(f"Cannot iterate cfg of type {type(cfg)}")


def build_model(model_cfg: Any, data_cfg: Any, train_cfg: Any) -> nn.Module:
    name = model_cfg.name
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. Registered: {sorted(MODEL_REGISTRY)}.\n"
            "If this is a TSlib baseline, port it into src/models/baselines/ first."
        )
    cls = MODEL_REGISTRY[name]

    kwargs = dict(
        seq_len=train_cfg.seq_len,
        pred_len=train_cfg.pred_len,
        n_vars=data_cfg.n_vars,
    )
    extras = {k: v for k, v in _to_dict(model_cfg).items() if k not in ("name", "type")}
    kwargs.update(extras)
    return cls(**kwargs)
