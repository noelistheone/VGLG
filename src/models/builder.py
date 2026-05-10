"""Build a model from a Hydra config dict.

Adding a new baseline:
  1. Drop a class into src/models/baselines/<name>.py
  2. Register it in MODEL_REGISTRY below
  3. Add a config at configs/model/<name>.yaml
"""
from __future__ import annotations

from typing import Any

import torch.nn as nn

from .baselines import DLinear, ModernTCN, PatchTST, TimeMixer, iTransformer
from .vglg import VGLG_CNN, VGLG_MLP, VGLG_Transformer

MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    # baselines (Week 1-2 reproductions)
    "dlinear": DLinear,
    "itransformer": iTransformer,
    "patchtst": PatchTST,
    "timemixer": TimeMixer,
    "moderntcn": ModernTCN,
    # ours
    "vglg_mlp": VGLG_MLP,
    "vglg_cnn": VGLG_CNN,
    "vglg_transformer": VGLG_Transformer,
}


def build_model(model_cfg: Any, data_cfg: Any, train_cfg: Any) -> nn.Module:
    name = model_cfg.name
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. Registered: {sorted(MODEL_REGISTRY)}.\n"
            "If this is a TSlib baseline, port it into src/models/baselines/ first."
        )
    cls = MODEL_REGISTRY[name]

    # Common args every model accepts; extras are filtered via **_unused.
    kwargs = dict(
        seq_len=train_cfg.seq_len,
        pred_len=train_cfg.pred_len,
        n_vars=data_cfg.n_vars,
    )
    # Model-specific hyperparams. Accept DictConfig / dict / SimpleNamespace.
    if hasattr(model_cfg, "items"):
        items = model_cfg.items()
    elif hasattr(model_cfg, "__dict__"):
        items = vars(model_cfg).items()
    else:
        raise TypeError(f"Cannot iterate model_cfg of type {type(model_cfg)}")
    model_extras = {k: v for k, v in items if k not in ("name", "type")}
    kwargs.update(model_extras)
    return cls(**kwargs)
