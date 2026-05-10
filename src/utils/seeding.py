"""Reproducible seeding across numpy / random / torch (incl. CUDA)."""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Trade exact determinism for speed; flip these if a study requires it.
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
