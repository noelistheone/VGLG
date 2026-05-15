"""DistillDataset: wraps a forecasting Dataset and pairs each window with the
cached Chronos teacher prediction at the same index.

Cache file shape: (n_windows, pred_len, n_vars). Index alignment is positional.

Two storage formats supported:
  - `.pt`  (torch.save, fp32) — used for ETTm caches; loaded fully into RAM
  - `.npy` (numpy fp16) — used for Electricity/Traffic h=720 where the full
    fp32 tensor (30 GB) would OOM the 31 GB system RAM. Loaded with
    `np.load(..., mmap_mode='r')` so only the windows actually accessed are
    paged in. Each `__getitem__` upcasts the slice to fp32.

Returned tuple per __getitem__:
    (seq_x, seq_y, seq_x_mark, seq_y_mark, teacher_pred)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .datasets import _ForecastDatasetBase


def _resolve_cache_path(p: str | Path) -> Path:
    """Pick whichever cache format exists on disk. Order of preference:
    requested path first, then .npy (mmap), then .pt."""
    p = Path(p)
    if p.exists():
        return p
    base = p.with_suffix("")
    for ext in (".npy", ".pt"):
        cand = base.with_suffix(ext)
        if cand.exists():
            return cand
    raise FileNotFoundError(f"Teacher cache not found: {p} (also tried {base}.npy/.pt)")


class DistillDataset(Dataset):
    def __init__(self, base: _ForecastDatasetBase, teacher_cache_path: str | Path):
        self.base = base
        cache_path = _resolve_cache_path(teacher_cache_path)
        self.cache_path = cache_path
        if cache_path.suffix == ".npy":
            self.teacher = np.load(cache_path, mmap_mode="r")
            self.is_numpy = True
            n = self.teacher.shape[0]
        else:
            self.teacher = torch.load(cache_path, weights_only=True)
            self.is_numpy = False
            n = self.teacher.shape[0]
        if n != len(base):
            raise ValueError(
                f"Teacher cache length {n} != dataset length {len(base)} for "
                f"{cache_path}. Re-run cache_teacher_predictions.py."
            )

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        seq_x, seq_y, seq_x_mark, seq_y_mark = self.base[i]
        if self.is_numpy:
            # mmap slice -> contiguous fp32 tensor (small per-sample copy)
            t_pred = torch.from_numpy(np.ascontiguousarray(self.teacher[i])).float()
        else:
            t_pred = self.teacher[i]
        return seq_x, seq_y, seq_x_mark, seq_y_mark, t_pred
