"""DistillDataset: wraps a forecasting Dataset and pairs each window with the
cached Chronos teacher prediction at the same index.

Cache file is built by `scripts/cache_teacher_predictions.py` and shaped
(n_windows, pred_len, n_vars). Index alignment is positional: window `i` from
the underlying dataset maps to teacher prediction `i`.

Returned tuple per __getitem__:
    (seq_x, seq_y, seq_x_mark, seq_y_mark, teacher_pred)
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

from .datasets import _ForecastDatasetBase


class DistillDataset(Dataset):
    def __init__(self, base: _ForecastDatasetBase, teacher_cache_path: str | Path):
        self.base = base
        self.teacher = torch.load(teacher_cache_path, weights_only=True)
        # Sanity: teacher length must match dataset length
        if self.teacher.shape[0] != len(base):
            raise ValueError(
                f"Teacher cache length {self.teacher.shape[0]} != dataset length "
                f"{len(base)} for {teacher_cache_path}. Re-run cache_teacher_predictions.py."
            )

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        seq_x, seq_y, seq_x_mark, seq_y_mark = self.base[i]
        return seq_x, seq_y, seq_x_mark, seq_y_mark, self.teacher[i]
