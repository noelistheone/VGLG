"""DataLoader factory. Build train/val/test loaders from a flat config dict."""
from __future__ import annotations

from typing import Any

from torch.utils.data import DataLoader

from .datasets import build_dataset


def build_dataloaders(
    data_cfg: Any,
    train_cfg: Any,
) -> dict[str, DataLoader]:
    """Returns dict with keys 'train', 'val', 'test'.

    `data_cfg` and `train_cfg` are duck-typed (Hydra DictConfig or dataclass).
    """
    common = dict(
        data_kind=data_cfg.data_kind,
        root_path=data_cfg.root_path,
        data_path=data_cfg.data_path,
        seq_len=train_cfg.seq_len,
        label_len=train_cfg.label_len,
        pred_len=train_cfg.pred_len,
        features=data_cfg.features,
        target=data_cfg.target,
        freq=data_cfg.freq,
    )
    loaders: dict[str, DataLoader] = {}
    for flag in ("train", "val", "test"):
        ds = build_dataset(flag=flag, **common)
        loaders[flag] = DataLoader(
            ds,
            batch_size=train_cfg.batch_size,
            shuffle=(flag == "train"),
            num_workers=train_cfg.num_workers,
            pin_memory=True,
            drop_last=(flag == "train"),
        )
    return loaders
