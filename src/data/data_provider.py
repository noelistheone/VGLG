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

    When `train_cfg.gpu_resident_data` is True, datasets store their tensors on
    cuda directly, num_workers is forced to 0 (CUDA tensors cannot cross worker
    process boundaries), and pin_memory is disabled (already on GPU).
    """
    gpu_resident = bool(getattr(train_cfg, "gpu_resident_data", False))
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
        gpu_resident=gpu_resident,
    )
    num_workers = 0 if gpu_resident else train_cfg.num_workers
    pin_memory = not gpu_resident
    loaders: dict[str, DataLoader] = {}
    for flag in ("train", "val", "test"):
        ds = build_dataset(flag=flag, **common)
        loaders[flag] = DataLoader(
            ds,
            batch_size=train_cfg.batch_size,
            shuffle=(flag == "train"),
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=(flag == "train"),
        )
    return loaders
