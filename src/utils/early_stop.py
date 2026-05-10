"""Early stopping by validation loss, with checkpointing."""
from __future__ import annotations

from pathlib import Path

import torch


class EarlyStopping:
    def __init__(self, patience: int = 3, delta: float = 0.0, verbose: bool = True):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_score: float | None = None
        self.best_loss: float = float("inf")
        self.early_stop = False

    def __call__(self, val_loss: float, model: torch.nn.Module, ckpt_path: str | Path) -> None:
        score = -val_loss
        if self.best_score is None or score > self.best_score + self.delta:
            self.best_score = score
            self.best_loss = val_loss
            self._save(model, ckpt_path)
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"  EarlyStopping: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

    def _save(self, model: torch.nn.Module, ckpt_path: str | Path) -> None:
        ckpt_path = Path(ckpt_path)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), ckpt_path)
        if self.verbose:
            print(f"  Validation improved to {self.best_loss:.6f}; saved checkpoint.")
