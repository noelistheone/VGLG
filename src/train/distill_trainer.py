"""Distillation trainer: same loop as src/train/trainer.py, but the training
loss is `MSE(student, target) + λ_trend·trend_kd + λ_freq·freq_kd + λ_diff·diff_kd`
where the KD terms compare the student to a cached Chronos teacher.

Validation and test still use plain MSE (no teacher needed at eval time).

Usage (Hydra):
    python -m src.train.distill_trainer model=metatsf_vglg data=ettm1 \
        train.pred_len=96 train.train_epochs=10 seed=2021 tag=distill
"""
from __future__ import annotations

import time
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from src.data.data_provider import build_dataloaders
from src.data.datasets import build_dataset
from src.data.distill_dataset import DistillDataset
from src.losses.distill import kd_loss_bundle
from src.models import build_model
from src.train.trainer import evaluate
from src.utils import EarlyStopping, seed_everything

try:
    import wandb
    _HAS_WANDB = True
except ImportError:
    _HAS_WANDB = False


def _adjust_lr(epoch: int, base_lr: float, scheme: str) -> float:
    if scheme == "type1":
        return base_lr * (0.5 ** ((epoch - 1) // 1))
    if scheme == "type3":
        return base_lr * 0.5 * (1 + np.cos(np.pi * epoch / 10))
    return base_lr


def _build_train_with_teacher(cfg: DictConfig) -> DataLoader:
    """Build the train DataLoader, but each batch carries the teacher prediction."""
    base = build_dataset(
        data_kind=cfg.data.data_kind,
        flag="train",
        root_path=cfg.data.root_path,
        data_path=cfg.data.data_path,
        seq_len=cfg.train.seq_len,
        label_len=cfg.train.label_len,
        pred_len=cfg.train.pred_len,
        features=cfg.data.features,
        target=cfg.data.target,
        freq=cfg.data.freq,
    )
    cache_path = (
        Path(cfg.train.teacher_cache_dir)
        / f"{cfg.data.name}_h{cfg.train.pred_len}_train.pt"
    )
    ds = DistillDataset(base, cache_path)
    return DataLoader(
        ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
        drop_last=True,
    )


def train(cfg: DictConfig) -> dict[str, float]:
    seed_everything(cfg.seed)
    device = cfg.device if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 60)

    run_name = (
        cfg.run_name
        or f"{cfg.data.name}_{cfg.model.name}_kd_h{cfg.train.pred_len}_s{cfg.seed}"
    )

    use_wandb = bool(getattr(cfg.train, "use_wandb", False)) and _HAS_WANDB
    if use_wandb:
        wandb.init(
            project=cfg.train.wandb_project, name=run_name,
            tags=[cfg.tag, cfg.data.name, cfg.model.name, "distill"],
            config=OmegaConf.to_container(cfg, resolve=True), reinit=True,
        )

    # Train loader has teacher predictions baked in; val/test are vanilla.
    train_loader = _build_train_with_teacher(cfg)
    vanilla = build_dataloaders(cfg.data, cfg.train)
    val_loader, test_loader = vanilla["val"], vanilla["test"]
    print(f"Loaded data | train={len(train_loader.dataset)} "
          f"val={len(val_loader.dataset)} test={len(test_loader.dataset)}")

    model = build_model(cfg.model, cfg.data, cfg.train).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Built model: {cfg.model.name} | params={n_params:,}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.train.learning_rate, weight_decay=cfg.train.weight_decay,
    )
    mse = nn.MSELoss()

    ckpt_path = Path(cfg.train.checkpoint_dir) / cfg.tag / f"{run_name}.pt"
    es = EarlyStopping(patience=cfg.train.patience)

    use_amp = cfg.train.use_amp and device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    l_mse_w = cfg.train.distill_lambda_mse
    l_tre_w = cfg.train.distill_lambda_trend
    l_frq_w = cfg.train.distill_lambda_freq
    l_dif_w = cfg.train.distill_lambda_diff
    warmup = cfg.train.distill_warmup_epochs

    for epoch in range(1, cfg.train.train_epochs + 1):
        model.train()
        t0 = time.time()
        running = 0.0
        running_kd = {"trend": 0.0, "freq": 0.0, "diff": 0.0}
        n_batches = 0
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, t_pred) in enumerate(train_loader):
            batch_x = batch_x.float().to(device, non_blocking=True)
            batch_y = batch_y.float().to(device, non_blocking=True)
            batch_x_mark = batch_x_mark.float().to(device, non_blocking=True)
            batch_y_mark = batch_y_mark.float().to(device, non_blocking=True)
            t_pred = t_pred.float().to(device, non_blocking=True)

            dec_inp = torch.zeros_like(batch_y[:, -cfg.train.pred_len:, :])
            dec_inp = torch.cat(
                [batch_y[:, : batch_y.size(1) - cfg.train.pred_len, :], dec_inp], dim=1
            )

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                out = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                out = out[:, -cfg.train.pred_len:, :]
                target = batch_y[:, -cfg.train.pred_len:, :]
                loss_mse = mse(out, target)
                if epoch <= warmup:
                    loss = loss_mse
                    l_tre = l_frq = l_dif = torch.tensor(0.0, device=device)
                else:
                    l_tre, l_frq, l_dif = kd_loss_bundle(out, t_pred)
                    loss = l_mse_w * loss_mse + l_tre_w * l_tre + l_frq_w * l_frq + l_dif_w * l_dif

            scaler.scale(loss).backward()
            if cfg.train.gradient_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.gradient_clip)
            scaler.step(optimizer)
            scaler.update()

            running += loss.item()
            running_kd["trend"] += l_tre.item()
            running_kd["freq"] += l_frq.item()
            running_kd["diff"] += l_dif.item()
            n_batches += 1
            if (i + 1) % cfg.train.log_interval == 0:
                print(f"  epoch {epoch} batch {i+1}/{len(train_loader)} "
                      f"loss={running / n_batches:.6f} "
                      f"trend={running_kd['trend']/n_batches:.4f} "
                      f"freq={running_kd['freq']/n_batches:.4f} "
                      f"diff={running_kd['diff']/n_batches:.4f}")

        train_loss = running / max(1, n_batches)
        val_metrics = evaluate(model, val_loader, device, cfg.train.pred_len)
        elapsed = time.time() - t0
        print(f"Epoch {epoch} | train_loss={train_loss:.6f} "
              f"val_mse={val_metrics['mse']:.6f} val_mae={val_metrics['mae']:.6f} "
              f"({elapsed:.1f}s)")

        if use_wandb:
            wandb.log({
                "epoch": epoch, "train/loss": train_loss,
                "train/kd_trend": running_kd["trend"] / max(1, n_batches),
                "train/kd_freq": running_kd["freq"] / max(1, n_batches),
                "train/kd_diff": running_kd["diff"] / max(1, n_batches),
                "val/mse": val_metrics["mse"], "val/mae": val_metrics["mae"],
                "lr": optimizer.param_groups[0]["lr"], "epoch_time_s": elapsed,
            })

        es(val_metrics["mse"], model, ckpt_path)
        if es.early_stop:
            print(f"Early stopping at epoch {epoch}.")
            break

        new_lr = _adjust_lr(epoch, cfg.train.learning_rate, cfg.train.lradj)
        for g in optimizer.param_groups:
            g["lr"] = new_lr

    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    test_metrics = evaluate(model, test_loader, device, cfg.train.pred_len)
    print(f"\nTest | mse={test_metrics['mse']:.6f} mae={test_metrics['mae']:.6f} "
          f"rmse={test_metrics['rmse']:.6f}")
    if use_wandb:
        wandb.log({f"test/{k}": v for k, v in test_metrics.items()})
        wandb.summary["n_parameters"] = n_params
        wandb.finish()
    return {"n_params": n_params, **{f"test_{k}": v for k, v in test_metrics.items()}}


@hydra.main(config_path="../../configs", config_name="distill_default", version_base=None)
def main(cfg: DictConfig) -> None:
    train(cfg)


if __name__ == "__main__":
    main()
