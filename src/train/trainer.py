"""Hydra-driven training entry point.

Usage:
    python -m src.train.trainer                           # default config
    python -m src.train.trainer model=vglg_mlp data=etth1 train.train_epochs=2
    python -m src.train.trainer -m model=dlinear,vglg_mlp data=etth1,etth2  # sweep
"""
from __future__ import annotations

import time
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from src.data import build_dataloaders
from src.models import build_model
from src.utils import EarlyStopping, metric, seed_everything

try:
    import wandb
    _HAS_WANDB = True
except ImportError:  # pragma: no cover
    _HAS_WANDB = False


def _adjust_lr(epoch: int, base_lr: float, scheme: str, total_epochs: int = 10) -> float:
    if scheme == "type1":
        return base_lr * (0.5 ** ((epoch - 1) // 1))
    if scheme == "type3":
        # cosine over [0, total_epochs]. Without total_epochs the curve would wrap
        # at epoch 10 and reflect back up — wrong for >10-epoch budgets.
        return base_lr * 0.5 * (1 + np.cos(np.pi * epoch / total_epochs))
    if scheme == "constant":
        return base_lr
    return base_lr


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    device: str,
    pred_len: int,
) -> dict[str, float]:
    model.eval()
    preds, trues = [], []
    for batch_x, batch_y, batch_x_mark, batch_y_mark in loader:
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        batch_x_mark = batch_x_mark.float().to(device)
        batch_y_mark = batch_y_mark.float().to(device)

        # decoder placeholder for autoregressive baselines
        dec_inp = torch.zeros_like(batch_y[:, -pred_len:, :])
        dec_inp = torch.cat([batch_y[:, : batch_y.size(1) - pred_len, :], dec_inp], dim=1)

        out = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        out = out[:, -pred_len:, :]
        target = batch_y[:, -pred_len:, :]
        preds.append(out.cpu().numpy())
        trues.append(target.cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    return metric(preds, trues)


def train(cfg: DictConfig) -> dict[str, float]:
    seed_everything(cfg.seed)
    device = cfg.device if torch.cuda.is_available() else "cpu"
    # TF32 matmul on Ampere — opt-in via train.matmul_precision. Default 'highest'
    # preserves the prior behaviour exactly.
    mp = getattr(cfg.train, "matmul_precision", "highest")
    if mp in ("high", "medium"):
        torch.set_float32_matmul_precision(mp)

    print("=" * 60)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 60)

    run_name = cfg.run_name or f"{cfg.data.name}_{cfg.model.name}_h{cfg.train.pred_len}_s{cfg.seed}"

    use_wandb = bool(getattr(cfg.train, "use_wandb", False)) and _HAS_WANDB
    if use_wandb:
        wandb.init(
            project=cfg.train.wandb_project,
            name=run_name,
            tags=[cfg.tag, cfg.data.name, cfg.model.name],
            config=OmegaConf.to_container(cfg, resolve=True),
            reinit=True,
        )

    loaders = build_dataloaders(cfg.data, cfg.train)
    print(f"Loaded data | train={len(loaders['train'].dataset)} "
          f"val={len(loaders['val'].dataset)} test={len(loaders['test'].dataset)}")

    model = build_model(cfg.model, cfg.data, cfg.train).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Built model: {cfg.model.name} | params={n_params:,}")
    if use_wandb:
        wandb.config.update({"n_parameters": n_params}, allow_val_change=True)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
    )
    criterion = nn.MSELoss() if cfg.train.loss == "mse" else nn.L1Loss()

    ckpt_path = Path(cfg.train.checkpoint_dir) / cfg.tag / f"{run_name}.pt"
    es = EarlyStopping(patience=cfg.train.patience)

    use_amp = cfg.train.use_amp and device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    for epoch in range(1, cfg.train.train_epochs + 1):
        model.train()
        t0 = time.time()
        running = 0.0
        n_batches = 0
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(loaders["train"]):
            batch_x = batch_x.float().to(device, non_blocking=True)
            batch_y = batch_y.float().to(device, non_blocking=True)
            batch_x_mark = batch_x_mark.float().to(device, non_blocking=True)
            batch_y_mark = batch_y_mark.float().to(device, non_blocking=True)

            dec_inp = torch.zeros_like(batch_y[:, -cfg.train.pred_len:, :])
            dec_inp = torch.cat(
                [batch_y[:, : batch_y.size(1) - cfg.train.pred_len, :], dec_inp], dim=1
            )

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                out = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                out = out[:, -cfg.train.pred_len:, :]
                target = batch_y[:, -cfg.train.pred_len:, :]
                loss = criterion(out, target)

            scaler.scale(loss).backward()
            if cfg.train.gradient_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.gradient_clip)
            scaler.step(optimizer)
            scaler.update()

            running += loss.item()
            n_batches += 1
            if (i + 1) % cfg.train.log_interval == 0:
                print(f"  epoch {epoch} batch {i+1}/{len(loaders['train'])} "
                      f"loss={running / n_batches:.6f}")

        train_loss = running / max(1, n_batches)

        # Validation
        val_metrics = evaluate(model, loaders["val"], device, cfg.train.pred_len)
        elapsed = time.time() - t0
        print(f"Epoch {epoch} | train_loss={train_loss:.6f} "
              f"val_mse={val_metrics['mse']:.6f} val_mae={val_metrics['mae']:.6f} "
              f"({elapsed:.1f}s)")

        if use_wandb:
            wandb.log({
                "epoch": epoch,
                "train/loss": train_loss,
                "val/mse": val_metrics["mse"],
                "val/mae": val_metrics["mae"],
                "lr": optimizer.param_groups[0]["lr"],
                "epoch_time_s": elapsed,
            })

        es(val_metrics["mse"], model, ckpt_path)
        if es.early_stop:
            print(f"Early stopping at epoch {epoch}.")
            break

        # LR schedule
        new_lr = _adjust_lr(
            epoch, cfg.train.learning_rate, cfg.train.lradj, cfg.train.train_epochs
        )
        for g in optimizer.param_groups:
            g["lr"] = new_lr

    # Final test using best checkpoint
    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    test_metrics = evaluate(model, loaders["test"], device, cfg.train.pred_len)
    print(f"\nTest | mse={test_metrics['mse']:.6f} mae={test_metrics['mae']:.6f} "
          f"rmse={test_metrics['rmse']:.6f}")
    if use_wandb:
        wandb.log({f"test/{k}": v for k, v in test_metrics.items()})
        wandb.summary["n_parameters"] = n_params
        wandb.finish()
    return {"n_params": n_params, **{f"test_{k}": v for k, v in test_metrics.items()}}


@hydra.main(config_path="../../configs", config_name="default", version_base=None)
def main(cfg: DictConfig) -> None:
    train(cfg)


if __name__ == "__main__":
    main()
