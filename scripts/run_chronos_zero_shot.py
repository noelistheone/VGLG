"""Run Chronos-Bolt zero-shot on the test set of selected datasets/horizons.

Writes a log file in the same format as src/train/trainer.py so
update_results_table.py picks it up. Logs go to:
    logs/main/<dataset>_chronos_zs_h<horizon>_s2021.log

Chronos zero-shot has no learnable parameters and no seed dependence on the
test side (we use seed 2021 as a placeholder so the results table groups it
correctly).

Usage:
    python scripts/run_chronos_zero_shot.py --datasets ettm1 ettm2 --horizons 96 192 336 720
"""
from __future__ import annotations

import argparse
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import build_dataloaders
from src.models.teacher import ChronosTeacher
from src.utils import metric

DATASETS = ["ettm1", "ettm2"]
HORIZONS = [96, 192, 336, 720]


def _compose_cfg(dataset: str, horizon: int, batch_size: int):
    cfg = OmegaConf.create({"seed": 2021, "device": "cuda"})
    cfg.data = OmegaConf.load(ROOT / "configs" / "data" / f"{dataset}.yaml")
    cfg.train = OmegaConf.load(ROOT / "configs" / "train" / "default.yaml")
    cfg.train.pred_len = horizon
    cfg.train.batch_size = batch_size
    cfg.train.num_workers = 2
    return cfg


def evaluate_chronos(teacher: ChronosTeacher, dataset: str, horizon: int,
                     batch_size: int, device: str = "cuda") -> dict:
    cfg = _compose_cfg(dataset, horizon, batch_size)
    loaders = build_dataloaders(cfg.data, cfg.train)
    test_loader = loaders["test"]
    preds, trues = [], []
    n_seen = 0
    t0 = time.time()
    for i, (batch_x, batch_y, _, _) in enumerate(test_loader):
        batch_x = batch_x.float().to(device)
        target = batch_y[:, -horizon:, :].float().to(device)
        out = teacher.predict(batch_x, horizon)
        preds.append(out.cpu().numpy())
        trues.append(target.cpu().numpy())
        n_seen += batch_x.size(0)
        if (i + 1) % 5 == 0:
            print(f"    batch {i+1}/{len(test_loader)} ({n_seen} samples)  "
                  f"elapsed {time.time()-t0:.1f}s", flush=True)
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    return metric(preds, trues), n_seen, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--horizons", nargs="+", type=int, default=HORIZONS)
    ap.add_argument("--model", default="amazon/chronos-bolt-base")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--out-dir", default="logs/main")
    ap.add_argument("--seed-tag", default=2021,
                    help="placeholder seed in log filename for table grouping")
    args = ap.parse_args()

    log_dir = ROOT / args.out_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.model} ...", flush=True)
    teacher = ChronosTeacher(model_name=args.model, device="cuda", dtype=torch.bfloat16)
    print("Teacher ready.\n", flush=True)

    for dataset, horizon in product(args.datasets, args.horizons):
        log_path = log_dir / f"{dataset}_chronos_zs_h{horizon}_s{args.seed_tag}.log"
        print(f"=== {dataset} h={horizon} -> {log_path.name} ===", flush=True)
        metrics, n_seen, elapsed = evaluate_chronos(
            teacher, dataset, horizon, args.batch_size,
        )
        # Write a trainer-compatible log so update_results_table.py picks it up.
        body = (
            f"# Chronos-Bolt zero-shot inference (no training)\n"
            f"data: {dataset}\n"
            f"pred_len: {horizon}\n"
            f"model: chronos_zs ({args.model})\n"
            f"params=0  # zero-shot, no trainable params\n"
            f"\nTested {n_seen} samples in {elapsed:.1f}s\n"
            f"Test | mse={metrics['mse']:.6f} mae={metrics['mae']:.6f} "
            f"rmse={metrics['rmse']:.6f}\n"
        )
        log_path.write_text(body)
        print(f"  Test | mse={metrics['mse']:.6f} mae={metrics['mae']:.6f} "
              f"rmse={metrics['rmse']:.6f}", flush=True)


if __name__ == "__main__":
    main()
