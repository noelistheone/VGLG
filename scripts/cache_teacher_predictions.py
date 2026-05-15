"""Cache Chronos-Bolt teacher predictions for the train+val splits of selected
(dataset, horizon) pairs, so distillation training can do an O(1) lookup per
sample instead of paying per-batch teacher inference cost.

Cache layout:
    cache/teacher/<dataset>_h<horizon>_<split>.pt   # tensor (n_windows, pred_len, n_vars)

Usage:
    python scripts/cache_teacher_predictions.py --datasets ettm1 ettm2 \
        --horizons 96 192 336 720 --splits train val
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
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.datasets import build_dataset
from src.models.teacher import ChronosTeacher

DATASETS = ["ettm1", "ettm2"]
HORIZONS = [96, 192, 336, 720]
SPLITS = ["train", "val"]


def _data_cfg(name: str):
    return OmegaConf.load(ROOT / "configs" / "data" / f"{name}.yaml")


def cache_one(teacher: ChronosTeacher, dataset: str, split: str, horizon: int,
              cache_dir: Path, batch_size: int = 8, fp16: bool = False):
    """Build a teacher cache, streaming each chunk to disk to avoid an
    intermediate full-tensor in RAM.

    fp16=True writes a numpy .npy file at half precision. This is critical for
    Electricity/Traffic h=720 where the full fp32 tensor (30 GB) would OOM the
    31 GB system RAM. Loaded by DistillDataset with `np.load(..., mmap_mode='r')`
    so only the windows actually touched are paged in.

    fp16=False keeps the legacy fp32 .pt format (used for ETTm caches).
    """
    cfg = _data_cfg(dataset)
    ds = build_dataset(
        data_kind=cfg.data_kind,
        flag=split,
        root_path=cfg.root_path,
        data_path=cfg.data_path,
        seq_len=96,
        label_len=48,
        pred_len=horizon,
        features=cfg.features,
        target=cfg.target,
        freq=cfg.freq,
    )
    n = len(ds)
    n_vars = ds.data_x.shape[-1]

    ext = "npy" if fp16 else "pt"
    cache_path = cache_dir / f"{dataset}_h{horizon}_{split}.{ext}"
    if cache_path.exists():
        if fp16:
            arr = np.load(cache_path, mmap_mode="r")
            if arr.shape == (n, horizon, n_vars):
                print(f"  [skip] {cache_path.name}: already has {n} entries")
                return
            print(f"  [redo] {cache_path.name}: shape mismatch {arr.shape}")
        else:
            cached = torch.load(cache_path, weights_only=True)
            if cached.shape[0] == n:
                print(f"  [skip] {cache_path.name}: already has {n} entries")
                return
            print(f"  [redo] {cache_path.name}: {cached.shape[0]} != {n}")

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2,
                        pin_memory=True)
    t0 = time.time()

    if fp16:
        # Pre-allocate the full numpy array on disk and write chunks in-place.
        tmp_path = cache_path.with_suffix(".npy.partial")
        # numpy.lib.format requires shape known up-front; use memmap for in-place writes.
        out_arr = np.lib.format.open_memmap(
            tmp_path, mode="w+", dtype=np.float16,
            shape=(n, horizon, n_vars),
        )
        offset = 0
        for i, (batch_x, _, _, _) in enumerate(loader):
            batch_x = batch_x.float().cuda(non_blocking=True)
            pred = teacher.predict(batch_x, horizon)                      # (B, pred, N)
            np_pred = pred.detach().cpu().to(torch.float16).numpy()
            b = np_pred.shape[0]
            out_arr[offset:offset + b] = np_pred
            offset += b
            if (i + 1) % 50 == 0:
                print(f"    {offset}/{n} ({100*offset/n:.0f}%)  "
                      f"elapsed {time.time()-t0:.0f}s", flush=True)
        out_arr.flush()
        del out_arr
        tmp_path.rename(cache_path)
        print(f"  Wrote {cache_path.name}: {(n, horizon, n_vars)} fp16 "
              f"in {time.time()-t0:.0f}s")
    else:
        out_chunks = []
        for i, (batch_x, _, _, _) in enumerate(loader):
            batch_x = batch_x.float().cuda(non_blocking=True)
            pred = teacher.predict(batch_x, horizon)
            out_chunks.append(pred.float().cpu())
            if (i + 1) % 50 == 0:
                done = (i + 1) * batch_size
                print(f"    {done}/{n} ({100*done/n:.0f}%)  "
                      f"elapsed {time.time()-t0:.0f}s", flush=True)
        out = torch.cat(out_chunks, dim=0)[:n]
        torch.save(out, cache_path)
        print(f"  Wrote {cache_path.name}: {tuple(out.shape)} fp32 "
              f"in {time.time()-t0:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--horizons", nargs="+", type=int, default=HORIZONS)
    ap.add_argument("--splits", nargs="+", default=SPLITS)
    ap.add_argument("--model", default="amazon/chronos-bolt-base")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--cache-dir", default="cache/teacher")
    ap.add_argument("--fp16", action="store_true",
                    help="Save as numpy fp16 (.npy) instead of torch fp32 (.pt). "
                         "Required for Electricity/Traffic h=720 to fit in 31GB RAM.")
    args = ap.parse_args()

    cache_dir = ROOT / args.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.model} ...", flush=True)
    teacher = ChronosTeacher(model_name=args.model, device="cuda",
                             dtype=torch.bfloat16)
    print("Teacher ready.\n", flush=True)

    for dataset, split, horizon in product(args.datasets, args.splits, args.horizons):
        print(f"=== {dataset} {split} h={horizon} ===", flush=True)
        cache_one(teacher, dataset, split, horizon, cache_dir, args.batch_size,
                  fp16=args.fp16)


if __name__ == "__main__":
    main()
