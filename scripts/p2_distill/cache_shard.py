"""Sharded teacher-cache helper: fill an index range [start, end) of an
already-allocated memmap file. Used when the main cache_teacher_predictions
script is too slow on one GPU and idle GPUs can help finish the tail.

Assumes the .npy.partial file already exists with the expected (n, horizon,
n_vars) shape (the lead process created it via np.lib.format.open_memmap).
Multiple shards can run concurrently as long as their [start, end) ranges
are disjoint (or perfectly overlapping — chronos forward is deterministic so
overwriting the same indices is harmless).

Usage:
    CUDA_VISIBLE_DEVICES=4 python scripts/p2_distill/cache_shard.py \
        --dataset electricity --horizon 720 \
        --start-idx 13400 --end-idx 15500 --batch-size 32
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.datasets import build_dataset  # noqa: E402
from src.models.teacher import ChronosTeacher  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--horizon", type=int, required=True)
    ap.add_argument("--start-idx", type=int, required=True)
    ap.add_argument("--end-idx", type=int, required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--split", default="train")
    ap.add_argument("--model", default="amazon/chronos-bolt-base")
    ap.add_argument("--cache-dir", default="cache/teacher")
    args = ap.parse_args()

    cfg = OmegaConf.load(ROOT / "configs" / "data" / f"{args.dataset}.yaml")
    ds = build_dataset(
        data_kind=cfg.data_kind, flag=args.split,
        root_path=cfg.root_path, data_path=cfg.data_path,
        seq_len=96, label_len=48, pred_len=args.horizon,
        features=cfg.features, target=cfg.target, freq=cfg.freq,
    )
    n = len(ds)
    n_vars = ds.data_x.shape[-1]
    start, end = args.start_idx, min(args.end_idx, n)
    if start >= end:
        print(f"start ({start}) >= end ({end}) — nothing to do.")
        return

    cache_dir = ROOT / args.cache_dir
    base_name = f"{args.dataset}_h{args.horizon}_{args.split}"
    partial = cache_dir / f"{base_name}.npy.partial"
    if not partial.exists():
        raise FileNotFoundError(
            f"Memmap partial not found: {partial}. Start a lead cache process "
            "first so the file is allocated, then run shards."
        )
    out_arr = np.lib.format.open_memmap(partial, mode="r+",
                                        dtype=np.float32,
                                        shape=(n, args.horizon, n_vars))
    print(f"Opened memmap {partial.name} shape={out_arr.shape} dtype={out_arr.dtype}",
          flush=True)

    print(f"Loading {args.model} ...", flush=True)
    teacher = ChronosTeacher(model_name=args.model, device="cuda",
                             dtype=torch.bfloat16)
    print(f"Teacher ready. Shard range [{start}, {end}) of {n} total.", flush=True)

    subset = Subset(ds, list(range(start, end)))
    loader = DataLoader(subset, batch_size=args.batch_size, shuffle=False,
                        num_workers=2, pin_memory=True)
    t0 = time.time()
    offset = start
    for i, (batch_x, _, _, _) in enumerate(loader):
        batch_x = batch_x.float().cuda(non_blocking=True)
        pred = teacher.predict(batch_x, args.horizon)
        np_pred = pred.detach().cpu().float().numpy()
        b = np_pred.shape[0]
        out_arr[offset:offset + b] = np_pred
        offset += b
        if (i + 1) % 25 == 0:
            done = offset - start
            todo = end - start
            print(f"    shard {start}->{end}: {done}/{todo} "
                  f"({100*done/todo:.0f}%)  elapsed {time.time()-t0:.0f}s",
                  flush=True)
    out_arr.flush()
    del out_arr
    print(f"  Shard [{start}, {end}) done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
