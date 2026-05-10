"""P1's Table 1 sweep: ETTm1 + ETTm2 x 12 models x 4 horizons x 3 seeds = 288 runs.

Resume-aware: a run is skipped iff its log file already exists and ends with a
'Test |' line (meaning the previous attempt completed). Logs go to
logs/main/<run_name>.log; numbers are also visible in the trailing tail of
each log.

Usage:
    python scripts/run_p1_table1.py                # run everything missing
    python scripts/run_p1_table1.py --dry-run      # print what would run
    python scripts/run_p1_table1.py --models dlinear vglg  # subset
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATASETS = ["ettm1", "ettm2"]
HORIZONS = [96, 192, 336, 720]
SEEDS = [2021, 2022, 2023]
MODELS = [
    "dlinear", "lstm", "gru", "segrnn",
    "timemixer", "moderntcn", "itransformer", "patchtst",
    "metatsf_mlp", "metatsf_conv", "metatsf_attn", "metatsf_vglg",
]
TAG = "main"
BATCH_SIZE = 64       # ETT only 7 vars, 4090 has plenty of room
NUM_WORKERS = 4
TRAIN_EPOCHS = 10


def run_name(model: str, dataset: str, horizon: int, seed: int) -> str:
    return f"{dataset}_{model}_h{horizon}_s{seed}"


def is_done(log_path: Path) -> bool:
    """A run is considered done iff its log ends with a 'Test |' line."""
    if not log_path.exists():
        return False
    try:
        with log_path.open("rb") as fh:
            fh.seek(0, 2)
            size = fh.tell()
            fh.seek(max(0, size - 4096))
            tail = fh.read().decode(errors="ignore")
    except OSError:
        return False
    return "Test |" in tail


def run_one(model: str, dataset: str, horizon: int, seed: int, log_dir: Path) -> tuple[str, float]:
    log_path = log_dir / f"{run_name(model, dataset, horizon, seed)}.log"
    cmd = [
        sys.executable, "-m", "src.train.trainer",
        f"model={model}",
        f"data={dataset}",
        f"train.pred_len={horizon}",
        f"train.batch_size={BATCH_SIZE}",
        f"train.num_workers={NUM_WORKERS}",
        f"train.train_epochs={TRAIN_EPOCHS}",
        f"seed={seed}",
        f"tag={TAG}",
    ]
    t0 = time.time()
    with log_path.open("w") as fh:
        r = subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT, timeout=3600)
    elapsed = time.time() - t0
    if r.returncode != 0:
        return "FAIL", elapsed
    if not is_done(log_path):
        return "FAIL_NO_TEST", elapsed
    return "OK", elapsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--horizons", nargs="+", type=int, default=HORIZONS)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    log_dir = ROOT / "logs" / TAG
    log_dir.mkdir(parents=True, exist_ok=True)

    combos = list(product(args.datasets, args.horizons, args.models, args.seeds))
    todo = []
    skipped = 0
    for d, h, m, s in combos:
        log_path = log_dir / f"{run_name(m, d, h, s)}.log"
        if is_done(log_path):
            skipped += 1
            continue
        todo.append((m, d, h, s))

    print(f"Total combos: {len(combos)} | already done: {skipped} | to run: {len(todo)}")
    if args.dry_run:
        for m, d, h, s in todo[:10]:
            print(f"  would run: {run_name(m, d, h, s)}")
        if len(todo) > 10:
            print(f"  ... and {len(todo) - 10} more")
        return

    t_start = time.time()
    n_ok = 0
    for i, (m, d, h, s) in enumerate(todo, 1):
        name = run_name(m, d, h, s)
        print(f"[{i:3d}/{len(todo)}] {name}", end=" ... ", flush=True)
        status, elapsed = run_one(m, d, h, s, log_dir)
        marker = "OK " if status == "OK" else f"X ({status})"
        wall = time.time() - t_start
        eta = wall / i * (len(todo) - i)
        print(f"{marker}  {elapsed:.0f}s  | wall {wall/60:.1f}min  eta {eta/60:.0f}min")
        if status == "OK":
            n_ok += 1

    print()
    print(f"=== Done: {n_ok}/{len(todo)} OK | total wall {(time.time() - t_start)/60:.1f}min ===")


if __name__ == "__main__":
    main()
