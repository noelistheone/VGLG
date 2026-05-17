"""Run MetaTSF-VGLG + KD on selected (dataset, horizon, seed) combos.

Resume-aware: skips runs whose log already ends with `Test |`.
Logs go to: logs/distill/<dataset>_metatsf_vglg_kd_h<horizon>_s<seed>.log

Prerequisite: run scripts/cache_teacher_predictions.py first to populate
cache/teacher/<dataset>_h<horizon>_train.pt.

Usage:
    python scripts/run_distill.py
    python scripts/run_distill.py --datasets ettm1 --horizons 96
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
TAG = "distill"
NUM_WORKERS = 4
TRAIN_EPOCHS = 10
# Per-dataset batch size (matched to verify_all sweep).
BATCH_SIZE = {
    "etth1": 32, "etth2": 32, "ettm1": 64, "ettm2": 64,
    "weather": 32, "electricity": 16, "traffic": 8, "f1weather": 32,
}


def is_done(log_path: Path) -> bool:
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


def run_one(dataset: str, horizon: int, seed: int, log_dir: Path) -> tuple[str, float]:
    log_path = log_dir / f"{dataset}_metatsf_vglg_kd_h{horizon}_s{seed}.log"
    bs = BATCH_SIZE.get(dataset, 16)
    cmd = [
        sys.executable, "-m", "src.train.distill_trainer",
        "--config-name=distill_default",
        f"data={dataset}",
        f"train.pred_len={horizon}",
        f"train.batch_size={bs}",
        f"train.num_workers={NUM_WORKERS}",
        f"train.train_epochs={TRAIN_EPOCHS}",
        f"seed={seed}",
        f"tag={TAG}",
    ]
    t0 = time.time()
    # Some long traffic h=720 runs take >1h just for cold-cache load + first epoch.
    # Catch TimeoutExpired so a stuck run doesn't abort the whole sweep.
    try:
        with log_path.open("w") as fh:
            r = subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT, timeout=14400)
    except subprocess.TimeoutExpired:
        return "TIMEOUT", time.time() - t0
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
    args = ap.parse_args()

    log_dir = ROOT / "logs" / TAG
    log_dir.mkdir(parents=True, exist_ok=True)

    combos = list(product(args.datasets, args.horizons, args.seeds))
    todo = []
    skipped = 0
    for d, h, s in combos:
        log_path = log_dir / f"{d}_metatsf_vglg_kd_h{h}_s{s}.log"
        if is_done(log_path):
            skipped += 1
            continue
        todo.append((d, h, s))

    print(f"Total combos: {len(combos)} | already done: {skipped} | to run: {len(todo)}")

    t_start = time.time()
    n_ok = 0
    for i, (d, h, s) in enumerate(todo, 1):
        name = f"{d}_h{h}_s{s}"
        print(f"[{i:3d}/{len(todo)}] {name}", end=" ... ", flush=True)
        status, elapsed = run_one(d, h, s, log_dir)
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
