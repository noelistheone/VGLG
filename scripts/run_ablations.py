"""Run Table 2 ablations: 10 VGLG variants x 3 datasets x 3 horizons x 3 seeds.

Each variant overrides the base metatsf_vglg config with a single change:
    - full         baseline (no override)
    - fixed_gate   mixer.gate_mode=fixed_0.5
    - local_only   mixer.gate_mode=fixed_1.0
    - global_only  mixer.gate_mode=fixed_0.0
    - kernel_15    mixer.kernel_size=15
    - kernel_51    mixer.kernel_size=51
    - rank_4       mixer.rank=4
    - rank_16      mixer.rank=16
    - rank_32      mixer.rank=32
    - no_revin     revin=false

Logs go to logs/ablation/<variant>/<dataset>_h<horizon>_s<seed>.log so the
update_results_table.py picks them up into Table 2.

Resume-aware: a run is skipped iff its log file exists and ends with 'Test |'.

Usage:
    python scripts/run_ablations.py                       # all 270 runs
    python scripts/run_ablations.py --variants full kernel_15
    python scripts/run_ablations.py --datasets etth1 --dry-run
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATASETS = ["etth1", "weather", "electricity"]
HORIZONS = [96, 336, 720]
SEEDS = [2021, 2022, 2023]

VARIANTS: dict[str, list[str]] = {
    # variant name -> list of Hydra overrides on top of `model=metatsf_vglg`
    "full":        [],
    "fixed_gate":  ["model.mixer.gate_mode=fixed_0.5"],
    "local_only":  ["model.mixer.gate_mode=fixed_1.0"],
    "global_only": ["model.mixer.gate_mode=fixed_0.0"],
    "kernel_15":   ["model.mixer.kernel_size=15"],
    "kernel_51":   ["model.mixer.kernel_size=51"],
    "rank_4":      ["model.mixer.rank=4"],
    "rank_16":     ["model.mixer.rank=16"],
    "rank_32":     ["model.mixer.rank=32"],
    "no_revin":    ["model.revin=false"],
}

TAG = "ablation"
# Per-dataset batch sizes that fit on a 24GB 4090.
BATCH_SIZE = {"etth1": 32, "weather": 32, "electricity": 16}
NUM_WORKERS = 4
TRAIN_EPOCHS = 10


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


def run_one(variant: str, dataset: str, horizon: int, seed: int,
            ablation_root: Path) -> tuple[str, float]:
    log_dir = ablation_root / variant
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{dataset}_h{horizon}_s{seed}.log"

    bs = BATCH_SIZE.get(dataset, 16)
    cmd = [
        sys.executable, "-m", "src.train.trainer",
        "model=metatsf_vglg",
        f"data={dataset}",
        f"train.pred_len={horizon}",
        f"train.batch_size={bs}",
        f"train.num_workers={NUM_WORKERS}",
        f"train.train_epochs={TRAIN_EPOCHS}",
        f"seed={seed}",
        f"tag={TAG}_{variant}",
        # Hydra needs to know how to nest the run dir; tag is already enough.
    ] + VARIANTS[variant]

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
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS.keys()))
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--horizons", nargs="+", type=int, default=HORIZONS)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    ablation_root = ROOT / "logs" / "ablation"
    ablation_root.mkdir(parents=True, exist_ok=True)

    combos = list(product(args.variants, args.datasets, args.horizons, args.seeds))
    todo = []
    skipped = 0
    for v, d, h, s in combos:
        log_path = ablation_root / v / f"{d}_h{h}_s{s}.log"
        if is_done(log_path):
            skipped += 1
            continue
        todo.append((v, d, h, s))

    print(f"Total combos: {len(combos)} | already done: {skipped} | to run: {len(todo)}")
    if args.dry_run:
        for v, d, h, s in todo[:10]:
            print(f"  would run: {v} / {d}_h{h}_s{s}")
        if len(todo) > 10:
            print(f"  ... and {len(todo) - 10} more")
        return

    t_start = time.time()
    n_ok = 0
    for i, (v, d, h, s) in enumerate(todo, 1):
        name = f"{v:>12s} / {d}_h{h}_s{s}"
        print(f"[{i:3d}/{len(todo)}] {name}", end=" ... ", flush=True)
        status, elapsed = run_one(v, d, h, s, ablation_root)
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
