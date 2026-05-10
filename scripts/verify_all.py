"""Sanity sweep: run every (model, dataset) combo for 1 epoch on a tiny budget.

The goal is to discover OOMs / shape errors / NaNs *before* launching the full
Week 5-6 matrix. Writes a CSV with one row per combo and a summary at the end.

Usage:
    python scripts/verify_all.py                       # all combos, h=96
    python scripts/verify_all.py --horizons 96 720     # spot-check long h
    python scripts/verify_all.py --models vglg_mlp     # subset of models
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS = [
    # baselines
    "dlinear", "lstm", "gru", "segrnn",
    "timemixer", "moderntcn", "itransformer", "patchtst",
    # ours: shared MetaTSF backbone, 4 different mixers
    "metatsf_mlp", "metatsf_conv", "metatsf_attn", "metatsf_vglg",
]
DEFAULT_DATASETS = ["etth1", "etth2", "ettm1", "ettm2", "weather", "electricity", "traffic", "f1weather"]

# Per-dataset batch sizes that fit on a 24GB 4090 for all 8 models.
BATCH_SIZE = {
    "etth1": 32, "etth2": 32, "ettm1": 32, "ettm2": 32,
    "weather": 32, "electricity": 16, "traffic": 8,
    "f1weather": 32,
}


def run_one(model: str, dataset: str, horizon: int, log_dir: Path) -> dict:
    bs = BATCH_SIZE.get(dataset, 16)
    log_path = log_dir / f"{dataset}_{model}_h{horizon}.log"
    cmd = [
        sys.executable, "-m", "src.train.trainer",
        f"model={model}",
        f"data={dataset}",
        f"train.pred_len={horizon}",
        f"train.batch_size={bs}",
        "train.num_workers=2",
        "train.train_epochs=1",
        "train.use_amp=true",
        "tag=verify",
    ]
    t0 = time.time()
    try:
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=900)
        elapsed = time.time() - t0
        out = r.stdout + r.stderr
        log_path.write_text(out)
        # parse Test mse line
        test_line = next((l for l in out.splitlines() if l.startswith("Test |")), None)
        params_line = next((l for l in out.splitlines() if "params=" in l), None)
        if r.returncode != 0:
            status = "FAIL"
            reason = "OOM" if "out of memory" in out.lower() else "ERROR"
        elif test_line and "nan" in test_line.lower():
            status = "FAIL"
            reason = "NaN"
        elif test_line is None:
            status = "FAIL"
            reason = "NO_TEST_OUTPUT"
        else:
            status = "OK"
            reason = ""
        return {
            "model": model, "dataset": dataset, "horizon": horizon,
            "batch_size": bs, "status": status, "reason": reason,
            "elapsed_s": round(elapsed, 1),
            "test_line": (test_line or "").strip(),
            "params": (params_line or "").strip(),
        }
    except subprocess.TimeoutExpired:
        return {
            "model": model, "dataset": dataset, "horizon": horizon,
            "batch_size": bs, "status": "FAIL", "reason": "TIMEOUT",
            "elapsed_s": 900, "test_line": "", "params": "",
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    ap.add_argument("--horizons", nargs="+", type=int, default=[96])
    ap.add_argument("--out", default="results/verify_sweep.csv")
    args = ap.parse_args()

    log_dir = ROOT / "logs" / "verify"
    log_dir.mkdir(parents=True, exist_ok=True)
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    combos = [(m, d, h) for d in args.datasets for h in args.horizons for m in args.models]
    print(f"Running {len(combos)} combos. Logs in {log_dir}, CSV at {out_path}.")
    print()

    results = []
    for i, (m, d, h) in enumerate(combos, 1):
        print(f"[{i:3d}/{len(combos)}] {m:20s} {d:12s} h={h:3d}  ...", end=" ", flush=True)
        r = run_one(m, d, h, log_dir)
        results.append(r)
        marker = "OK " if r["status"] == "OK" else f"X ({r['reason']})"
        print(f"{marker}  {r['elapsed_s']}s  {r['test_line']}")

    # Write CSV
    with out_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)

    # Summary
    n_ok = sum(1 for r in results if r["status"] == "OK")
    print()
    print(f"=== Summary: {n_ok}/{len(results)} passed ===")
    failures = [r for r in results if r["status"] != "OK"]
    if failures:
        print("Failures:")
        for r in failures:
            print(f"  {r['model']:20s} {r['dataset']:12s} h={r['horizon']:3d}  {r['reason']}")
    else:
        print("All combos passed.")
    print(f"\nFull CSV: {out_path}")


if __name__ == "__main__":
    main()
