"""Find the max batch size that fits on a 48GB A6000 for P2's worst-case combos.

Tests Traffic and Electricity at h=720 (and a couple h=96 spot-checks) with the
heaviest models (PatchTST, iTransformer, ModernTCN, MetaTSF-VGLG). For each
combo, tries candidate batch sizes in descending order and stops at the first
one that finishes 1 epoch without OOM.

Output: results/batch_sizes_p2.json — consumed by run_p2_main.py.

Usage:
    python scripts/probe_batch_p2.py --gpu 4
    python scripts/probe_batch_p2.py --gpu 4 --skip-light    # only h=720 probes
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# (dataset, model, horizon, candidate batches — try descending, stop at first OK)
# Heavy models first; if PatchTST fits, lighter models fit too.
PROBES: list[tuple[str, str, int, list[int]]] = [
    # Traffic h=720 — worst case (862 vars × 720 outputs)
    ("traffic",     "patchtst",      720, [16, 12, 8, 4]),
    ("traffic",     "itransformer",  720, [16, 12, 8, 4]),
    ("traffic",     "moderntcn",     720, [16, 12, 8, 4]),
    ("traffic",     "metatsf_vglg",  720, [16, 12, 8]),
    # Traffic h=336
    ("traffic",     "patchtst",      336, [24, 16, 12, 8]),
    # Traffic h=96 — spot-check headroom
    ("traffic",     "patchtst",       96, [32, 24, 16]),
    # Electricity h=720 — second-worst (321 vars)
    ("electricity", "patchtst",      720, [32, 24, 16, 12]),
    ("electricity", "itransformer",  720, [32, 24, 16, 12]),
    ("electricity", "metatsf_vglg",  720, [32, 24, 16]),
    # Electricity h=336
    ("electricity", "patchtst",      336, [48, 32, 24, 16]),
    # Electricity h=96
    ("electricity", "patchtst",       96, [64, 48, 32, 24]),
]

# Optional "light" probes for ablation datasets (ETTh1, Weather) — only matters
# if we ever increase from defaults. Cheap to test.
LIGHT_PROBES: list[tuple[str, str, int, list[int]]] = [
    ("etth1",   "patchtst", 720, [64, 48, 32]),
    ("weather", "patchtst", 720, [48, 32, 24]),
]


def run_one(model: str, dataset: str, h: int, bs: int, gpu: int, timeout: int,
            success_after_batch: int = 20) -> tuple[str, float]:
    """Run 1 epoch and stream stdout. Resolve as soon as we know the answer:
      - 'out of memory' in any line          -> OOM, kill
      - 'batch N/' for N >= success_after_batch -> OK (bs survived N forward+backward+step)
      - 'Epoch 1 |' line                     -> OK (whole epoch completed)
      - process exits cleanly                -> OK
      - outer timeout fires with no OOM      -> OK_SLOW (didn't OOM in `timeout`s)
    """
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    cmd = [
        sys.executable, "-u", "-m", "src.train.trainer",
        f"model={model}", f"data={dataset}",
        f"train.pred_len={h}", f"train.batch_size={bs}",
        "train.train_epochs=1", "train.num_workers=2",
        f"train.log_interval={success_after_batch}",
        "train.patience=99",
        "tag=probe",
    ]
    t0 = time.time()
    proc = subprocess.Popen(
        cmd, env=env, cwd=ROOT,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )

    result = {"status": None}

    def reader():
        try:
            for line in proc.stdout:
                low = line.lower()
                if "out of memory" in low or "cuda out of memory" in low:
                    result["status"] = "OOM"
                    proc.terminate()
                    return
                # First batch log proves a full forward+backward+optimizer step succeeded.
                if f"batch {success_after_batch}/" in line or line.startswith("Epoch 1 |"):
                    result["status"] = "OK"
                    proc.terminate()
                    return
        except Exception:
            pass

    t = threading.Thread(target=reader, daemon=True)
    t.start()

    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        if result["status"] is None:
            result["status"] = "OK_SLOW"
        proc.kill()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass

    t.join(timeout=5)
    elapsed = time.time() - t0
    if result["status"] is None:
        # Process exited before the reader saw the success/OOM signal — inspect rc
        rc = proc.poll()
        result["status"] = "OK" if rc == 0 else "ERROR"
    return result["status"], elapsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=4)
    ap.add_argument("--skip-light", action="store_true", help="Skip ETTh1/Weather probes")
    ap.add_argument("--timeout", type=int, default=300,
                    help="Per-run wall-clock cap. Probes kill on OOM or after first batch log, so "
                         "this is just a safety net for runs that produce no output.")
    ap.add_argument("--out", default="results/batch_sizes_p2.json")
    args = ap.parse_args()

    probes = list(PROBES) + ([] if args.skip_light else LIGHT_PROBES)

    print(f"Probing on GPU {args.gpu}. Each run is 1 epoch, cap {args.timeout}s.")
    print(f"{len(probes)} probes total. Stops at first batch that fits.\n")

    fit: dict[str, int] = {}    # "dataset_h{h}" -> bs
    csv_rows: list[str] = ["dataset,model,horizon,batch_size,status,elapsed_s"]

    for ds, model, h, bs_options in probes:
        key = f"{ds}_h{h}"
        for bs in bs_options:
            print(f"  {ds:12s} {model:18s} h={h:3d} bs={bs:3d} ...", end=" ", flush=True)
            status, elapsed = run_one(model, ds, h, bs, args.gpu, args.timeout)
            print(f"{status:8s} {elapsed:6.1f}s")
            csv_rows.append(f"{ds},{model},{h},{bs},{status},{elapsed:.1f}")
            if status in ("OK", "OK_SLOW"):
                # Keep the smallest fit per (ds, h) across models — heaviest model wins
                if key not in fit or bs < fit[key]:
                    fit[key] = bs
                break

    out_json = ROOT / args.out
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(fit, indent=2, sort_keys=True))
    out_csv = out_json.with_suffix(".csv")
    out_csv.write_text("\n".join(csv_rows))

    print(f"\nResults: {out_json}")
    print(f"Raw log: {out_csv}\n")
    print("Recommended BATCH (smallest fitting bs across models per (ds, h)):")
    for k in sorted(fit):
        print(f"  {k!r:24s}: {fit[k]}")
    print("\nNext step: run_p2_main.py will read this JSON automatically.")


if __name__ == "__main__":
    main()
