"""P2 launcher — 369 runs across 3 A6000s.

Workload (per docs/compute_allocation.md):
  Table 1 main:    Electricity + Traffic × 12 models × 4 horizons × 3 seeds = 288
  Table 2 rank:    {ETTh1, Weather, Electricity} × ranks {4,16,32}
                                                × 3 horizons × 3 seeds      =  81
  Total                                                                     = 369

Distribution: by seed (each GPU owns one of {2021, 2022, 2023}) — balanced load.

Skip-if-done: a run is considered complete if its log contains a final
'Test |' line. Otherwise it re-runs.

Usage:
    python scripts/run_p2_main.py --dry-run
    python scripts/run_p2_main.py --gpus 4 5 6
    python scripts/run_p2_main.py --gpus 4 5 6 --only-table 1
    python scripts/run_p2_main.py --gpus 4 5 6 --limit 6        # smoke test
    python scripts/run_p2_main.py --gpus 4 5 6 --rerun-failed   # ignore prior FAIL/OOM
    python scripts/run_p2_main.py --gpus 4 5 6 --wandb

Tip: launch inside tmux so SSH drops don't kill the run.
    tmux new -s p2 'python scripts/run_p2_main.py --gpus 4 5 6 2>&1 | tee logs/p2_master.log'
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MODELS = [
    # baselines (8)
    "dlinear", "lstm", "gru", "segrnn",
    "timemixer", "moderntcn", "itransformer", "patchtst",
    # ours (4) — same backbone, different mixer
    "metatsf_mlp", "metatsf_conv", "metatsf_attn", "metatsf_vglg",
]
SEEDS = [2021, 2022, 2023]
HORIZONS = [96, 192, 336, 720]
TABLE1_DATASETS = ["electricity", "traffic"]
TABLE2_DATASETS = ["etth1", "weather", "electricity"]
TABLE2_HORIZONS = [96, 336, 720]
TABLE2_RANKS = [4, 16, 32]

# Conservative defaults for 48GB A6000. probe_batch_p2.py overrides via JSON.
DEFAULT_BATCH: dict[str, int] = {
    "electricity_h96":  32, "electricity_h192": 32,
    "electricity_h336": 24, "electricity_h720": 16,
    "traffic_h96":      16, "traffic_h192":     16,
    "traffic_h336":     12, "traffic_h720":      8,
    "etth1_h96":        64, "etth1_h336":       64, "etth1_h720": 48,
    "weather_h96":      64, "weather_h336":     48, "weather_h720": 32,
}


def load_batch_table() -> dict[str, int]:
    probe_path = ROOT / "results" / "batch_sizes_p2.json"
    if probe_path.exists():
        probe = json.loads(probe_path.read_text())
        print(f"Loaded probe results from {probe_path}: {len(probe)} entries override defaults.")
        return {**DEFAULT_BATCH, **probe}
    print(f"No probe file at {probe_path} — using DEFAULT_BATCH. "
          "Run probe_batch_p2.py first to tighten these.")
    return dict(DEFAULT_BATCH)


def batch_for(batch_table: dict[str, int], dataset: str, h: int) -> int:
    key = f"{dataset}_h{h}"
    if key not in batch_table:
        raise KeyError(f"No batch size configured for {key!r}. "
                       f"Add it to DEFAULT_BATCH or run probe_batch_p2.py.")
    return batch_table[key]


# ---------- command generation ----------

def table1_tasks(batch_table: dict[str, int], use_wandb: bool, gpu_resident: bool) -> list[dict]:
    nw = 0 if gpu_resident else 4
    tasks = []
    for dataset in TABLE1_DATASETS:
        for model in MODELS:
            for h in HORIZONS:
                for seed in SEEDS:
                    bs = batch_for(batch_table, dataset, h)
                    run_name = f"{dataset}_{model}_h{h}_s{seed}"
                    cmd = [
                        sys.executable, "-u", "-m", "src.train.trainer",
                        f"model={model}", f"data={dataset}",
                        f"train.pred_len={h}",
                        f"train.batch_size={bs}",
                        f"train.num_workers={nw}",
                        f"train.gpu_resident_data={'true' if gpu_resident else 'false'}",
                        f"seed={seed}",
                        "tag=main",
                    ]
                    if use_wandb:
                        cmd.append("train.use_wandb=true")
                    tasks.append({
                        "table": 1, "tag": "main", "run_name": run_name,
                        "dataset": dataset, "model": model,
                        "horizon": h, "seed": seed, "rank": None,
                        "batch_size": bs, "cmd": cmd,
                    })
    return tasks


def table2_tasks(batch_table: dict[str, int], use_wandb: bool, gpu_resident: bool) -> list[dict]:
    nw = 0 if gpu_resident else 4
    tasks = []
    for dataset in TABLE2_DATASETS:
        for rank in TABLE2_RANKS:
            for h in TABLE2_HORIZONS:
                for seed in SEEDS:
                    bs = batch_for(batch_table, dataset, h)
                    tag = f"ablation_rank_r{rank}"
                    # Hydra auto-derives run_name from data+model+horizon+seed.
                    # We mirror that for our log path.
                    run_name = f"{dataset}_metatsf_vglg_h{h}_s{seed}"
                    cmd = [
                        sys.executable, "-u", "-m", "src.train.trainer",
                        "model=metatsf_vglg", f"data={dataset}",
                        f"train.pred_len={h}",
                        f"train.batch_size={bs}",
                        f"train.num_workers={nw}",
                        f"train.gpu_resident_data={'true' if gpu_resident else 'false'}",
                        f"seed={seed}",
                        f"model.mixer.rank={rank}",
                        f"tag={tag}",
                    ]
                    if use_wandb:
                        cmd.append("train.use_wandb=true")
                    tasks.append({
                        "table": 2, "tag": tag, "run_name": run_name,
                        "dataset": dataset, "model": f"metatsf_vglg_r{rank}",
                        "horizon": h, "seed": seed, "rank": rank,
                        "batch_size": bs, "cmd": cmd,
                    })
    return tasks


# ---------- log paths + skip logic ----------

def log_path(tag: str, run_name: str) -> Path:
    return ROOT / "logs" / "p2" / tag / f"{run_name}.log"


def is_done(task: dict) -> bool:
    """Run is done iff its log file ends with a 'Test |' summary line."""
    log = log_path(task["tag"], task["run_name"])
    if not log.exists():
        return False
    try:
        with log.open("rb") as fh:
            fh.seek(0, 2)
            size = fh.tell()
            fh.seek(max(0, size - 4096), 0)
            tail = fh.read().decode("utf-8", errors="replace")
        return "Test |" in tail
    except OSError:
        return False


def prior_status(task: dict) -> str:
    """Inspect log to classify a previously-attempted run."""
    log = log_path(task["tag"], task["run_name"])
    if not log.exists():
        return "NEW"
    try:
        text = log.read_text(errors="replace")
    except OSError:
        return "NEW"
    if "Test |" in text:
        return "DONE"
    if "out of memory" in text.lower():
        return "OOM"
    return "FAIL"


# ---------- execution ----------

def run_task(task: dict, gpu_id: int) -> dict:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    log = log_path(task["tag"], task["run_name"])
    log.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with log.open("w") as fh:
        r = subprocess.run(task["cmd"], env=env, cwd=ROOT,
                           stdout=fh, stderr=subprocess.STDOUT)
    elapsed = time.time() - t0
    text = log.read_text(errors="replace")
    test_line = next((ln for ln in text.splitlines() if ln.startswith("Test |")), "")
    if "out of memory" in text.lower():
        status = "OOM"
    elif r.returncode != 0:
        status = "FAIL"
    elif not test_line:
        status = "FAIL"
    else:
        status = "OK"
    return {
        **task, "gpu": gpu_id, "status": status,
        "elapsed_s": round(elapsed, 1), "test_line": test_line.strip(),
    }


def gpu_worker(gpu_id: int, tasks: list[dict], dry_run: bool, rerun_failed: bool) -> list[dict]:
    results = []
    for i, task in enumerate(tasks, 1):
        head = f"[GPU{gpu_id}] {i:3d}/{len(tasks)}"
        prior = prior_status(task)
        if prior == "DONE":
            print(f"{head} SKIP  {task['run_name']}  (already complete)", flush=True)
            results.append({**task, "gpu": gpu_id, "status": "SKIP",
                            "elapsed_s": 0.0, "test_line": ""})
            continue
        if prior in ("FAIL", "OOM") and not rerun_failed:
            print(f"{head} SKIP  {task['run_name']}  (prior {prior} — pass --rerun-failed to retry)", flush=True)
            results.append({**task, "gpu": gpu_id, "status": f"SKIP_{prior}",
                            "elapsed_s": 0.0, "test_line": ""})
            continue
        if dry_run:
            print(f"{head} DRY   CUDA_VISIBLE_DEVICES={gpu_id} {' '.join(task['cmd'])}", flush=True)
            continue
        print(f"{head} START {task['run_name']}  bs={task['batch_size']}", flush=True)
        r = run_task(task, gpu_id)
        print(f"{head} {r['status']:4s}  {task['run_name']}  ({r['elapsed_s']:.0f}s)  {r['test_line']}", flush=True)
        results.append(r)
    return results


# ---------- entry ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", nargs="+", type=int, default=[4, 5, 6],
                    help="Exactly 3 GPU ids (one per seed)")
    ap.add_argument("--only-table", type=int, choices=[1, 2], default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=None,
                    help="Per-GPU cap on tasks (smoke test)")
    ap.add_argument("--rerun-failed", action="store_true",
                    help="Re-run tasks marked FAIL/OOM in previous launcher run")
    ap.add_argument("--wandb", action="store_true",
                    help="Enable W&B logging (train.use_wandb=true)")
    ap.add_argument("--gpu-resident", action="store_true",
                    help="Pre-load data to GPU; sets num_workers=0. "
                         "Verified bit-identical to legacy path — safe to mix with prior runs.")
    args = ap.parse_args()

    if len(args.gpus) != len(SEEDS):
        sys.exit(f"Need exactly {len(SEEDS)} entries in --gpus (one per seed). "
                 f"Repeats are allowed and mean 'put these seeds on the same GPU'. Got: {args.gpus}")

    batch_table = load_batch_table()
    print()

    all_tasks: list[dict] = []
    if args.only_table in (None, 1):
        all_tasks.extend(table1_tasks(batch_table, args.wandb, args.gpu_resident))
    if args.only_table in (None, 2):
        all_tasks.extend(table2_tasks(batch_table, args.wandb, args.gpu_resident))

    # Partition by seed; merge per unique GPU when --gpus has duplicates
    # (so two seeds on the same GPU run sequentially in one worker — no concurrent training).
    gpu_for_seed = dict(zip(SEEDS, args.gpus))
    unique_gpus = list(dict.fromkeys(args.gpus))   # preserve first-seen order
    by_gpu: dict[int, list[dict]] = {g: [] for g in unique_gpus}
    for t in all_tasks:
        by_gpu[gpu_for_seed[t["seed"]]].append(t)

    # Pre-flight summary
    total = sum(len(v) for v in by_gpu.values())
    print(f"=== P2 launcher: {total} tasks across {len(unique_gpus)} unique GPU(s) ===")
    for gpu in unique_gpus:
        seeds_here = [s for s in SEEDS if gpu_for_seed[s] == gpu]
        ts = by_gpu[gpu]
        n_t1 = sum(1 for t in ts if t["table"] == 1)
        n_t2 = sum(1 for t in ts if t["table"] == 2)
        n_done = sum(1 for t in ts if prior_status(t) == "DONE")
        n_fail = sum(1 for t in ts if prior_status(t) in ("FAIL", "OOM"))
        seeds_str = ",".join(str(s) for s in seeds_here)
        print(f"  GPU{gpu}  seeds=[{seeds_str}]  total={len(ts)}  T1={n_t1}  T2={n_t2}  "
              f"already_done={n_done}  prior_failed={n_fail}")

    if args.limit:
        for g in by_gpu:
            by_gpu[g] = by_gpu[g][: args.limit]
        print(f"\n--limit {args.limit}: keeping first {args.limit} per GPU "
              f"({sum(len(v) for v in by_gpu.values())} tasks)")

    print()
    t_start = time.time()
    all_results: list[dict] = []
    with ThreadPoolExecutor(max_workers=len(unique_gpus)) as ex:
        futures = {
            ex.submit(gpu_worker, gpu, by_gpu[gpu], args.dry_run, args.rerun_failed): gpu
            for gpu in unique_gpus
        }
        for fut in as_completed(futures):
            all_results.extend(fut.result())

    elapsed_min = (time.time() - t_start) / 60
    counts: dict[str, int] = {}
    for r in all_results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    counts_str = "  ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    print(f"\n=== Done in {elapsed_min:.1f} min:  {counts_str} ===")

    # CSV with one row per task
    out_csv = ROOT / "results" / "p2_runs.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = ["table", "tag", "dataset", "model", "horizon", "seed", "rank",
              "batch_size", "gpu", "status", "elapsed_s", "run_name", "test_line"]
    with out_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_results)
    print(f"CSV: {out_csv}")

    failures = [r for r in all_results if r["status"] in ("FAIL", "OOM")]
    if failures:
        print(f"\nFailures ({len(failures)}):")
        for r in failures:
            print(f"  {r['status']:4s}  {r['tag']:18s} {r['run_name']}")
        print("\nRe-run with --rerun-failed after fixing batch sizes / configs.")


if __name__ == "__main__":
    main()
