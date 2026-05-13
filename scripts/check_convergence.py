"""Check whether P1 sweep runs converged or were cut off at the epoch budget.

Parses every logs/main/*.log, extracts per-epoch val_mse, and classifies each
run as one of:
  CONVERGED      best_epoch < last_epoch  (early stopping kicked in or val loss
                                           hit its minimum and started rising)
  RAN_FULL_FLAT  last_epoch == max_epochs AND val_mse still trending down
                 (improvement < 1% per epoch over last 3) — likely OK
  RAN_FULL_DOWN  last_epoch == max_epochs AND val_mse improving > 1% per epoch
                 (clearly under-trained; needs more epochs)

Usage:
    python scripts/check_convergence.py
    python scripts/check_convergence.py --max-epochs 10
"""
from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EPOCH_RE = re.compile(r"Epoch (\d+) \| train_loss=([\d.]+) val_mse=([\d.]+) val_mae=([\d.]+)")
LOG_NAME_RE = re.compile(r"^(?P<dataset>[a-z0-9]+)_(?P<model>[a-z_]+)_h(?P<h>\d+)_s(?P<seed>\d+)\.log$")


def parse_epochs(path: Path) -> list[tuple[int, float, float]]:
    """Returns list of (epoch, val_mse, val_mae)."""
    out = []
    try:
        text = path.read_text(errors="ignore")
    except OSError:
        return out
    for m in EPOCH_RE.finditer(text):
        out.append((int(m.group(1)), float(m.group(3)), float(m.group(4))))
    return out


def classify(epochs: list[tuple[int, float, float]], max_epochs: int) -> tuple[str, dict]:
    if not epochs:
        return "NO_EPOCHS", {}
    last_epoch = epochs[-1][0]
    val = [e[1] for e in epochs]
    best_epoch = epochs[val.index(min(val))][0]
    info = {
        "last_epoch": last_epoch,
        "best_epoch": best_epoch,
        "best_val_mse": min(val),
        "final_val_mse": val[-1],
    }
    if last_epoch < max_epochs or best_epoch < last_epoch:
        return "CONVERGED", info
    # ran the full budget; was val_mse still meaningfully decreasing in the last 3?
    if len(val) >= 3:
        recent_drop = (val[-3] - val[-1]) / max(val[-3], 1e-8)
        info["recent_rel_drop"] = recent_drop
        if recent_drop > 0.01:                # >1% decrease over last 3 epochs
            return "RAN_FULL_DOWN", info
    return "RAN_FULL_FLAT", info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="logs/main")
    ap.add_argument("--max-epochs", type=int, default=10)
    ap.add_argument("--show", default="RAN_FULL_DOWN",
                    help="Comma-separated list of categories to detail-list")
    args = ap.parse_args()

    log_dir = ROOT / args.logs
    if not log_dir.exists():
        raise SystemExit(f"Log dir not found: {log_dir}")

    show_set = set(args.show.split(","))
    cats = Counter()
    by_model: dict[str, Counter] = defaultdict(Counter)
    detail: list[tuple[str, dict]] = []
    n_total = 0
    for log_path in sorted(log_dir.glob("*.log")):
        m = LOG_NAME_RE.match(log_path.name)
        if not m:
            continue
        epochs = parse_epochs(log_path)
        if not epochs:
            continue
        n_total += 1
        cat, info = classify(epochs, args.max_epochs)
        cats[cat] += 1
        by_model[m["model"]][cat] += 1
        if cat in show_set:
            detail.append((log_path.name, info))

    print(f"Analyzed {n_total} runs from {log_dir}")
    print()
    print("=== Overall convergence breakdown ===")
    for cat in ["CONVERGED", "RAN_FULL_FLAT", "RAN_FULL_DOWN", "NO_EPOCHS"]:
        n = cats[cat]
        pct = 100 * n / max(n_total, 1)
        print(f"  {cat:18s} {n:4d}  ({pct:5.1f}%)")
    print()
    print("=== By model (CONVERGED / RAN_FULL_FLAT / RAN_FULL_DOWN) ===")
    for model in sorted(by_model):
        c = by_model[model]
        print(f"  {model:18s}  {c['CONVERGED']:3d} / {c['RAN_FULL_FLAT']:3d} / {c['RAN_FULL_DOWN']:3d}")
    print()
    if detail:
        print(f"=== Runs in show category ({args.show}) ===")
        for name, info in detail[:40]:
            drop = info.get("recent_rel_drop", 0)
            print(f"  {name:60s}  last_ep={info['last_epoch']}  best_ep={info['best_epoch']}  "
                  f"best_val={info['best_val_mse']:.4f}  recent_drop={drop:.1%}")
        if len(detail) > 40:
            print(f"  ... and {len(detail) - 40} more")


if __name__ == "__main__":
    main()
