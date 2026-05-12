"""Skip P2's Table 2 (rank ablation) without restarting the launcher.

Trick: the launcher checks `prior_status()` before each task, which inspects the
last 4 KB of the run's log for a 'Test |' line. We pre-create such a stub log
for every T2 run, so the worker will SKIP them when its turn comes.

Net effect:
  * Currently in-flight T1 tasks finish normally (no compute wasted).
  * Each GPU finishes its 96 T1 tasks, then breezes through 27 instant SKIPs.
  * Launcher exits cleanly.
  * No T2 trainer subprocess is ever spawned.

To un-cancel: remove the stub logs and re-run with `--only-table 2 --rerun-failed`.
    rm -rf logs/p2/ablation_rank_r4 logs/p2/ablation_rank_r16 logs/p2/ablation_rank_r32
    python scripts/run_p2_main.py --gpus 4 5 6 --only-table 2

The aggregator (aggregate_p2.py) reads the same stubs but the real metric regex
won't match, so they show up as PENDING (not OK), keeping the partial markdown
honest.

Usage:
    python scripts/cancel_p2_table2.py            # dry-run, lists what would be created
    python scripts/cancel_p2_table2.py --yes      # actually create stubs
"""
from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATASETS = ["etth1", "weather", "electricity"]
RANKS = [4, 16, 32]
HORIZONS = [96, 336, 720]
SEEDS = [2021, 2022, 2023]

STUB = (
    "# This log is a CANCELLATION STUB written by scripts/cancel_p2_table2.py.\n"
    "# No trainer was run. The launcher reads this file's 'Test |' line as a\n"
    "# completion marker and skips the task. The aggregator's metric regex\n"
    "# will not match, so the run shows up as PENDING in summary tables.\n"
    "Test | CANCELLED (no metrics — Table 2 ablation skipped)\n"
)


def stub_paths() -> list[Path]:
    out = []
    for rank in RANKS:
        tag = f"ablation_rank_r{rank}"
        for ds in DATASETS:
            for h in HORIZONS:
                for seed in SEEDS:
                    run_name = f"{ds}_metatsf_vglg_h{h}_s{seed}"
                    out.append(ROOT / "logs" / "p2" / tag / f"{run_name}.log")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yes", action="store_true", help="Actually write the stubs")
    args = ap.parse_args()

    paths = stub_paths()
    existing = [p for p in paths if p.exists()]
    new = [p for p in paths if not p.exists()]

    print(f"Total T2 stubs to manage: {len(paths)} (= 3 ranks × 3 datasets × 3 horizons × 3 seeds)")
    print(f"  Already exist: {len(existing)}")
    print(f"  Will create:   {len(new)}")
    print()
    if existing:
        print("Examples already on disk:")
        for p in existing[:3]:
            print(f"  {p.relative_to(ROOT)}")
        print()

    if not args.yes:
        print("Dry-run only. Re-run with --yes to write stubs and cancel Table 2.")
        return

    for p in paths:
        if p.exists():
            continue
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(STUB)

    print(f"Wrote {len(new)} stubs. The running launcher will now SKIP all T2 tasks.")
    print()
    print("To restore Table 2 later:")
    print("  rm -rf logs/p2/ablation_rank_r4 logs/p2/ablation_rank_r16 logs/p2/ablation_rank_r32")
    print("  python scripts/run_p2_main.py --gpus 4 5 6 --only-table 2")


if __name__ == "__main__":
    main()
