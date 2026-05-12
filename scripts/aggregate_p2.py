"""Aggregate P2's run logs into CSV + markdown for the results table.

Scans every log under logs/p2/ for the trainer's final 'Test | mse=... mae=...'
line, parses (dataset, model, horizon, seed, rank) from the file path, and
emits:

  results/p2_runs_full.csv     — one row per completed run (raw)
  results/p2_summary.csv       — mean & std over seeds, one row per
                                 (dataset, model, horizon, rank)
  results/p2_results_partial.md — drop-in markdown for our slice of Tables 1 & 2

Safe to re-run anytime — incomplete logs are reported as PENDING, not errors.

Usage:
    python scripts/aggregate_p2.py
    python scripts/aggregate_p2.py --only-table 1   # main only
    python scripts/aggregate_p2.py --logs-dir logs/p2 --out-dir results/
"""
from __future__ import annotations

import argparse
import csv
import re
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Order in which we want models to appear in tables
MODEL_ORDER = [
    "dlinear", "lstm", "gru", "segrnn",
    "timemixer", "moderntcn", "itransformer", "patchtst",
    "metatsf_mlp", "metatsf_conv", "metatsf_attn", "metatsf_vglg",
]
MODEL_LABEL = {
    "dlinear": "DLinear", "lstm": "LSTM", "gru": "GRU", "segrnn": "SegRNN",
    "timemixer": "TimeMixer", "moderntcn": "ModernTCN",
    "itransformer": "iTransformer", "patchtst": "PatchTST",
    "metatsf_mlp": "MetaTSF-MLP", "metatsf_conv": "MetaTSF-Conv",
    "metatsf_attn": "MetaTSF-Attn", "metatsf_vglg": "MetaTSF-VGLG",
}

TABLE1_DATASETS = ["electricity", "traffic"]            # P2's slice
TABLE2_DATASETS = ["etth1", "weather", "electricity"]   # rank ablation
TABLE1_HORIZONS = [96, 192, 336, 720]
TABLE2_HORIZONS = [96, 336, 720]

TEST_RE = re.compile(
    r"^Test\s*\|\s*mse=([\d.eE+\-nan]+)\s+mae=([\d.eE+\-nan]+)\s+rmse=([\d.eE+\-nan]+)"
)
PARAMS_RE = re.compile(r"Built model:\s+\S+\s+\|\s+params=([\d,]+)")
TAG_RANK_RE = re.compile(r"^ablation_rank_r(\d+)$")


def parse_run_name(stem: str) -> dict | None:
    """`electricity_metatsf_vglg_h96_s2021` -> dict, or None if unparseable."""
    parts = stem.split("_")
    if len(parts) < 4:
        return None
    if not parts[-1].startswith("s") or not parts[-2].startswith("h"):
        return None
    try:
        seed = int(parts[-1][1:])
        horizon = int(parts[-2][1:])
    except ValueError:
        return None
    return {
        "dataset": parts[0],
        "model": "_".join(parts[1:-2]),
        "horizon": horizon,
        "seed": seed,
    }


def parse_tag(tag: str) -> tuple[str, int | None]:
    m = TAG_RANK_RE.match(tag)
    if m:
        return "ablation_rank", int(m.group(1))
    return tag, None


def parse_log(path: Path) -> dict:
    """Returns dict with status + (test_mse, test_mae, test_rmse, n_params) when complete."""
    info = {"status": "PENDING", "test_mse": None, "test_mae": None,
            "test_rmse": None, "n_params": None}
    try:
        text = path.read_text(errors="replace")
    except OSError:
        info["status"] = "MISSING"
        return info
    if "out of memory" in text.lower():
        info["status"] = "OOM"
    for line in text.splitlines():
        m = PARAMS_RE.search(line)
        if m:
            info["n_params"] = int(m.group(1).replace(",", ""))
        m = TEST_RE.match(line)
        if m:
            mse, mae, rmse = (float(x) if x.lower() != "nan" else float("nan") for x in m.groups())
            info["test_mse"] = mse
            info["test_mae"] = mae
            info["test_rmse"] = rmse
            info["status"] = "NAN" if mse != mse else "OK"  # NaN check
    return info


def scan(logs_dir: Path) -> list[dict]:
    rows = []
    if not logs_dir.exists():
        return rows
    for tag_dir in sorted(logs_dir.iterdir()):
        if not tag_dir.is_dir():
            continue
        kind, rank = parse_tag(tag_dir.name)
        for log in sorted(tag_dir.glob("*.log")):
            parsed = parse_run_name(log.stem)
            if parsed is None:
                continue
            info = parse_log(log)
            rows.append({
                **parsed, "tag": tag_dir.name, "kind": kind, "rank": rank,
                "log_path": str(log.relative_to(ROOT)), **info,
            })
    return rows


def write_full_csv(rows: list[dict], out: Path) -> None:
    fields = ["kind", "tag", "dataset", "model", "horizon", "seed", "rank",
              "n_params", "status", "test_mse", "test_mae", "test_rmse", "log_path"]
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def aggregate(rows: list[dict]) -> list[dict]:
    """Mean & std over seeds, grouped by (kind, dataset, model, horizon, rank)."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        if r["status"] != "OK":
            continue
        key = (r["kind"], r["dataset"], r["model"], r["horizon"], r["rank"])
        groups[key].append(r)
    out = []
    for key, members in sorted(groups.items()):
        mses = [m["test_mse"] for m in members]
        maes = [m["test_mae"] for m in members]
        out.append({
            "kind": key[0], "dataset": key[1], "model": key[2],
            "horizon": key[3], "rank": key[4],
            "n_seeds": len(members),
            "n_params": members[0]["n_params"],
            "mse_mean": statistics.mean(mses),
            "mse_std": statistics.stdev(mses) if len(mses) > 1 else 0.0,
            "mae_mean": statistics.mean(maes),
            "mae_std": statistics.stdev(maes) if len(maes) > 1 else 0.0,
            "seeds": ",".join(sorted(str(m["seed"]) for m in members)),
        })
    return out


def write_summary_csv(summary: list[dict], out: Path) -> None:
    fields = ["kind", "dataset", "model", "horizon", "rank", "n_seeds",
              "n_params", "mse_mean", "mse_std", "mae_mean", "mae_std", "seeds"]
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for s in summary:
            row = dict(s)
            for k in ("mse_mean", "mse_std", "mae_mean", "mae_std"):
                row[k] = f"{row[k]:.6f}"
            w.writerow(row)


# ---------- markdown rendering ----------

def _fmt(x: float | None) -> str:
    if x is None:
        return "—"
    return f"{x:.4f}"


def _lookup(summary: list[dict], **filters):
    for s in summary:
        if all(s.get(k) == v for k, v in filters.items()):
            return s
    return None


def md_table1_perdataset(summary: list[dict], dataset: str) -> str:
    """Per-horizon detail for one dataset (Table 1a/b/f/g style)."""
    label = dataset.capitalize() if dataset != "etth1" else "ETTh1"
    if dataset == "electricity":
        label = "Electricity"
    if dataset == "traffic":
        label = "Traffic"

    out = [
        f"### {label}",
        "",
        "| Model              | h=96 MSE | h=96 MAE | h=192 MSE | h=192 MAE | h=336 MSE | h=336 MAE | h=720 MSE | h=720 MAE | n_seeds |",
        "|--------------------|---------:|---------:|----------:|----------:|----------:|----------:|----------:|----------:|--------:|",
    ]
    for model in MODEL_ORDER:
        cells = []
        n_seeds_seen = []
        for h in TABLE1_HORIZONS:
            s = _lookup(summary, kind="main", dataset=dataset, model=model, horizon=h)
            if s is None:
                cells.extend(["—", "—"])
            else:
                cells.extend([_fmt(s["mse_mean"]), _fmt(s["mae_mean"])])
                n_seeds_seen.append(s["n_seeds"])
        n_str = "/".join(str(n) for n in n_seeds_seen) if n_seeds_seen else "0"
        out.append(f"| {MODEL_LABEL[model]:18s} | " + " | ".join(c.rjust(8) for c in cells) + f" | {n_str:>7s} |")
    return "\n".join(out)


def md_table1_avg(summary: list[dict]) -> str:
    """Headline Table 1: avg-across-horizon MSE per (dataset, model)."""
    out = [
        "## Table 1 — Average MSE across horizons {96, 192, 336, 720}",
        "",
        "P2's slice (Electricity, Traffic). Other datasets filled by P1/P4/P5.",
        "",
        f"| Model              | {' | '.join(d.capitalize().rjust(11) for d in TABLE1_DATASETS)} | Avg (P2) |",
        f"|--------------------|{'|'.join('-' * 12 for _ in TABLE1_DATASETS)}|---------:|",
    ]
    for model in MODEL_ORDER:
        per_ds_avgs = []
        for ds in TABLE1_DATASETS:
            mses = [_lookup(summary, kind="main", dataset=ds, model=model, horizon=h)
                    for h in TABLE1_HORIZONS]
            ok = [m["mse_mean"] for m in mses if m is not None]
            per_ds_avgs.append(sum(ok) / len(ok) if ok else None)
        cells = " | ".join(_fmt(v).rjust(11) for v in per_ds_avgs)
        non_none = [v for v in per_ds_avgs if v is not None]
        overall = sum(non_none) / len(non_none) if non_none else None
        out.append(f"| {MODEL_LABEL[model]:18s} | {cells} | {_fmt(overall).rjust(8)} |")
    return "\n".join(out)


def md_table2(summary: list[dict]) -> str:
    """Rank ablation: rank ∈ {4, 16, 32} on {ETTh1, Weather, Electricity}.

    Rank=8 (full) baseline for Electricity is taken from Table 1's metatsf_vglg.
    For ETTh1 and Weather it has to come from P4/P5's runs (marked as —).
    """
    out = [
        "## Table 2 — VGLG rank ablation (MSE)",
        "",
        "Mean across horizons {96, 336, 720} × 3 seeds.",
        "Rank 8 = full (default config). Rank 8 on ETTh1/Weather comes from P4/P5's "
        "main runs; Electricity rank 8 is taken from our Table 1 metatsf_vglg.",
        "",
        f"| Rank | {' | '.join(d.capitalize().rjust(13) for d in TABLE2_DATASETS)} | Overall |",
        f"|-----:|{'|'.join('-' * 14 for _ in TABLE2_DATASETS)}|--------:|",
    ]
    ranks = [4, 8, 16, 32]
    for rank in ranks:
        per_ds = []
        for ds in TABLE2_DATASETS:
            if rank == 8:
                # full = Table 1 metatsf_vglg
                if ds == "electricity":
                    mses = [_lookup(summary, kind="main", dataset=ds,
                                    model="metatsf_vglg", horizon=h)
                            for h in TABLE2_HORIZONS]
                    ok = [m["mse_mean"] for m in mses if m is not None]
                    per_ds.append(sum(ok) / len(ok) if ok else None)
                else:
                    per_ds.append(None)  # P4/P5
            else:
                mses = [_lookup(summary, kind="ablation_rank", dataset=ds,
                                model="metatsf_vglg", horizon=h, rank=rank)
                        for h in TABLE2_HORIZONS]
                ok = [m["mse_mean"] for m in mses if m is not None]
                per_ds.append(sum(ok) / len(ok) if ok else None)
        cells = " | ".join(_fmt(v).rjust(13) for v in per_ds)
        non_none = [v for v in per_ds if v is not None]
        overall = sum(non_none) / len(non_none) if non_none else None
        label = f"{rank} (full)" if rank == 8 else str(rank)
        out.append(f"| {label:>4s} | {cells} | {_fmt(overall).rjust(7)} |")
    return "\n".join(out)


def write_markdown(summary: list[dict], rows: list[dict], out: Path) -> None:
    # Count of completed runs per (table, dataset)
    counts: dict[tuple, dict[str, int]] = defaultdict(lambda: {"OK": 0, "PENDING": 0, "OOM": 0, "NAN": 0})
    for r in rows:
        counts[(r["kind"], r["dataset"])][r["status"]] += 1

    md = [
        "# P2 results (partial)",
        "",
        "Generated by `scripts/aggregate_p2.py`. Re-run any time during the main loop "
        "to refresh — incomplete runs are reported as gaps (—).",
        "",
        "## Completion status",
        "",
        "| Table | Dataset | OK | Pending | OOM | NaN |",
        "|------:|---------|---:|--------:|----:|----:|",
    ]
    for (kind, ds), c in sorted(counts.items()):
        md.append(f"| {kind:>14s} | {ds:11s} | {c['OK']:2d} | {c['PENDING']:2d} | {c['OOM']:2d} | {c['NAN']:2d} |")

    md += ["", md_table1_avg(summary), ""]
    for ds in TABLE1_DATASETS:
        md += [md_table1_perdataset(summary, ds), ""]
    md += [md_table2(summary), ""]

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(md))


# ---------- entry ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs-dir", default="logs/p2")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--only-table", type=int, choices=[1, 2], default=None,
                    help="Restrict aggregation to one table")
    args = ap.parse_args()

    logs_dir = ROOT / args.logs_dir
    out_dir = ROOT / args.out_dir

    rows = scan(logs_dir)
    if args.only_table == 1:
        rows = [r for r in rows if r["kind"] == "main"]
    elif args.only_table == 2:
        rows = [r for r in rows if r["kind"] == "ablation_rank"]

    if not rows:
        print(f"No logs found under {logs_dir}. Nothing to aggregate.")
        return

    write_full_csv(rows, out_dir / "p2_runs_full.csv")
    summary = aggregate(rows)
    write_summary_csv(summary, out_dir / "p2_summary.csv")
    write_markdown(summary, rows, out_dir / "p2_results_partial.md")

    n_total = len(rows)
    n_ok = sum(1 for r in rows if r["status"] == "OK")
    n_pending = sum(1 for r in rows if r["status"] == "PENDING")
    n_bad = sum(1 for r in rows if r["status"] in ("OOM", "NAN"))
    print(f"Scanned {n_total} log files: {n_ok} OK, {n_pending} pending, {n_bad} OOM/NaN.")
    print(f"  flat CSV:     {out_dir / 'p2_runs_full.csv'}")
    print(f"  summary CSV:  {out_dir / 'p2_summary.csv'}")
    print(f"  markdown:     {out_dir / 'p2_results_partial.md'}")
    if n_bad:
        bad_runs = [r for r in rows if r["status"] in ("OOM", "NAN")]
        print(f"\n{n_bad} bad runs:")
        for r in bad_runs[:20]:
            print(f"  {r['status']:4s} {r['log_path']}")


if __name__ == "__main__":
    main()
