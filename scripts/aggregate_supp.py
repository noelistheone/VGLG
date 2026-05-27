#!/usr/bin/env python
"""Aggregate Method B supplementary runs and write the results doc.

Parses the 196 supp logs in `logs/supp_method_b/`, joins each (model, dataset,
horizon) against the main results in `docs/results_table.md`, computes Δ%, and
emits `docs/supp_method_b_results.md`.

Usage:
    python scripts/aggregate_supp.py
"""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LOG_DIR = REPO / "logs" / "supp_method_b"
MAIN_TABLE = REPO / "docs" / "results_table.md"
OUT = REPO / "docs" / "supp_method_b_results.md"

SUPP_MODELS = ["patchtst", "moderntcn", "segrnn", "lstm", "gru", "itransformer", "timemixer"]
DATASETS = ["etth1", "etth2", "ettm1", "ettm2", "weather", "electricity", "traffic"]
HORIZONS = [96, 192, 336, 720]

# How results_table.md spells each model name in its tables.
MODEL_DISPLAY = {
    "patchtst": "PatchTST",
    "moderntcn": "ModernTCN",
    "segrnn": "SegRNN",
    "lstm": "LSTM",
    "gru": "GRU",
    "itransformer": "iTransformer",
    "timemixer": "TimeMixer",
}

# Recipe overrides we applied per model (for the report table).
SUPP_RECIPE = {
    "patchtst":     {"epochs": 30, "patience": 8,  "lr": 1e-4, "lradj": "cosine"},
    "moderntcn":    {"epochs": 50, "patience": 10, "lr": 1e-3, "lradj": "cosine"},
    "segrnn":       {"epochs": 30, "patience": 8,  "lr": 1e-3, "lradj": "cosine"},
    "timemixer":    {"epochs": 10, "patience": 5,  "lr": 1e-2, "lradj": "cosine"},
    "itransformer": {"epochs": 10, "patience": 3,  "lr": 5e-4, "lradj": "step"},
    "lstm":         {"epochs": 10, "patience": 3,  "lr": 1e-3, "lradj": "step"},
    "gru":          {"epochs": 10, "patience": 3,  "lr": 1e-3, "lradj": "step"},
}


# ──────────────────────────────────────────────────────────────────────────
# Parse a supp log file → (mse, mae, rmse, last_epoch, early_stopped)
TEST_LINE = re.compile(r"^Test \| mse=([0-9.eE+-]+) mae=([0-9.eE+-]+) rmse=([0-9.eE+-]+)")
EPOCH_LINE = re.compile(r"^Epoch (\d+) \|")
ES_LINE = re.compile(r"^Early stopping at epoch (\d+)\.")


def parse_log(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return None
    test = None
    last_epoch = 0
    early_stopped = False
    for line in text.splitlines():
        m = TEST_LINE.match(line)
        if m:
            test = (float(m[1]), float(m[2]), float(m[3]))
        m = EPOCH_LINE.match(line)
        if m:
            last_epoch = max(last_epoch, int(m[1]))
        if ES_LINE.match(line):
            early_stopped = True
    if test is None:
        return None
    return {
        "mse": test[0], "mae": test[1], "rmse": test[2],
        "last_epoch": last_epoch, "early_stopped": early_stopped,
    }


# ──────────────────────────────────────────────────────────────────────────
# Parse main results table → {(dataset, model_display, horizon): (mse, mae)}
DATASET_HEADER = re.compile(r"^## Table 1[a-h] — (\w+) \(per-horizon detail\)")
NUM = r"[0-9.]+"


def parse_main_table() -> dict[tuple[str, str, int], tuple[float, float]]:
    out: dict[tuple[str, str, int], tuple[float, float]] = {}
    current_dataset = None
    for line in MAIN_TABLE.read_text().splitlines():
        m = DATASET_HEADER.match(line)
        if m:
            current_dataset = m.group(1).lower()
            continue
        if current_dataset is None:
            continue
        if not line.startswith("|"):
            continue
        if "h=96 MSE" in line or "---" in line:
            continue
        cells = [c.strip() for c in line.split("|")[1:-1]]
        if len(cells) < 9:
            continue
        # Cell 0 is model name (may have markdown bold).
        model_raw = cells[0].replace("**", "").strip()
        if model_raw.startswith("_"):
            continue
        try:
            vals = [float(c) for c in cells[1:9]]
        except ValueError:
            continue
        # vals = [mse96, mae96, mse192, mae192, mse336, mae336, mse720, mae720]
        for i, h in enumerate(HORIZONS):
            out[(current_dataset, model_raw, h)] = (vals[2*i], vals[2*i+1])
    return out


# ──────────────────────────────────────────────────────────────────────────
def collect_supp() -> dict[tuple[str, str, int], dict | None]:
    out: dict = {}
    for model in SUPP_MODELS:
        for ds in DATASETS:
            for h in HORIZONS:
                p = LOG_DIR / f"{ds}_{model}_h{h}_s2021.log"
                out[(model, ds, h)] = parse_log(p)
    return out


# ──────────────────────────────────────────────────────────────────────────
def fmt_delta_pct(supp: float, main: float) -> str:
    if main <= 0:
        return "n/a"
    pct = (supp - main) / main * 100
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.1f}%"


def main() -> None:
    main_vals = parse_main_table()
    supp_vals = collect_supp()

    # Build the report.
    lines: list[str] = []
    lines.append("# Method B — Supplementary Experiment Results")
    lines.append("")
    lines.append("Each baseline was rerun under a recipe closer to its original paper's training "
                 "budget / learning rate / lr schedule (see `supp_experiments_plan.md`). The four "
                 "MetaTSF variants and DLinear are unchanged from the main results — they are the "
                 "fixed reference. Δ% = (supp − main) / main · 100, **negative is better.**")
    lines.append("")
    lines.append("Auto-generated by `scripts/aggregate_supp.py`.")
    lines.append("")

    # ----- Recipe summary --------------------------------------------------
    lines.append("## Recipe overrides applied per model")
    lines.append("")
    lines.append("| Model | Main recipe | Supp recipe |")
    lines.append("|---|---|---|")
    for m in SUPP_MODELS:
        r = SUPP_RECIPE[m]
        main_str = "10 ep / lr 1e-4 / step / pat 3"
        supp_str = (f"{r['epochs']} ep / lr {r['lr']:.0e} / {r['lradj']} / pat {r['patience']}")
        lines.append(f"| {MODEL_DISPLAY[m]} | {main_str} | {supp_str} |")
    lines.append("")

    # ----- Run coverage ----------------------------------------------------
    completed = sum(1 for v in supp_vals.values() if v is not None)
    total = len(supp_vals)
    missing = [k for k, v in supp_vals.items() if v is None]
    lines.append(f"## Coverage: {completed}/{total} runs completed")
    lines.append("")
    if missing:
        lines.append("Missing runs:")
        for model, ds, h in missing:
            lines.append(f"- `{ds}_{model}_h{h}`")
    else:
        lines.append("All runs completed.")
    lines.append("")

    # ----- Per-model summary (Δ% averaged over all (dataset, horizon) ------
    lines.append("## Per-model summary (mean Δ% across all dataset × horizon cells)")
    lines.append("")
    lines.append("| Model | n | mean Δ% | min Δ% | max Δ% | # better | # worse | # >10% better |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for m in SUPP_MODELS:
        deltas = []
        for ds in DATASETS:
            for h in HORIZONS:
                supp = supp_vals.get((m, ds, h))
                main_key = (ds, MODEL_DISPLAY[m], h)
                if supp is None or main_key not in main_vals:
                    continue
                main_mse = main_vals[main_key][0]
                if main_mse <= 0:
                    continue
                deltas.append((supp["mse"] - main_mse) / main_mse * 100)
        if not deltas:
            lines.append(f"| {MODEL_DISPLAY[m]} | 0 | — | — | — | — | — | — |")
            continue
        better = sum(1 for d in deltas if d < 0)
        worse = sum(1 for d in deltas if d > 0)
        big_better = sum(1 for d in deltas if d <= -10)
        mean = sum(deltas) / len(deltas)
        lines.append(
            f"| {MODEL_DISPLAY[m]} | {len(deltas)} | {mean:+.1f}% | {min(deltas):+.1f}% | "
            f"{max(deltas):+.1f}% | {better} | {worse} | {big_better} |"
        )
    lines.append("")

    # ----- Full Δ table (per model: rows datasets × cols horizons) ---------
    lines.append("## Full per-(model, dataset, horizon) Δ table")
    lines.append("")
    lines.append("Format: `supp / main / Δ%`. Cell shows `—` if the run is missing.")
    lines.append("")
    for m in SUPP_MODELS:
        lines.append(f"### {MODEL_DISPLAY[m]}")
        lines.append("")
        lines.append("| Dataset | h=96 | h=192 | h=336 | h=720 |")
        lines.append("|---|---|---|---|---|")
        for ds in DATASETS:
            cells = []
            for h in HORIZONS:
                supp = supp_vals.get((m, ds, h))
                main_key = (ds, MODEL_DISPLAY[m], h)
                if supp is None or main_key not in main_vals:
                    cells.append("—")
                    continue
                main_mse = main_vals[main_key][0]
                cells.append(f"{supp['mse']:.3f} / {main_mse:.3f} / **{fmt_delta_pct(supp['mse'], main_mse)}**")
            lines.append(f"| {ds} | " + " | ".join(cells) + " |")
        lines.append("")

    # ----- Headline: main vs supp Avg(7), same 7-dataset metric -----------
    lines.append("## Headline — Avg(7) MSE, main vs supp (same 7-dataset metric, **excludes f1weather**)")
    lines.append("")
    lines.append("Both columns use the same 7 datasets (ETTh1/h2, ETTm1/m2, Weather, Electricity, "
                 "Traffic) averaged over 4 horizons. **f1weather is intentionally excluded from all "
                 "scoring in this report** — it was not rerun and is a noisy custom dataset where "
                 "all models score ≈2.0 MSE, which would distort cross-model comparisons.")
    lines.append("")
    lines.append("| Model | Main Avg(7) | Supp Avg(7) | Δ vs main | Rerun? |")
    lines.append("|---|---:|---:|---:|---|")

    def main_avg7(display_name: str) -> float | None:
        vals = []
        for ds in DATASETS:
            mses = [main_vals[(ds, display_name, h)][0]
                    for h in HORIZONS if (ds, display_name, h) in main_vals]
            if len(mses) == len(HORIZONS):
                vals.append(sum(mses) / len(mses))
        return sum(vals) / len(vals) if len(vals) == len(DATASETS) else None

    def supp_avg7(model: str) -> tuple[float | None, bool]:
        """Returns (avg, complete)."""
        vals, partial = [], False
        for ds in DATASETS:
            mses = [supp_vals[(model, ds, h)]["mse"]
                    for h in HORIZONS if supp_vals.get((model, ds, h)) is not None]
            if len(mses) == len(HORIZONS):
                vals.append(sum(mses) / len(mses))
            elif mses:
                vals.append(sum(mses) / len(mses))
                partial = True
        if len(vals) != len(DATASETS):
            return None, partial
        return sum(vals) / len(vals), partial

    all_rows = []  # (label, main_avg, supp_avg, partial, rerun_flag)
    for m in SUPP_MODELS:
        disp = MODEL_DISPLAY[m]
        ma = main_avg7(disp)
        sa, partial = supp_avg7(m)
        all_rows.append((disp, ma, sa, partial, True))
    for unchanged in ["DLinear", "MetaTSF-MLP", "MetaTSF-Conv", "MetaTSF-Attn", "MetaTSF-VGLG"]:
        ma = main_avg7(unchanged)
        all_rows.append((unchanged, ma, ma, False, False))

    # Sort by supp avg (best first), MetaTSF/DLinear at original positions.
    all_rows.sort(key=lambda r: r[2] if r[2] is not None else 999)
    for label, ma, sa, partial, rerun in all_rows:
        ma_s = f"{ma:.3f}" if ma is not None else "—"
        sa_s = f"{sa:.3f}{'†' if partial else ''}" if sa is not None else "—"
        if ma is not None and sa is not None and ma > 0:
            d = (sa - ma) / ma * 100
            d_s = f"{'+' if d >= 0 else ''}{d:.1f}%"
        else:
            d_s = "—"
        flag = "rerun" if rerun else "_unchanged_"
        # Mark MetaTSF rows in italics.
        if label.startswith("MetaTSF") or label == "DLinear":
            label = f"_{label}_"
        lines.append(f"| {label} | {ma_s} | {sa_s} | {d_s} | {flag} |")
    lines.append("")
    lines.append("† = partial coverage; see Coverage section.")
    lines.append("")

    # ----- Updated leaderboard --------------------------------------------
    # Average MSE per (model, dataset) across the 4 horizons. Drop f1weather
    # (not rerun) and dataset-average across the 7 covered datasets.
    lines.append("## Per-dataset breakdown — mean MSE under each model's own recipe")
    lines.append("")
    lines.append("Each cell is the per-dataset mean MSE under the **supplementary** recipe for "
                 "the 7 baselines. MetaTSF and DLinear values are copied from main (unchanged). "
                 "Average column excludes f1weather.")
    lines.append("")
    header_cols = ["Model"] + [d for d in DATASETS] + ["Avg(7)"]
    lines.append("| " + " | ".join(header_cols) + " |")
    lines.append("|" + "|".join(["---"] + ["---:"] * (len(header_cols) - 1)) + "|")

    # Supp baselines
    for m in SUPP_MODELS:
        row = [MODEL_DISPLAY[m]]
        ds_avgs = []
        for ds in DATASETS:
            supp_mses = [
                supp_vals[(m, ds, h)]["mse"]
                for h in HORIZONS
                if supp_vals.get((m, ds, h)) is not None
            ]
            if len(supp_mses) == len(HORIZONS):
                avg = sum(supp_mses) / len(supp_mses)
                row.append(f"{avg:.3f}")
                ds_avgs.append(avg)
            elif supp_mses:
                avg = sum(supp_mses) / len(supp_mses)
                row.append(f"{avg:.3f}†")  # partial
                ds_avgs.append(avg)
            else:
                row.append("—")
        if ds_avgs:
            row.append(f"**{sum(ds_avgs) / len(ds_avgs):.3f}**")
        else:
            row.append("—")
        lines.append("| " + " | ".join(row) + " |")

    # MetaTSF + DLinear: copy from main average per dataset (across 4 horizons).
    for unchanged_model in ["DLinear", "MetaTSF-MLP", "MetaTSF-Conv", "MetaTSF-Attn", "MetaTSF-VGLG"]:
        # These rows exist in the main Table 1 (the 9-dataset average row).
        # Pull from main_vals: average across 4 horizons per dataset.
        row = [f"_{unchanged_model}_"]
        ds_avgs = []
        for ds in DATASETS:
            mses = []
            for h in HORIZONS:
                key = (ds, unchanged_model, h)
                if key in main_vals:
                    mses.append(main_vals[key][0])
            if len(mses) == len(HORIZONS):
                avg = sum(mses) / len(mses)
                row.append(f"{avg:.3f}")
                ds_avgs.append(avg)
            else:
                row.append("—")
        if ds_avgs:
            row.append(f"_{sum(ds_avgs) / len(ds_avgs):.3f}_")
        else:
            row.append("—")
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    lines.append("† partial coverage — see Coverage section.")
    lines.append("")
    lines.append("Underlined / italicised rows are reference values copied from main results "
                 "(unchanged in supp).")
    lines.append("")

    # ----- Analysis prose --------------------------------------------------
    lines.append("---")
    lines.append("")
    lines.append("## Analysis")
    lines.append("")
    lines.append("### 1. The original \"0.55–0.59 MSE cluster\" was an artifact of the unified recipe")
    lines.append("")
    lines.append("In the main results (uniform 10-epoch / lr=1e-4 / step-halving / patience=3 across "
                 "all 12 models), 7 of the 8 baselines cluster in a narrow Avg(7) band of ~0.348 to "
                 "~0.439. Under each model's own \"fair\" recipe, that band shifts and tightens to "
                 "**0.352 – 0.370** — a span of 0.018 instead of 0.091. The cluster narrowing is "
                 "the strongest single piece of evidence that training recipe was confounding the "
                 "main results: models that previously looked very different (e.g. GRU at 0.439 vs "
                 "PatchTST at 0.360) are within 5 % of each other once trained appropriately.")
    lines.append("")
    lines.append("### 2. The biggest beneficiaries are the simple baselines we under-trained")
    lines.append("")
    lines.append("LSTM, GRU, and TimeMixer improve by 10–17 % on average. The shared cause is that "
                 "they all used learning rates 5–100× higher than ours in their original setups, and "
                 "our LR=1e-4 essentially froze them at the initial loss landscape. This means the "
                 "main paper's framing of \"MetaTSF beats simple RNNs by a wide margin\" was an LR-tuning "
                 "advantage we accidentally granted to ourselves.")
    lines.append("")
    lines.append("### 3. ModernTCN got worse, not better")
    lines.append("")
    lines.append("ModernTCN is the only baseline whose Avg(7) regressed (+3.4 %). The 50-epoch + "
                 "cosine recipe overtrains it on the small ETT datasets (cells like ETTh1 h=720 go "
                 "+31 % worse). Its main-recipe 10-epoch budget was apparently a sweet spot we "
                 "stumbled onto, not a handicap. This is a useful counter-example: \"longer training "
                 "with cosine\" is not universally better, and our supplementary recipe is itself "
                 "imperfect for some architectures.")
    lines.append("")
    lines.append("### 4. MetaTSF / VGLG rank in the bottom half of the supplementary leaderboard")
    lines.append("")
    lines.append("Under the Avg(7) metric, MetaTSF-VGLG sits at 0.378 — worse than every rerun "
                 "baseline (best: PatchTST/iTransformer at 0.352; worst rerun: LSTM/GRU at ~0.370). "
                 "It also already ranked #10/12 under the original main metric once f1weather is "
                 "removed (the 8-dataset Avg=0.581 in `results_table.md` was inflated by f1weather "
                 "where every model scores ≈2.0). So the supplementary results do not change VGLG's "
                 "relative position — they confirm it.")
    lines.append("")
    lines.append("### 5. What this means for the paper / pre")
    lines.append("")
    lines.append("- **VGLG as a mixer architecture is not validated as superior** by these "
                 "results; under fair training recipes, simple baselines match or beat it.")
    lines.append("- **The MetaTSF backbone separation idea is still useful** — the four-mixer "
                 "internal ablation (MLP / Conv / Attn / VGLG sharing one backbone) is a clean "
                 "methodological contribution, even if no single mixer wins clearly.")
    lines.append("- **The honest claim** is now: training-recipe choices dominate architectural "
                 "differences in this benchmark slice, and VGLG's specific gate / low-rank-global "
                 "design did not yield architecture-level gains beyond what fair-recipe baselines "
                 "already provide.")
    lines.append("- **The KD result on f1weather (Avg=0.570 vs Chronos teacher 0.752) remains "
                 "the most defensible positive finding** — it is not affected by this supplementary "
                 "study because that dataset is intentionally excluded here.")
    lines.append("")

    OUT.write_text("\n".join(lines))
    print(f"Wrote {OUT} ({len(lines)} lines)")
    print(f"Coverage: {completed}/{total}")


if __name__ == "__main__":
    main()
