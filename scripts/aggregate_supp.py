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
# All MetaTSF horizons use the flat wd=1e-3 sweep (tuned2). A horizon-aware wd
# (round 3) was investigated — see logs/metatsf_tuned3/, logs/metatsf_h720_scout/
# and Analysis §5 — but rejected: the four mixers disagree on the optimal
# long-horizon wd, so no shared schedule beats flat wd=1e-3 on the 4-mixer mean.
METATSF_TAG_BY_H = {h: "metatsf_tuned2" for h in (96, 192, 336, 720)}
MAIN_TABLE = REPO / "docs" / "results_table.md"
OUT = REPO / "docs" / "supp_method_b_results.md"

SUPP_MODELS = ["patchtst", "moderntcn", "segrnn", "lstm", "gru", "itransformer", "timemixer"]
METATSF_MIXERS = ["metatsf_mlp", "metatsf_conv", "metatsf_attn", "metatsf_vglg"]
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
    "metatsf_mlp": "MetaTSF-MLP",
    "metatsf_conv": "MetaTSF-Conv",
    "metatsf_attn": "MetaTSF-Attn",
    "metatsf_vglg": "MetaTSF-VGLG",
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
    # All 4 MetaTSF mixers share the same tuned recipe. Round 1 (lr scout)
    # picked R3 = 30ep/lr1e-3/cosine/pat8. Round 2 (reg scout) added
    # weight_decay=1e-3 — the runs overfit hard (train loss collapses, val
    # plateaus by epoch 1-4), and wd=1e-3 was the sole scout winner. Round 3
    # (h720 scout) investigated a horizon-aware wd but it was rejected (mixers
    # disagree; see §5), so flat wd=1e-3 is retained for all horizons.
    "metatsf_mlp":  {"epochs": 30, "patience": 8, "lr": 1e-3, "lradj": "cosine", "wd": 1e-3},
    "metatsf_conv": {"epochs": 30, "patience": 8, "lr": 1e-3, "lradj": "cosine", "wd": 1e-3},
    "metatsf_attn": {"epochs": 30, "patience": 8, "lr": 1e-3, "lradj": "cosine", "wd": 1e-3},
    "metatsf_vglg": {"epochs": 30, "patience": 8, "lr": 1e-3, "lradj": "cosine", "wd": 1e-3},
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
    """Combine the 7 supp_method_b baselines + 4 tuned MetaTSF mixers."""
    out: dict = {}
    for model in SUPP_MODELS:
        for ds in DATASETS:
            for h in HORIZONS:
                p = LOG_DIR / f"{ds}_{model}_h{h}_s2021.log"
                out[(model, ds, h)] = parse_log(p)
    # MetaTSF tuned logs live in horizon-specific directories (see METATSF_TAG_BY_H).
    for model in METATSF_MIXERS:
        for ds in DATASETS:
            for h in HORIZONS:
                p = REPO / "logs" / METATSF_TAG_BY_H[h] / f"{ds}_{model}_h{h}_s2021.log"
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
                 "MetaTSF variants were re-tuned under a single shared recipe — `30 ep / lr 1e-3 / "
                 "cosine / patience 8 / weight_decay 1e-3` — reached over three rounds: an lr scout "
                 "(round 1) fixed the first four hyperparameters; a regularization scout (round 2) "
                 "added `weight_decay 1e-3` after the runs were found to overfit heavily; and a "
                 "round 3 investigated a **horizon-aware** weight decay but rejected it (the four "
                 "mixers disagree on the optimal long-horizon wd, so no shared schedule beats flat "
                 "wd=1e-3 — see Analysis §5). All four mixers share this identical recipe "
                 "(preserving the same-backbone thesis). DLinear is the only model unchanged from "
                 "main. Δ% = (supp − main) / main · 100, **negative is better.**")
    lines.append("")
    lines.append("Auto-generated by `scripts/aggregate_supp.py`.")
    lines.append("")

    # ----- Recipe summary --------------------------------------------------
    lines.append("## Recipe overrides applied per model")
    lines.append("")
    lines.append("| Model | Main recipe | Supp recipe |")
    lines.append("|---|---|---|")
    for m in SUPP_MODELS + METATSF_MIXERS:
        r = SUPP_RECIPE[m]
        main_str = "10 ep / lr 1e-4 / step / pat 3"
        if r.get("wd_disp"):
            wd_part = f" / wd {r['wd_disp']}"
        elif r.get("wd"):
            wd_part = f" / wd {r['wd']:.0e}"
        else:
            wd_part = ""
        supp_str = (f"{r['epochs']} ep / lr {r['lr']:.0e} / {r['lradj']} / pat {r['patience']}{wd_part}")
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
    for m in SUPP_MODELS + METATSF_MIXERS:
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
    for m in SUPP_MODELS + METATSF_MIXERS:
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
    for m in SUPP_MODELS + METATSF_MIXERS:
        disp = MODEL_DISPLAY[m]
        ma = main_avg7(disp)
        sa, partial = supp_avg7(m)
        all_rows.append((disp, ma, sa, partial, True))
    # Only DLinear remains unchanged.
    ma = main_avg7("DLinear")
    all_rows.append(("DLinear", ma, ma, False, False))

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
    lines.append("Each cell is the per-dataset mean MSE under each model's **supplementary** "
                 "recipe (7 reruns + 4 retuned MetaTSF). DLinear values are copied from main "
                 "(unchanged). Average column excludes f1weather.")
    lines.append("")
    header_cols = ["Model"] + [d for d in DATASETS] + ["Avg(7)"]
    lines.append("| " + " | ".join(header_cols) + " |")
    lines.append("|" + "|".join(["---"] + ["---:"] * (len(header_cols) - 1)) + "|")

    # Supp baselines + tuned MetaTSF mixers
    for m in SUPP_MODELS + METATSF_MIXERS:
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

    # DLinear only — unchanged from main.
    for unchanged_model in ["DLinear"]:
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
    lines.append("### 4. MetaTSF retuned — lr scout, then regularization")
    lines.append("")
    lines.append("The MetaTSF variants were tuned in stages, always with one recipe shared "
                 "across all four mixers (to keep the same-backbone ablation honest). **Round 1** "
                 "(lr scout) moved from the main recipe to `30 ep / lr 1e-3 / cosine / pat 8`, "
                 "tightening Avg(7) from 0.371–0.381 to 0.361–0.366. **Round 2** started from a "
                 "diagnosis: every MetaTSF run overfits — training loss collapses while validation "
                 "MSE plateaus by epoch 1–4 (Traffic is the extreme case: train 0.08 / val 0.46 / "
                 "test 0.57). `weight_decay` was 0.0. A regularization scout (wd ∈ {1e-4,1e-3,1e-2}, "
                 "dropout ∈ {0.2,0.3}, combos) found **wd = 1e-3** the clean winner — a U-shaped "
                 "minimum on Traffic h96 (0.561 → **0.501** → 0.561) with dropout adding nothing on "
                 "top. Applied across the full sweep it lowers Avg(7) for all four mixers by "
                 "1.2–2.9 %, to **0.3536–0.3587 (spread 0.005)**. The leaderboard effect is real: "
                 "all four MetaTSF variants now beat ModernTCN (0.360), and **MetaTSF-Attn (0.354) "
                 "reaches the top three** — within 0.002 of the iTransformer / PatchTST leaders "
                 "(0.352). VGLG at 0.359 ranks #7/12.")
    lines.append("")
    lines.append("### 5. Regularization closes most of MetaTSF's Traffic gap")
    lines.append("")
    lines.append("Traffic (N=862) was essentially the *entire* Avg(7) deficit between MetaTSF and "
                 "the leaders: under round-1 tuning all four mixers sat at 0.533–0.568 while "
                 "iTransformer was at 0.468, and that ~0.10 per-cell gap, divided across 7 "
                 "datasets, accounted for nearly all of MetaTSF's headline shortfall. wd = 1e-3 "
                 "attacks exactly this — the dense ChannelMLP was memorizing the 862-variate "
                 "covariance. Traffic mean MSE drops to MLP 0.506, Conv 0.516, Attn 0.506, "
                 "**VGLG 0.523** (from 0.568, a −7.9 % cut — the largest of any mixer). VGLG is "
                 "still the weakest MetaTSF mixer on Traffic, but the sibling gap narrows from "
                 "~0.030 to ~0.013, and all four now trail iTransformer by ~0.04–0.055 rather than "
                 "0.10. The residual gap is genuine: explicit variate-attention is a better "
                 "inductive bias than VGLG's 4-statistic gate on 862 weakly-correlated series. "
                 "**Trade-off, stated honestly:** wd = 1e-3 (scouted on h96) is too strong at the "
                 "longest horizon. 25 of 112 MetaTSF cells regress versus the main baseline, 7 of "
                 "them by more than 3 % — almost all at h720 on ETTh2 / Weather / ETTm1 / ETTm2 "
                 "(worst: MLP · ETTh2 · h720, +16 %). The Traffic and short-horizon gains outweigh "
                 "them, so Avg(7) still improves for all four mixers — but this residual h720 "
                 "weakness is real and we tried to remove it (round 3 below).")
    lines.append("")
    lines.append("### 5b. Horizon-aware weight decay was investigated and rejected (negative result)")
    lines.append("")
    lines.append("The natural fix for the h720 regressions is to relax weight decay at long "
                 "horizons. Round 3 built a 4-point wd curve per (dataset, horizon) — reusing the "
                 "existing wd=0 (round 1) and wd=1e-3 (round 2) endpoints and scouting the "
                 "intermediate {1e-4, 5e-4} — then chose the schedule on the **full 4-mixer family** "
                 "rather than on VGLG alone. The result is a clean negative: the mixers genuinely "
                 "disagree on the optimal long-horizon wd. At h336 the family mean is best at wd=1e-3 "
                 "(0.3733) — only VGLG prefers 5e-4. At h720 the family mean barely favours wd=1e-4 "
                 "(0.4321 vs 0.4339 at 1e-3, a 0.4 % / noise-level edge), and the per-mixer optima "
                 "are split three ways: MLP and Conv want wd=0, Attn wants wd=1e-3, VGLG wants "
                 "wd=1e-4. Applying the best family schedule (drop only h720 to 1e-4) moves the "
                 "family Avg(7) by 0.0004 (0.3563 → 0.3559, within seed noise), *increases* the "
                 ">3 % regression count from 7 to 9 (lowering h720 wd helps VGLG/MLP/Conv but hurts "
                 "Attn, which loses its top-3 spot), and does not touch the worst cell: "
                 "MLP · ETTh2 · h720 is +14–16 % over main at *every* wd including 0, i.e. it is an "
                 "intrinsic recipe interaction (that cell simply prefers the short 10-epoch main "
                 "budget), not a weight-decay artifact. We therefore keep the flat wd=1e-3 recipe "
                 "as canonical; the horizon-aware runs live in `logs/metatsf_tuned3/` and "
                 "`logs/metatsf_h720_scout/` for the record.")
    lines.append("")
    lines.append("### 6. What this means for the paper / pre")
    lines.append("")
    lines.append("- **MetaTSF backbone is competitive-to-strong under fair recipes** — Attn / Conv "
                 "/ MLP rank #3 / #5 / #6 on Avg(7), all beating ModernTCN, SegRNN and the RNNs; "
                 "the best MetaTSF variant (Attn, 0.354) is within 0.002 of the iTransformer / "
                 "PatchTST leaders (0.352).")
    lines.append("- **VGLG (#7) is in the same tight band**. The round-2 regularization narrowed "
                 "its Traffic deficit substantially, but it remains the weakest MetaTSF mixer "
                 "there: its variate-gate still does not beat plain variate-attention on 862 "
                 "variates.")
    lines.append("- **The four-mixer ablation stays clean**: same backbone + same recipe + same "
                 "data → Avg(7) spread of just 0.005, and < 0.5 % MSE differences on most datasets. "
                 "This is the internal consistency you want from a methodology paper, even if it "
                 "weakens the case that VGLG's gate is the decisive component.")
    lines.append("- **The KD result on f1weather (Avg=0.570 vs Chronos teacher 0.752) remains "
                 "the most defensible positive finding** — it is not affected by this supplementary "
                 "study because that dataset is intentionally excluded here.")
    lines.append("")

    OUT.write_text("\n".join(lines))
    print(f"Wrote {OUT} ({len(lines)} lines)")
    print(f"Coverage: {completed}/{total}")


if __name__ == "__main__":
    main()
