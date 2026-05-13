"""Regenerate docs/results_table.md from logs/main logs.

Idempotent: scans every log, parses the final 'Test | mse=... mae=...' line,
groups by (dataset, model, horizon), averages across seeds, and rewrites the
markdown tables. Supports both flat logs/main/*.log files and Hydra-style
logs/main/<run_name>/trainer.log directories. Cells with no completed runs
render as `—`.

Usage:
    python scripts/update_results_table.py
    python scripts/update_results_table.py --logs logs/main --out docs/results_table.md
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from omegaconf import OmegaConf

from src.data import build_dataloaders
from src.models import build_model
from src.train.trainer import evaluate

DATASETS = ["etth1", "etth2", "ettm1", "ettm2", "weather", "electricity", "traffic", "f1weather"]
DATASET_LABELS = {
    "etth1": "ETTh1", "etth2": "ETTh2", "ettm1": "ETTm1", "ettm2": "ETTm2",
    "weather": "Weather", "electricity": "Electricity", "traffic": "Traffic",
    "f1weather": "f1weather",
}
HORIZONS = [96, 192, 336, 720]

MODELS = [
    ("dlinear", "DLinear", "Linear (no mixer)"),
    ("lstm", "LSTM", "RNN (classic)"),
    ("gru", "GRU", "RNN (classic)"),
    ("segrnn", "SegRNN", "Modern RNN"),
    ("timemixer", "TimeMixer", "MLP (multi-scale)"),
    ("moderntcn", "ModernTCN", "CNN (large kernel)"),
    ("itransformer", "iTransformer", "Trf. (inverted)"),
    ("patchtst", "PatchTST", "Trf. (patch)"),
    ("metatsf_mlp", "**MetaTSF-MLP**", "**Ours / MLP**"),
    ("metatsf_conv", "**MetaTSF-Conv**", "**Ours / Conv**"),
    ("metatsf_attn", "**MetaTSF-Attn**", "**Ours / Attn**"),
    ("metatsf_vglg", "**MetaTSF-VGLG**", "**Ours / VGLG**"),
    # Distillation rows: Chronos zero-shot teacher + our student trained with KD
    ("chronos_zs",     "_Chronos-Bolt_ ZS", "Foundation"),
    ("metatsf_vglg_kd", "**MetaTSF-VGLG + KD**", "**Ours + KD**"),
]

# Table 2: VGLG ablation variants. Each runs on (ETTh1, Weather, Electricity)
# x (h=96, 336, 720) x 3 seeds = 27 runs per variant.
ABLATION_DATASETS = ["etth1", "weather", "electricity"]
ABLATION_HORIZONS = [96, 336, 720]
ABLATION_VARIANTS = [
    ("full",         "**MetaTSF-VGLG (full)**"),
    ("fixed_gate",   "– fixed gate g=0.5"),
    ("local_only",   "– local only (g=1)"),
    ("global_only",  "– global only (g=0)"),
    ("kernel_15",    "kernel = 15 (default 31)"),
    ("kernel_51",    "kernel = 51"),
    ("rank_4",       "rank = 4 (default 8)"),
    ("rank_16",      "rank = 16"),
    ("rank_32",      "rank = 32"),
    ("no_revin",     "no RevIN"),
]

LOG_NAME_RE = re.compile(r"^(?P<dataset>[a-z0-9]+)_(?P<model>[a-z_]+)_h(?P<h>\d+)_s(?P<seed>\d+)\.log$")
CKPT_NAME_RE = re.compile(r"^(?P<dataset>[a-z0-9]+)_(?P<model>[a-z_]+)_h(?P<h>\d+)_s(?P<seed>\d+)\.pt$")
TEST_LINE_RE = re.compile(r"Test \| mse=([\d.]+)\s+mae=([\d.]+)\s+rmse=([\d.]+)")
PARAMS_RE = re.compile(r"params=([\d,]+)")


def parse_log(path: Path) -> tuple[float, float, int | None] | None:
    """Return (mse, mae, n_params) or None if the log is incomplete."""
    try:
        text = path.read_text(errors="ignore")
    except OSError:
        return None
    m = TEST_LINE_RE.search(text)
    if not m:
        return None
    mse, mae = float(m.group(1)), float(m.group(2))
    pm = PARAMS_RE.search(text)
    n_params = int(pm.group(1).replace(",", "")) if pm else None
    return mse, mae, n_params


def collect(log_dirs: Path | list[Path]) -> tuple[dict, dict]:
    """Returns (results, params) where:
        results[(dataset, model, horizon)] -> list of (mse, mae) across seeds
        params[model] -> the parameter count seen for that model

    `log_dirs` accepts a single Path (back-compat) or a list of Paths so both
    logs/main and logs/distill can be aggregated together.
    """
    if isinstance(log_dirs, Path):
        log_dirs = [log_dirs]
    results: dict[tuple[str, str, int], list[tuple[float, float]]] = defaultdict(list)
    params: dict[str, int] = {}
    log_paths = []
    for d in log_dirs:
        if not d.exists():
            continue
        log_paths.extend(d.glob("*.log"))
        log_paths.extend(d.glob("*/trainer.log"))
    for log_path in sorted(log_paths):
        run_name = log_path.stem
        if log_path.name == "trainer.log":
            run_name = log_path.parent.name
        # The distill dispatcher writes <dataset>_metatsf_vglg_kd_h..._s....log
        # so we strip the trailing "_kd" suffix and remap to model key
        # "metatsf_vglg_kd" via the existing regex.
        m = LOG_NAME_RE.match(f"{run_name}.log")
        if not m:
            continue
        out = parse_log(log_path)
        if out is None:
            continue
        mse, mae, n_params = out
        key = (m["dataset"], m["model"], int(m["h"]))
        results[key].append((mse, mae))
        if n_params is not None:
            params[m["model"]] = min(n_params, params.get(m["model"], n_params))
    return results, params


def _compose_cfg(dataset: str, model: str, horizon: int, seed: int, tag: str):
    cfg = OmegaConf.create({
        "seed": seed,
        "device": "cuda",
        "project": "vglg-tsf",
        "run_name": None,
        "tag": tag,
    })
    cfg.data = OmegaConf.load(ROOT / "configs" / "data" / f"{dataset}.yaml")
    cfg.model = OmegaConf.load(ROOT / "configs" / "model" / f"{model}.yaml")
    cfg.train = OmegaConf.load(ROOT / "configs" / "train" / "default.yaml")
    cfg.train.pred_len = horizon
    cfg.train.num_workers = 0
    return cfg


def collect_checkpoints(ckpt_dir: Path, tag: str = "main") -> tuple[dict, dict]:
    """Evaluate checkpoint files when stdout logs were not persisted."""
    results: dict[tuple[str, str, int], list[tuple[float, float]]] = defaultdict(list)
    params: dict[str, int] = {}
    ckpts = sorted(ckpt_dir.glob("*.pt"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for i, ckpt_path in enumerate(ckpts, 1):
        m = CKPT_NAME_RE.match(ckpt_path.name)
        if not m:
            continue
        dataset = m["dataset"]
        model_name = m["model"]
        horizon = int(m["h"])
        seed = int(m["seed"])
        cfg = _compose_cfg(dataset, model_name, horizon, seed, tag)
        loaders = build_dataloaders(cfg.data, cfg.train)
        model = build_model(cfg.model, cfg.data, cfg.train).to(device)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        params[model_name] = min(n_params, params.get(model_name, n_params))
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        metrics = evaluate(model, loaders["test"], device, horizon)
        results[(dataset, model_name, horizon)].append((metrics["mse"], metrics["mae"]))
        print(
            f"[{i}/{len(ckpts)}] {ckpt_path.name} "
            f"mse={metrics['mse']:.6f} mae={metrics['mae']:.6f}",
            flush=True,
        )
    return results, params


def collect_ablations(ablation_root: Path) -> dict:
    """Returns dict[(variant, dataset, horizon)] -> list[(mse, mae)] across seeds.

    Logs are expected at: <ablation_root>/<variant>/<dataset>_h<horizon>_s<seed>.log
    """
    out: dict[tuple[str, str, int], list[tuple[float, float]]] = defaultdict(list)
    if not ablation_root.exists():
        return out
    abl_log_re = re.compile(r"^(?P<dataset>[a-z0-9]+)_h(?P<h>\d+)_s(?P<seed>\d+)\.log$")
    for variant_dir in sorted(ablation_root.iterdir()):
        if not variant_dir.is_dir():
            continue
        variant = variant_dir.name
        for log_path in sorted(variant_dir.glob("*.log")):
            m = abl_log_re.match(log_path.name)
            if not m:
                continue
            parsed = parse_log(log_path)
            if parsed is None:
                continue
            mse, mae, _ = parsed
            out[(variant, m["dataset"], int(m["h"]))].append((mse, mae))
    return out


def fmt_cell(values: list[float]) -> str:
    if not values:
        return "  `—`  "
    mean = sum(values) / len(values)
    return f"{mean:.3f}"


def fmt_delta(value: float | None, baseline: float | None) -> str:
    if value is None or baseline is None:
        return "  `—`  "
    return f"{value - baseline:+.3f}"


def fmt_params(n: int | None) -> str:
    if n is None:
        return "  `—`  "
    if n >= 1_000_000:
        return f"{n / 1e6:.1f}M"
    if n >= 1000:
        return f"{n / 1000:.0f}k"
    return f"{n}"


def build_main_table(results, params) -> list[str]:
    """Table 1: rows = 12 models, cols = 8 datasets, each cell = mean MSE
    across all 4 horizons and 3 seeds."""
    header = (
        "| Model              | Family             |  #Params | "
        + " | ".join(DATASET_LABELS[d] for d in DATASETS)
        + " | **Avg** |"
    )
    sep = "|--------------------|--------------------|---------:|" + "------:|" * (len(DATASETS) + 1)
    rows = [header, sep]
    for mkey, mname, fam in MODELS:
        cells = []
        all_vals = []
        for d in DATASETS:
            vals = []
            for h in HORIZONS:
                vals.extend(v[0] for v in results.get((d, mkey, h), []))
            cells.append(fmt_cell(vals))
            all_vals.extend(vals)
        avg = fmt_cell(all_vals)
        rows.append(
            f"| {mname:<18s} | {fam:<18s} | {fmt_params(params.get(mkey)):>8s} | "
            + " | ".join(cells) + f" | {avg} |"
        )
    return rows


def build_ablation_table(abl: dict) -> list[str]:
    """Table 2: rows = 10 variants, cols = 3 datasets + Overall + Δ vs full.

    A dataset cell shows the mean MSE only when ALL 9 expected runs are
    present (3 horizons × 3 seeds); otherwise shows `(N/9)` so partial
    averages don't masquerade as final numbers.
    """
    n_per_dataset = len(ABLATION_HORIZONS) * 3
    n_per_variant = len(ABLATION_DATASETS) * n_per_dataset
    header = (
        "| Variant                          | "
        + " | ".join(DATASET_LABELS[d] for d in ABLATION_DATASETS)
        + " | Overall | Δ vs full |"
    )
    sep = "|----------------------------------|" + "------:|" * (len(ABLATION_DATASETS) + 2)
    rows = [header, sep]

    overall: dict[str, float | None] = {}
    n_done: dict[str, int] = {}
    for vkey, _ in ABLATION_VARIANTS:
        vals_full = []
        n = 0
        for d in ABLATION_DATASETS:
            ds_vals = []
            for h in ABLATION_HORIZONS:
                ds_vals.extend(v[0] for v in abl.get((vkey, d, h), []))
            n += len(ds_vals)
            if len(ds_vals) == n_per_dataset:
                vals_full.extend(ds_vals)
        overall[vkey] = (sum(vals_full) / len(vals_full)) if vals_full else None
        n_done[vkey] = n
    baseline = overall.get("full")

    for vkey, vlabel in ABLATION_VARIANTS:
        cells = []
        for d in ABLATION_DATASETS:
            vals = []
            for h in ABLATION_HORIZONS:
                vals.extend(v[0] for v in abl.get((vkey, d, h), []))
            if len(vals) == n_per_dataset:
                cells.append(f"{sum(vals) / len(vals):.3f}")
            elif len(vals) == 0:
                cells.append("  `—`  ")
            else:
                cells.append(f"({len(vals)}/{n_per_dataset})")
        ov = overall[vkey]
        if ov is None:
            ov_str = f"({n_done[vkey]}/{n_per_variant})"
        else:
            ov_str = f"{ov:.3f}"
        if vkey == "full":
            delta_str = "—"
        elif ov is None or baseline is None:
            delta_str = "  `—`  "
        else:
            delta_str = fmt_delta(ov, baseline)
        rows.append(
            f"| {vlabel:<32s} | " + " | ".join(cells) + f" | {ov_str} | {delta_str} |"
        )
    return rows


def build_detail_table(dataset: str, results) -> list[str]:
    """Per-dataset table: rows = 12 models, cols = 4 horizons x {MSE, MAE}."""
    cols = []
    for h in HORIZONS:
        cols.append(f"h={h} MSE")
        cols.append(f"h={h} MAE")
    header = "| Model              | " + " | ".join(cols) + " |"
    sep = "|--------------------|" + "------:|" * len(cols)
    rows = [header, sep]
    for mkey, mname, _ in MODELS:
        cells = []
        for h in HORIZONS:
            vals = results.get((dataset, mkey, h), [])
            mse_vals = [v[0] for v in vals]
            mae_vals = [v[1] for v in vals]
            cells.append(fmt_cell(mse_vals))
            cells.append(fmt_cell(mae_vals))
        rows.append(f"| {mname:<18s} | " + " | ".join(cells) + " |")
    return rows


def build_doc(results, params, abl: dict | None = None) -> str:
    if abl is None:
        abl = {}
    n_runs = sum(len(v) for v in results.values())
    n_datasets_with_data = len({d for d, _, _ in results.keys()})
    out = []
    out.append("# Final Results Tables")
    out.append("")
    out.append(
        "Numbers below are **MSE / MAE on the test set**, mean across 3 seeds "
        "(2021/2022/2023). Lower is better. Cells render as `—` when no run has "
        "completed yet. Regenerate with `python scripts/update_results_table.py`."
    )
    out.append("")
    out.append(f"Currently aggregated: **{n_runs} runs** across {n_datasets_with_data} dataset(s).")
    out.append("")
    out.append("---")
    out.append("")
    out.append("## Table 1 — Main results, average MSE across horizons {96, 192, 336, 720}")
    out.append("")
    out.extend(build_main_table(results, params))
    out.append("")
    out.append("> Each cell is the mean MSE across all available (horizon, seed) runs for "
               "that (model, dataset). The four MetaTSF rows share an identical backbone — "
               "only the TokenMixer module differs.")
    out.append("")
    out.append("---")
    out.append("")
    for letter, dataset in zip("abcdefgh", DATASETS):
        out.append(f"## Table 1{letter} — {DATASET_LABELS[dataset]} (per-horizon detail)")
        out.append("")
        out.extend(build_detail_table(dataset, results))
        out.append("")
    out.append("---")
    out.append("")
    n_abl_runs = sum(len(v) for v in abl.values())
    out.append("## Table 2 — VGLG TokenMixer ablation")
    out.append("")
    out.append(
        f"Each variant trained on **3 datasets × 3 horizons × 3 seeds = 27 runs** "
        f"(currently {n_abl_runs} done). A dataset cell shows the mean MSE only "
        f"when all 9 expected runs are present; otherwise `(N/9)`. Δ column = "
        f"overall − full (positive = worse than full)."
    )
    out.append("")
    out.extend(build_ablation_table(abl))
    out.append("")
    out.append(
        "**Datasets**: ETTh1, Weather, Electricity. **Horizons**: 96, 336, 720. "
        "Logs expected at `logs/ablation/<variant>/<dataset>_h<horizon>_s<seed>.log`."
    )
    out.append("")
    out.append("---")
    out.append("")
    out.append("## Table 3 — Distillation (TBD)")
    out.append("")
    out.append(
        "Populated once the Chronos teacher is cached and the 5 distillation "
        "configurations are trained."
    )
    out.append("")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="logs/main", help="Directory with main run logs")
    ap.add_argument(
        "--distill-logs",
        default="logs/distill",
        help="Directory with KD distillation run logs",
    )
    ap.add_argument(
        "--checkpoints",
        default=None,
        help="Evaluate checkpoint directory if logs do not contain test metrics",
    )
    ap.add_argument(
        "--ablation-logs",
        default="logs/ablation",
        help="Directory with ablation run logs (one subdir per variant)",
    )
    ap.add_argument("--out", default="docs/results_table.md")
    args = ap.parse_args()

    log_dir = ROOT / args.logs
    distill_dir = ROOT / args.distill_logs
    abl_dir = ROOT / args.ablation_logs
    out_path = ROOT / args.out
    if not log_dir.exists():
        raise SystemExit(f"Log dir not found: {log_dir}")

    results, params = collect([log_dir, distill_dir])
    n_runs = sum(len(v) for v in results.values())
    if n_runs == 0 and args.checkpoints:
        ckpt_dir = ROOT / args.checkpoints
        if not ckpt_dir.exists():
            raise SystemExit(f"Checkpoint dir not found: {ckpt_dir}")
        print(f"No completed runs found in logs; evaluating checkpoints from {ckpt_dir}")
        results, params = collect_checkpoints(ckpt_dir)
        n_runs = sum(len(v) for v in results.values())
    abl = collect_ablations(abl_dir)
    n_abl = sum(len(v) for v in abl.values())
    print(f"Parsed {n_runs} main runs from {log_dir}")
    print(f"Parsed {n_abl} ablation runs from {abl_dir}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(build_doc(results, params, abl))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
