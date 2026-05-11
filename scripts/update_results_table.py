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


def collect(log_dir: Path) -> tuple[dict, dict]:
    """Returns (results, params) where:
        results[(dataset, model, horizon)] -> list of (mse, mae) across seeds
        params[model] -> the parameter count seen for that model (any dataset/horizon ok)
    """
    results: dict[tuple[str, str, int], list[tuple[float, float]]] = defaultdict(list)
    params: dict[str, int] = {}
    log_paths = list(log_dir.glob("*.log")) + list(log_dir.glob("*/trainer.log"))
    for log_path in sorted(log_paths):
        run_name = log_path.stem
        if log_path.name == "trainer.log":
            run_name = log_path.parent.name
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


def fmt_cell(values: list[float]) -> str:
    if not values:
        return "  `—`  "
    mean = sum(values) / len(values)
    return f"{mean:.3f}"


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


def build_doc(results, params) -> str:
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
    out.append("## Table 2 — Ablations (TBD)")
    out.append("")
    out.append(
        "Populated once the ablation sweeps finish. Variants: full / fixed gate / "
        "local-only / global-only / kernel ∈ {15, 31, 51} / rank ∈ {4, 8, 16, 32} / "
        "no RevIN."
    )
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
    ap.add_argument("--logs", default="logs/main", help="Directory with run log files")
    ap.add_argument(
        "--checkpoints",
        default=None,
        help="Evaluate checkpoint directory if logs do not contain test metrics",
    )
    ap.add_argument("--out", default="docs/results_table.md")
    args = ap.parse_args()

    log_dir = ROOT / args.logs
    out_path = ROOT / args.out
    if not log_dir.exists():
        raise SystemExit(f"Log dir not found: {log_dir}")

    results, params = collect(log_dir)
    n_runs = sum(len(v) for v in results.values())
    if n_runs == 0 and args.checkpoints:
        ckpt_dir = ROOT / args.checkpoints
        if not ckpt_dir.exists():
            raise SystemExit(f"Checkpoint dir not found: {ckpt_dir}")
        print(f"No completed runs found in logs; evaluating checkpoints from {ckpt_dir}")
        results, params = collect_checkpoints(ckpt_dir)
        n_runs = sum(len(v) for v in results.values())
    print(f"Parsed {n_runs} completed runs from {log_dir}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(build_doc(results, params))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
