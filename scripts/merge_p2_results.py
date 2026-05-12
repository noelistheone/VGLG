"""Fill P2's slice of `docs/results_table.md` from `results/p2_summary.csv`
without disturbing cells already filled by other teammates.

`scripts/update_results_table.py` regenerates the whole markdown from scratch,
which would wipe P1/P5 cells unless every collaborator's logs are on the same
machine. This merger reads each table, replaces only the cells where our P2
summary has data (datasets: Electricity, Traffic), and recomputes the row-Avg
in Table 1.

Usage:
    python scripts/merge_p2_results.py            # in-place edit of docs/results_table.md
    python scripts/merge_p2_results.py --dry-run  # print the diff to stdout
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Order/naming matches the existing markdown
TABLE1_DATASETS = ["etth1", "etth2", "ettm1", "ettm2", "weather", "electricity", "traffic", "f1weather"]
MODELS = [
    ("dlinear",      "DLinear"),
    ("lstm",         "LSTM"),
    ("gru",          "GRU"),
    ("segrnn",       "SegRNN"),
    ("timemixer",    "TimeMixer"),
    ("moderntcn",    "ModernTCN"),
    ("itransformer", "iTransformer"),
    ("patchtst",     "PatchTST"),
    ("metatsf_mlp",  "**MetaTSF-MLP**"),
    ("metatsf_conv", "**MetaTSF-Conv**"),
    ("metatsf_attn", "**MetaTSF-Attn**"),
    ("metatsf_vglg", "**MetaTSF-VGLG**"),
]
P2_DATASETS = {"electricity", "traffic"}
HORIZONS = [96, 192, 336, 720]


def load_p2_summary() -> dict:
    """Returns {(model, dataset, horizon): (mse, mae)}."""
    out: dict[tuple[str, str, int], tuple[float, float]] = {}
    with (ROOT / "results" / "p2_summary.csv").open() as fh:
        for row in csv.DictReader(fh):
            if row["kind"] != "main":
                continue
            key = (row["model"], row["dataset"], int(row["horizon"]))
            out[key] = (float(row["mse_mean"]), float(row["mae_mean"]))
    return out


# A markdown cell may be either a number like " 0.490 " / "0.490" or a placeholder
# like "  `—`  " or "  `—` ". This pattern matches both forms.
_CELL_NUM_RE = re.compile(r"^\s*(?:`?-`?|`?—`?|[\d.]+)\s*$")


def parse_cell(s: str) -> float | None:
    s = s.strip()
    if s in ("", "—", "-") or s.startswith("`"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def fmt_num(x: float) -> str:
    return f"{x:.3f}"


_DASH = "  `—`  "


def fmt_cell_or_dash(v: float | None) -> str:
    return fmt_num(v) if v is not None else _DASH


def merge_table1_main(lines: list[str], p2: dict) -> list[str]:
    """Replace the Electricity and Traffic columns in Table 1's row for each
    model, then recompute the trailing Avg cell from all non-`—` cells.

    Table 1 columns:
      | Model | Family | #Params | ETTh1 | ETTh2 | ETTm1 | ETTm2 | Weather | Electricity | Traffic | f1weather | **Avg** |
      idx       0         1          2         3       4       5       6        7            8           9         10              11
    """
    out = []
    in_table1 = False
    for line in lines:
        if line.startswith("## Table 1 —"):
            in_table1 = True
            out.append(line)
            continue
        if in_table1 and line.startswith("## "):  # next section header ends Table 1
            in_table1 = False
            out.append(line)
            continue
        if in_table1 and line.startswith("|") and not line.startswith("|---") and "Model" not in line:
            cells = [c for c in line.split("|")[1:-1]]
            if len(cells) != 12:
                out.append(line)
                continue
            model_name = cells[0].strip()
            mkey = next((k for k, name in MODELS if name == model_name), None)
            if mkey is None:
                out.append(line)
                continue

            # Compute P2 averages for Electricity and Traffic columns
            for ds, col_idx in (("electricity", 8), ("traffic", 9)):
                vals = [p2.get((mkey, ds, h)) for h in HORIZONS]
                mses = [v[0] for v in vals if v is not None]
                if mses:
                    mean = sum(mses) / len(mses)
                    cells[col_idx] = f" {fmt_num(mean)} "

            # Recompute Avg from all non-dash data columns (idx 3..10)
            data_vals = [parse_cell(c) for c in cells[3:11]]
            present = [v for v in data_vals if v is not None]
            if present:
                avg = sum(present) / len(present)
                cells[11] = f" {fmt_num(avg)} "
            out.append("|" + "|".join(cells) + "|")
            continue
        out.append(line)
    return out


def merge_perdataset_table(lines: list[str], p2: dict, header_prefix: str, dataset: str) -> list[str]:
    """Replace per-horizon detail rows for one dataset.

    Per-dataset columns (8 numeric):
      h=96 MSE | h=96 MAE | h=192 MSE | h=192 MAE | h=336 MSE | h=336 MAE | h=720 MSE | h=720 MAE
    """
    out = []
    in_section = False
    for line in lines:
        if line.startswith(header_prefix):
            in_section = True
            out.append(line)
            continue
        if in_section and line.startswith("## "):
            in_section = False
            out.append(line)
            continue
        if in_section and line.startswith("|") and not line.startswith("|---") and "Model" not in line:
            cells = [c for c in line.split("|")[1:-1]]
            if len(cells) != 9:
                out.append(line)
                continue
            model_name = cells[0].strip()
            mkey = next((k for k, name in MODELS if name == model_name), None)
            if mkey is None:
                out.append(line)
                continue
            for j, h in enumerate(HORIZONS):
                v = p2.get((mkey, dataset, h))
                if v is None:
                    continue
                mse, mae = v
                cells[1 + 2 * j]     = f" {fmt_num(mse)} "
                cells[1 + 2 * j + 1] = f" {fmt_num(mae)} "
            out.append("|" + "|".join(cells) + "|")
            continue
        out.append(line)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--md", default="docs/results_table.md")
    args = ap.parse_args()

    p2 = load_p2_summary()
    print(f"Loaded {len(p2)} P2 data points from results/p2_summary.csv "
          f"({sum(1 for k in p2 if k[1] == 'electricity')} electricity, "
          f"{sum(1 for k in p2 if k[1] == 'traffic')} traffic).")

    md_path = ROOT / args.md
    original = md_path.read_text()
    lines = original.split("\n")

    lines = merge_table1_main(lines, p2)
    lines = merge_perdataset_table(lines, p2, "## Table 1f", "electricity")
    lines = merge_perdataset_table(lines, p2, "## Table 1g", "traffic")

    merged = "\n".join(lines)

    if args.dry_run:
        # Show diff
        import difflib
        diff = difflib.unified_diff(
            original.splitlines(keepends=True),
            merged.splitlines(keepends=True),
            fromfile="docs/results_table.md (before)",
            tofile="docs/results_table.md (after)",
            n=2,
        )
        print("".join(diff))
        return

    md_path.write_text(merged)
    print(f"Wrote {md_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
