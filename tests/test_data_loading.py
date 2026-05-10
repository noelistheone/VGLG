"""Sanity check: every dataset loads with expected shape and no NaNs."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

DATA_ROOT = Path(__file__).resolve().parents[1] / "data"

EXPECTED = {
    "ETT-small/ETTh1.csv": {"rows": 17420, "cols": 8},
    "ETT-small/ETTh2.csv": {"rows": 17420, "cols": 8},
    "ETT-small/ETTm1.csv": {"rows": 69680, "cols": 8},
    "ETT-small/ETTm2.csv": {"rows": 69680, "cols": 8},
    "weather/weather.csv": {"rows": 52696, "cols": 22},
    "electricity/electricity.csv": {"rows": 26304, "cols": 322},
    "traffic/traffic.csv": {"rows": 17544, "cols": 863},
    # FastF1 weather: 2023+2024 sessions, built by scripts/build_f1_weather.py.
    # Row count varies depending on how many sessions were fetched before any
    # rate-limit cap; we only assert column structure and no-NaN here.
    "f1weather/f1weather.csv": {"rows": None, "cols": 8},
}


@pytest.mark.parametrize("name,expected", list(EXPECTED.items()))
def test_dataset_shape(name: str, expected: dict[str, int | None]) -> None:
    path = DATA_ROOT / name
    assert path.exists(), f"Missing dataset file: {path}"
    df = pd.read_csv(path)
    if expected["rows"] is not None:
        assert df.shape[0] == expected["rows"], (
            f"{name}: got {df.shape[0]} rows, expected {expected['rows']}"
        )
    else:
        assert df.shape[0] > 1000, f"{name}: only {df.shape[0]} rows, expected >1000"
    assert df.shape[1] == expected["cols"], (
        f"{name}: got {df.shape[1]} cols, expected {expected['cols']}"
    )
    assert df.isna().sum().sum() == 0, f"{name}: contains NaN"
