"""Build a multivariate time-series CSV from F1 session weather data.

We pull weather (AirTemp, TrackTemp, Humidity, Pressure, Rainfall,
WindDirection, WindSpeed) from every official session of one or more F1
seasons, concatenate them along the time axis (using each session's actual
timestamps), and write a single CSV that drops in alongside ETT/Weather/etc.

Usage:
    python scripts/build_f1_weather.py --year 2024 --out data/f1weather/f1weather.csv
    python scripts/build_f1_weather.py --year 2023 2024  # both seasons
"""
from __future__ import annotations

import argparse
from pathlib import Path

import fastf1
import pandas as pd

WEATHER_COLS = [
    "AirTemp", "Humidity", "Pressure", "Rainfall",
    "TrackTemp", "WindDirection", "WindSpeed",
]
SESSION_NAMES = ["FP1", "FP2", "FP3", "Q", "Sprint", "SQ", "R"]


def fetch_session_weather(year: int, round_no: int, session_name: str) -> pd.DataFrame | None:
    """Return a DataFrame with `Time` (timestamp) + WEATHER_COLS, or None on failure."""
    try:
        s = fastf1.get_session(year, round_no, session_name)
        s.load(laps=False, telemetry=False, weather=True, messages=False)
    except Exception as e:
        print(f"  [skip] {year} R{round_no} {session_name}: {e}")
        return None

    wd = s.weather_data
    if wd is None or len(wd) == 0:
        return None
    # Time column is timedelta from session start; convert to absolute timestamp.
    # Prefer session_info["StartDate"]; fall back to s.date (event date).
    info = getattr(s, "session_info", None)
    base = None
    if isinstance(info, dict):
        base = info.get("StartDate")
    if base is None:
        base = s.date
    base = pd.Timestamp(base)
    t = base + wd["Time"]
    df = pd.DataFrame({"date": t})
    for c in WEATHER_COLS:
        if c in wd.columns:
            df[c] = wd[c].values
        else:
            df[c] = pd.NA
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", nargs="+", type=int, default=[2024])
    ap.add_argument("--out", default="data/f1weather/f1weather.csv")
    ap.add_argument("--cache", default=str(Path.home() / ".cache" / "fastf1"))
    args = ap.parse_args()

    Path(args.cache).mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(args.cache)

    all_dfs = []
    for year in args.year:
        sched = fastf1.get_event_schedule(year, include_testing=False)
        print(f"\n=== {year}: {len(sched)} events ===")
        for _, row in sched.iterrows():
            rnd = int(row["RoundNumber"])
            name = row["EventName"]
            print(f"[{year} R{rnd:02d}] {name}")
            for sess in SESSION_NAMES:
                df = fetch_session_weather(year, rnd, sess)
                if df is None or len(df) == 0:
                    continue
                df["source"] = f"{year}_R{rnd:02d}_{sess}"
                all_dfs.append(df)

    if not all_dfs:
        raise SystemExit("No weather data fetched.")

    big = pd.concat(all_dfs, ignore_index=True)
    big = big.sort_values("date").reset_index(drop=True)
    big = big.drop(columns=["source"])
    # interpolate any small gaps within a session (rare); drop rows that are all-NaN
    big = big.dropna(subset=WEATHER_COLS, how="all")
    big[WEATHER_COLS] = big[WEATHER_COLS].astype(float)
    big = big.interpolate(method="linear", limit=5).dropna()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    big.to_csv(out, index=False)
    print(f"\nWrote {out}: {len(big)} rows x {len(big.columns)} cols")
    print(f"Date range: {big['date'].iloc[0]} .. {big['date'].iloc[-1]}")
    print(f"Columns: {list(big.columns)}")


if __name__ == "__main__":
    main()
