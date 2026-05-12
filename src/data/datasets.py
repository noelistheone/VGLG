"""PyTorch Datasets for ETT / custom (Weather, Electricity, Traffic) CSVs.

Conventions follow the Time-Series-Library (THUML) splits exactly so that our
numbers are directly comparable to published baselines.

ETTh: 12 months train / 4 months val / 4 months test
ETTm: same months, but at 15-min resolution
custom: 70% train / 10% val / 20% test on the time axis

Each item returned by __getitem__ is the standard 4-tuple:
    seq_x        (seq_len, n_vars)        encoder input
    seq_y        (label_len + pred_len, n_vars) decoder target window
    seq_x_mark   (seq_len, n_time_features)
    seq_y_mark   (label_len + pred_len, n_time_features)
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


def _time_features(dates: pd.DatetimeIndex, freq: str = "h") -> np.ndarray:
    """Map raw timestamps to a few normalized cyclic features in [-0.5, 0.5].

    The exact features included depend on `freq` to mirror TSlib behaviour.
    Returned shape: (T, n_features) with n_features in {3, 4, 5}.
    """
    features = []
    if freq in ("t", "min"):
        features.append((dates.minute / 59.0) - 0.5)
    features.append((dates.hour / 23.0) - 0.5)
    features.append((dates.dayofweek / 6.0) - 0.5)
    features.append((dates.day - 1) / 30.0 - 0.5)
    features.append((dates.dayofyear - 1) / 365.0 - 0.5)
    return np.stack(features, axis=-1).astype(np.float32)


class _ForecastDatasetBase(Dataset):
    flag_to_idx = {"train": 0, "val": 1, "test": 2}

    def __init__(
        self,
        root_path: str,
        data_path: str,
        flag: Literal["train", "val", "test"],
        seq_len: int,
        label_len: int,
        pred_len: int,
        features: Literal["M", "S", "MS"] = "M",
        target: str = "OT",
        scale: bool = True,
        freq: str = "h",
        gpu_resident: bool = False,
        device: str = "cuda",
    ):
        super().__init__()
        assert flag in self.flag_to_idx
        self.root_path = Path(root_path)
        self.data_path = data_path
        self.flag = flag
        self.set_type = self.flag_to_idx[flag]
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.features = features
        self.target = target
        self.scale = scale
        self.freq = freq
        self.gpu_resident = gpu_resident
        self.device = device
        self.scaler = StandardScaler()
        self._read_data()

    # subclasses implement border computation
    def _borders(self, n_rows: int) -> tuple[list[int], list[int]]:
        raise NotImplementedError

    def _read_data(self) -> None:
        df_raw = pd.read_csv(self.root_path / self.data_path)
        # `date` first column, then variates
        cols = list(df_raw.columns)
        assert cols[0].lower() in ("date", "timestamp", "time"), (
            f"Expected first column to be a date column, got '{cols[0]}'"
        )
        date_col = cols[0]

        if self.features in ("M", "MS"):
            df_data = df_raw[cols[1:]]
        else:  # S
            df_data = df_raw[[self.target]]

        n_rows = len(df_raw)
        border1s, border2s = self._borders(n_rows)
        b1, b2 = border1s[self.set_type], border2s[self.set_type]

        if self.scale:
            train_data = df_data.iloc[border1s[0]:border2s[0]].values
            self.scaler.fit(train_data)
            data = self.scaler.transform(df_data.values).astype(np.float32)
        else:
            data = df_data.values.astype(np.float32)

        # time features
        dates = pd.to_datetime(df_raw[date_col].values)
        time_feats = _time_features(dates, freq=self.freq)

        if self.gpu_resident:
            # Bit-identical to the numpy path: torch.from_numpy preserves the
            # underlying float32 representation, then .to(device) is a pure copy.
            # data_x and data_y share the same content — share storage to save VRAM.
            data_t = torch.from_numpy(data[b1:b2]).to(self.device)
            self.data_x = data_t
            self.data_y = data_t
            self.data_stamp = torch.from_numpy(time_feats[b1:b2]).to(self.device)
        else:
            self.data_x = data[b1:b2]
            self.data_y = data[b1:b2]
            self.data_stamp = time_feats[b1:b2]

    def __len__(self) -> int:
        return max(0, len(self.data_x) - self.seq_len - self.pred_len + 1)

    def __getitem__(self, idx: int):
        s = idx
        s_end = s + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        if self.gpu_resident:
            # Already torch tensors on GPU — slices are views (zero copy).
            return seq_x, seq_y, seq_x_mark, seq_y_mark
        return (
            torch.from_numpy(seq_x),
            torch.from_numpy(seq_y),
            torch.from_numpy(seq_x_mark),
            torch.from_numpy(seq_y_mark),
        )

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return self.scaler.inverse_transform(x)


class DatasetETTHour(_ForecastDatasetBase):
    """ETTh1 / ETTh2: hourly resolution, fixed month-based splits."""

    def _borders(self, n_rows: int) -> tuple[list[int], list[int]]:
        # 12 months train / 4 months val / 4 months test
        m = 30 * 24
        border1s = [0, 12 * m - self.seq_len, 12 * m + 4 * m - self.seq_len]
        border2s = [12 * m, 12 * m + 4 * m, 12 * m + 8 * m]
        return border1s, border2s


class DatasetETTMinute(_ForecastDatasetBase):
    """ETTm1 / ETTm2: 15-min resolution, fixed month-based splits."""

    def _borders(self, n_rows: int) -> tuple[list[int], list[int]]:
        # 12 / 4 / 4 months in 15-minute steps
        m = 30 * 24 * 4
        border1s = [0, 12 * m - self.seq_len, 12 * m + 4 * m - self.seq_len]
        border2s = [12 * m, 12 * m + 4 * m, 12 * m + 8 * m]
        return border1s, border2s


class DatasetCustom(_ForecastDatasetBase):
    """Weather / Electricity / Traffic: 70/10/20 split on the time axis."""

    def _borders(self, n_rows: int) -> tuple[list[int], list[int]]:
        n_train = int(n_rows * 0.7)
        n_test = int(n_rows * 0.2)
        n_val = n_rows - n_train - n_test
        border1s = [0, n_train - self.seq_len, n_rows - n_test - self.seq_len]
        border2s = [n_train, n_train + n_val, n_rows]
        return border1s, border2s


def build_dataset(
    data_kind: str,
    flag: str,
    root_path: str,
    data_path: str,
    seq_len: int,
    label_len: int,
    pred_len: int,
    features: str = "M",
    target: str = "OT",
    freq: str = "h",
    gpu_resident: bool = False,
    device: str = "cuda",
) -> _ForecastDatasetBase:
    cls = {
        "ETTh": DatasetETTHour,
        "ETTm": DatasetETTMinute,
        "custom": DatasetCustom,
    }[data_kind]
    return cls(
        root_path=root_path,
        data_path=data_path,
        flag=flag,
        seq_len=seq_len,
        label_len=label_len,
        pred_len=pred_len,
        features=features,
        target=target,
        freq=freq,
        gpu_resident=gpu_resident,
        device=device,
    )
