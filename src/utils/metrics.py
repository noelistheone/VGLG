"""Standard forecasting metrics. Inputs are numpy arrays of any shape."""
from __future__ import annotations

import numpy as np


def MAE(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - true)))


def MSE(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean((pred - true) ** 2))


def RMSE(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.sqrt(MSE(pred, true)))


def MAPE(pred: np.ndarray, true: np.ndarray, eps: float = 1e-8) -> float:
    return float(np.mean(np.abs((pred - true) / (true + eps))))


def metric(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    return {"mse": MSE(pred, true), "mae": MAE(pred, true), "rmse": RMSE(pred, true)}
