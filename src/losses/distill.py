"""Distillation losses for Chronos-Bolt teacher transfer.

Three complementary signals on top of the standard MSE objective:
  - trend_kd_loss : MSE in the low-frequency Fourier coefficients (k smallest)
  - freq_kd_loss  : MSE on the magnitude spectrum
  - diff_kd_loss  : MSE on the first time difference (captures local slope)
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def trend_kd_loss(student_pred: torch.Tensor, teacher_pred: torch.Tensor, k: int = 8) -> torch.Tensor:
    s = torch.fft.rfft(student_pred, dim=1)
    t = torch.fft.rfft(teacher_pred, dim=1)
    k = min(k, s.size(1))
    s_low = s[:, :k]
    t_low = t[:, :k]
    return F.mse_loss(s_low.real, t_low.real) + F.mse_loss(s_low.imag, t_low.imag)


def freq_kd_loss(student_pred: torch.Tensor, teacher_pred: torch.Tensor) -> torch.Tensor:
    s_mag = torch.fft.rfft(student_pred, dim=1).abs()
    t_mag = torch.fft.rfft(teacher_pred, dim=1).abs()
    return F.mse_loss(s_mag, t_mag)


def diff_kd_loss(student_pred: torch.Tensor, teacher_pred: torch.Tensor) -> torch.Tensor:
    s_diff = student_pred[:, 1:] - student_pred[:, :-1]
    t_diff = teacher_pred[:, 1:] - teacher_pred[:, :-1]
    return F.mse_loss(s_diff, t_diff)


def total_distill_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    teacher_pred: torch.Tensor,
    lambdas: tuple[float, float, float, float] = (1.0, 0.5, 0.5, 0.3),
) -> tuple[torch.Tensor, dict[str, float]]:
    l_mse = F.mse_loss(pred, target)
    l_trend = trend_kd_loss(pred, teacher_pred)
    l_freq = freq_kd_loss(pred, teacher_pred)
    l_diff = diff_kd_loss(pred, teacher_pred)
    total = lambdas[0] * l_mse + lambdas[1] * l_trend + lambdas[2] * l_freq + lambdas[3] * l_diff
    parts = {
        "mse": l_mse.item(),
        "trend": l_trend.item(),
        "freq": l_freq.item(),
        "diff": l_diff.item(),
    }
    return total, parts
