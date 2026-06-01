"""Distillation losses for Chronos-Bolt teacher transfer.

The training objective is the task MSE plus a weighted set of KD terms that pull
the student toward the (cached) teacher's forecast:

  - softmse_kd : MSE(student, teacher) in the time domain — the *direct*
                 soft-target term (classic KD). This is the dominant KD signal.
  - trend_kd   : MSE on the low-frequency Fourier coefficients (k smallest),
                 ortho-normalized, to match the teacher's coarse shape.
  - diff_kd    : MSE on the first time difference (local slope).

All Fourier transforms use ``norm="ortho"`` so that, by Parseval's theorem,
spectral-domain errors live on the *same scale* as the time-domain MSE. The
previous implementation used the default (un-normalized) ``rfft``, whose
coefficient magnitudes grow with ``pred_len`` (e.g. trend ~ 160 vs mse ~ 0.3):
the loss weights then had to be crushed to ~1e-4 to keep training stable, which
made the KD signal negligible, and any transient blow-up of the student output
sent the squared-magnitude terms to ~1e7 and diverged training. Ortho
normalization fixes both: the terms are O(MSE), so the weights are O(0.1-1) and
training is numerically stable.

``freq_kd`` (full magnitude spectrum) is retained for API compatibility but is
*not* part of the default bundle: it double-counts the low-frequency bins that
``trend_kd`` already covers and is phase-blind (two predictions identical up to a
time shift have zero freq loss).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def softmse_kd_loss(student_pred: torch.Tensor, teacher_pred: torch.Tensor) -> torch.Tensor:
    """Direct soft-target distillation: MSE between student and teacher forecasts."""
    return F.mse_loss(student_pred, teacher_pred)


def trend_kd_loss(student_pred: torch.Tensor, teacher_pred: torch.Tensor, k: int = 8) -> torch.Tensor:
    s = torch.fft.rfft(student_pred.float(), dim=1, norm="ortho")
    t = torch.fft.rfft(teacher_pred.float(), dim=1, norm="ortho")
    k = min(k, s.size(1))
    s_low = s[:, :k]
    t_low = t[:, :k]
    return F.mse_loss(s_low.real, t_low.real) + F.mse_loss(s_low.imag, t_low.imag)


def freq_kd_loss(student_pred: torch.Tensor, teacher_pred: torch.Tensor) -> torch.Tensor:
    """Full-spectrum magnitude match (ortho-normalized). Deprecated: redundant
    with trend_kd over the low band and phase-blind; kept for API compatibility."""
    s_mag = torch.fft.rfft(student_pred.float(), dim=1, norm="ortho").abs()
    t_mag = torch.fft.rfft(teacher_pred.float(), dim=1, norm="ortho").abs()
    return F.mse_loss(s_mag, t_mag)


def diff_kd_loss(student_pred: torch.Tensor, teacher_pred: torch.Tensor) -> torch.Tensor:
    s_diff = student_pred[:, 1:] - student_pred[:, :-1]
    t_diff = teacher_pred[:, 1:] - teacher_pred[:, :-1]
    return F.mse_loss(s_diff, t_diff)


def kd_loss_bundle(
    student_pred: torch.Tensor,
    teacher_pred: torch.Tensor,
    k: int = 8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the three default KD terms (soft, trend, diff), sharing one rfft.

    Returns ``(l_soft, l_trend, l_diff)``. Inputs are upcast to fp32 so the FFT is
    numerically safe even when the student output arrives in fp16 under AMP.
    """
    student_pred = student_pred.float()
    teacher_pred = teacher_pred.float()

    l_soft = F.mse_loss(student_pred, teacher_pred)

    s_fft = torch.fft.rfft(student_pred, dim=1, norm="ortho")
    t_fft = torch.fft.rfft(teacher_pred, dim=1, norm="ortho")
    kk = min(k, s_fft.size(1))
    s_low, t_low = s_fft[:, :kk], t_fft[:, :kk]
    l_trend = F.mse_loss(s_low.real, t_low.real) + F.mse_loss(s_low.imag, t_low.imag)

    s_diff = student_pred[:, 1:] - student_pred[:, :-1]
    t_diff = teacher_pred[:, 1:] - teacher_pred[:, :-1]
    l_diff = F.mse_loss(s_diff, t_diff)

    return l_soft, l_trend, l_diff


def total_distill_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    teacher_pred: torch.Tensor,
    lambdas: tuple[float, float, float, float] = (1.0, 0.5, 0.1, 0.1),
) -> tuple[torch.Tensor, dict[str, float]]:
    """`lambdas = (task_mse, soft, trend, diff)`."""
    l_mse = F.mse_loss(pred, target)
    l_soft, l_trend, l_diff = kd_loss_bundle(pred, teacher_pred)
    total = lambdas[0] * l_mse + lambdas[1] * l_soft + lambdas[2] * l_trend + lambdas[3] * l_diff
    parts = {
        "mse": l_mse.item(),
        "soft": l_soft.item(),
        "trend": l_trend.item(),
        "diff": l_diff.item(),
    }
    return total, parts
