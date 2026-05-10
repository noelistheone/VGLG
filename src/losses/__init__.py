from .distill import (
    diff_kd_loss,
    freq_kd_loss,
    total_distill_loss,
    trend_kd_loss,
)

__all__ = [
    "trend_kd_loss",
    "freq_kd_loss",
    "diff_kd_loss",
    "total_distill_loss",
]
