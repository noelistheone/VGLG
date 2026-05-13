"""ChronosTeacher: wraps amazon/chronos-bolt-{small,base,large} for forecasting.

Chronos-Bolt is *univariate*: each variate is forecast independently, then we
stack them. We loop one variate at a time (over the channel axis) so the
multi-variate VGLG/baseline output shape `(B, pred_len, N)` is preserved.

Usage:
    teacher = ChronosTeacher("amazon/chronos-bolt-base")
    pred = teacher.predict(context_BLN, pred_len)   # -> (B, pred_len, N)
"""
from __future__ import annotations

import torch

from chronos import ChronosBoltPipeline


class ChronosTeacher:
    def __init__(
        self,
        model_name: str = "amazon/chronos-bolt-base",
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.model_name = model_name
        self.device = device
        self.pipe = ChronosBoltPipeline.from_pretrained(
            model_name, device_map=device, torch_dtype=dtype,
        )

    @torch.no_grad()
    def predict(self, context: torch.Tensor, pred_len: int,
                mode: str = "mean") -> torch.Tensor:
        """`context` shape (B, L, N) -> point forecast (B, pred_len, N).

        `mode` selects the point estimate:
            "mean"   : MSE-optimal, default
            "median" : MAE-optimal (q50)
        """
        assert mode in ("mean", "median"), f"Unknown mode: {mode}"
        B, L, N = context.shape
        outputs = []
        for n in range(N):
            ctx_n = context[:, :, n].float()                          # (B, L)
            quantiles, mean = self.pipe.predict_quantiles(
                inputs=ctx_n,
                prediction_length=pred_len,
                quantile_levels=[0.5],
            )
            if mode == "mean":
                outputs.append(mean)                                  # (B, pred_len)
            else:
                outputs.append(quantiles[:, :, 0])                    # (B, pred_len)
        out = torch.stack(outputs, dim=-1)                            # (B, pred_len, N)
        return out.to(context.device).to(context.dtype)
