"""ChronosTeacher: wraps amazon/chronos-bolt-{small,base,large} for forecasting.

Chronos-Bolt is *univariate*. Instead of an N-iteration Python loop over
variates (slow for Electricity/Traffic with 321/862 channels), we flatten
the variate axis into the batch so the model sees one big (B*N, L) tensor
and process it in GPU-sized chunks.

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
                mode: str = "mean", chunk_size: int = 256) -> torch.Tensor:
        """`context` shape (B, L, N) -> point forecast (B, pred_len, N).

        Flattens to (B*N, L) and runs the Chronos forward in chunks of size
        `chunk_size`. For Traffic with 862 variates this is roughly 40x faster
        than the naive per-variate loop because Python overhead dominates when
        each call processes only a small batch.

        `mode`: "mean" (MSE-optimal, default) or "median" (q50, MAE-optimal).
        """
        assert mode in ("mean", "median"), f"Unknown mode: {mode}"
        B, L, N = context.shape
        # (B, L, N) -> (B, N, L) -> (B*N, L)
        flat = context.permute(0, 2, 1).reshape(B * N, L).float()

        outs = []
        for start in range(0, B * N, chunk_size):
            chunk = flat[start:start + chunk_size]
            quantiles, mean = self.pipe.predict_quantiles(
                chunk,
                prediction_length=pred_len,
                quantile_levels=[0.5],
            )
            outs.append(mean if mode == "mean" else quantiles[:, :, 0])
        flat_out = torch.cat(outs, dim=0)                             # (B*N, pred_len)
        out = flat_out.reshape(B, N, pred_len).permute(0, 2, 1)       # (B, pred_len, N)
        return out.to(context.device).to(context.dtype)
