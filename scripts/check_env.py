"""End-of-setup sanity check. Run with `python scripts/check_env.py`."""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import torch

print(f"Python: {sys.version.split()[0]}")
print(f"PyTorch: {torch.__version__}")
print(f"NumPy: {np.__version__}")
print(f"Pandas: {pd.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}, {p.total_memory / 1e9:.1f} GB")

if torch.cuda.is_available():
    x = torch.randn(1024, 1024, device="cuda")
    y = x @ x.T
    print(f"GPU matmul OK: output shape={tuple(y.shape)}")

# Optional: Chronos check (only if installed; skip silently otherwise)
try:
    from chronos import ChronosBoltPipeline  # type: ignore  # noqa: F401
    print("Chronos package import OK (model load deferred to first use).")
except ImportError:
    print("Chronos not installed yet (OK for now; needed for week 7-8).")
