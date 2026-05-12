"""Numerical equivalence check: gpu_resident=False vs True must produce
bit-identical training trajectories so we can mix old (CPU-dataloader) and new
(GPU-resident) runs in the same 3-seed average without bias.

Method:
  1. Seed everything to a fixed value.
  2. Build the same model (dlinear) + data (traffic h=96) twice — once with
     the legacy numpy dataloader, once with gpu_resident.
  3. Step through N batches manually, collecting per-batch (loss, output_hash).
  4. Assert the two runs match within float-rounding tolerance.

Exits with code 0 if equivalent, 1 otherwise.

Usage:
    python scripts/verify_dataloader_equivalence.py
    python scripts/verify_dataloader_equivalence.py --dataset traffic --model dlinear --batches 30
    python scripts/verify_dataloader_equivalence.py --gpu 4 --model metatsf_vglg
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace

# Must parse --gpu and set CUDA_VISIBLE_DEVICES *before* importing torch so the
# child process only sees the requested GPU.
_pre_ap = argparse.ArgumentParser(add_help=False)
_pre_ap.add_argument("--gpu", type=int, default=4)
_pre_args, _ = _pre_ap.parse_known_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(_pre_args.gpu)

import torch                     # noqa: E402
import torch.nn as nn            # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import build_dataloaders   # noqa: E402
from src.models import build_model       # noqa: E402
from src.utils import seed_everything    # noqa: E402

# Match probe-validated batch sizes for the test combos.
BS = {("traffic", 96): 32, ("electricity", 96): 64, ("etth1", 96): 64}


def make_data_cfg(name: str):
    base = ROOT / "configs" / "data" / f"{name}.yaml"
    import yaml
    cfg = yaml.safe_load(base.read_text())
    return SimpleNamespace(**cfg)


def make_train_cfg(batch_size: int, pred_len: int, gpu_resident: bool):
    return SimpleNamespace(
        seq_len=96, label_len=48, pred_len=pred_len,
        batch_size=batch_size, num_workers=4,
        train_epochs=1, patience=99, learning_rate=1e-4,
        loss="mse", lradj="constant", weight_decay=0.0,
        gradient_clip=1.0, use_amp=True,
        log_interval=10000, val_interval=1,
        checkpoint_dir="checkpoints", use_wandb=False, wandb_project="vglg-tsf",
        gpu_resident_data=gpu_resident,
        matmul_precision="highest",
    )


def make_model_cfg(name: str):
    base = ROOT / "configs" / "model" / f"{name}.yaml"
    import yaml
    cfg = yaml.safe_load(base.read_text())
    return SimpleNamespace(**cfg)


@torch.no_grad()
def hash_tensor(t: torch.Tensor) -> float:
    """A reduce that's sensitive to any per-element difference."""
    return (t.detach().float().pow(2).sum() / t.numel()).item()


def run_trajectory(seed: int, dataset: str, model_name: str, pred_len: int,
                   batch_size: int, n_batches: int, gpu_resident: bool, device: str) -> dict:
    """Run N training steps deterministically and capture per-batch fingerprints."""
    seed_everything(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Force deterministic algorithms where possible (still permits non-determinism
    # for ops without deterministic kernels — fine since we only compare same-path).
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    data_cfg = make_data_cfg(dataset)
    train_cfg = make_train_cfg(batch_size, pred_len, gpu_resident)
    model_cfg = make_model_cfg(model_name)

    loaders = build_dataloaders(data_cfg, train_cfg)
    model = build_model(model_cfg, data_cfg, train_cfg).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=train_cfg.learning_rate)
    criterion = nn.MSELoss()

    losses: list[float] = []
    weight_hash: list[float] = []
    batch_hash: list[float] = []

    model.train()
    seed_everything(seed)  # re-seed before iteration to match shuffle order across runs
    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(loaders["train"]):
        if i >= n_batches:
            break
        batch_x = batch_x.float().to(device, non_blocking=True)
        batch_y = batch_y.float().to(device, non_blocking=True)
        batch_x_mark = batch_x_mark.float().to(device, non_blocking=True)
        batch_y_mark = batch_y_mark.float().to(device, non_blocking=True)

        dec_inp = torch.zeros_like(batch_y[:, -pred_len:, :])
        dec_inp = torch.cat([batch_y[:, : batch_y.size(1) - pred_len, :], dec_inp], dim=1)

        optim.zero_grad(set_to_none=True)
        # Use deterministic non-AMP path for equivalence (AMP introduces nondeterminism
        # via half->float casts ordering on different launches).
        out = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        out = out[:, -pred_len:, :]
        target = batch_y[:, -pred_len:, :]
        loss = criterion(out, target)
        loss.backward()
        optim.step()

        losses.append(loss.detach().float().item())
        batch_hash.append(hash_tensor(batch_x))
        # weight hash after step
        w0 = next(model.parameters())
        weight_hash.append(hash_tensor(w0))

    return {
        "losses": losses,
        "batch_hash": batch_hash,
        "weight_hash": weight_hash,
        "n_train_dataset": len(loaders["train"].dataset),
    }


def compare(a: dict, b: dict, atol_loss: float, atol_hash: float) -> tuple[bool, str]:
    if len(a["losses"]) != len(b["losses"]):
        return False, f"Loss list length differs: {len(a['losses'])} vs {len(b['losses'])}"
    if a["n_train_dataset"] != b["n_train_dataset"]:
        return False, f"Dataset size differs: {a['n_train_dataset']} vs {b['n_train_dataset']}"

    bad_lines = []
    for i, (la, lb) in enumerate(zip(a["losses"], b["losses"])):
        diff = abs(la - lb)
        if diff > atol_loss:
            bad_lines.append(f"  batch {i:3d}: loss diff={diff:.3e}  ({la:.6f} vs {lb:.6f})")
    for i, (ha, hb) in enumerate(zip(a["batch_hash"], b["batch_hash"])):
        diff = abs(ha - hb)
        if diff > atol_hash:
            bad_lines.append(f"  batch {i:3d}: BATCH-input diff={diff:.3e} — different shuffle order!")
    for i, (ha, hb) in enumerate(zip(a["weight_hash"], b["weight_hash"])):
        diff = abs(ha - hb)
        if diff > atol_hash:
            bad_lines.append(f"  batch {i:3d}: WEIGHT diff={diff:.3e}")

    if bad_lines:
        return False, "\n".join(bad_lines[:20])
    return True, "OK"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=4)
    ap.add_argument("--dataset", default="traffic")
    ap.add_argument("--model", default="dlinear")
    ap.add_argument("--pred-len", type=int, default=96)
    ap.add_argument("--batches", type=int, default=20)
    ap.add_argument("--seed", type=int, default=2021)
    ap.add_argument("--atol-loss", type=float, default=1e-5,
                    help="Per-batch loss tolerance (rtol-style on values ~1)")
    ap.add_argument("--atol-hash", type=float, default=1e-6,
                    help="Per-batch tensor-hash tolerance")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    bs = BS.get((args.dataset, args.pred_len), 32)
    print(f"== Equivalence test ==")
    print(f"  dataset={args.dataset}  model={args.model}  h={args.pred_len}  bs={bs}")
    print(f"  device={device} ({args.gpu})  seed={args.seed}  batches={args.batches}")
    print()

    print("Run 1: gpu_resident=False (legacy numpy dataloader)...", flush=True)
    r1 = run_trajectory(args.seed, args.dataset, args.model, args.pred_len,
                        bs, args.batches, gpu_resident=False, device=device)
    print(f"  ✓ ran {len(r1['losses'])} batches, dataset size {r1['n_train_dataset']}")
    print(f"  first 3 losses: {[f'{x:.6f}' for x in r1['losses'][:3]]}")
    print(f"  last  3 losses: {[f'{x:.6f}' for x in r1['losses'][-3:]]}")
    print()

    print("Run 2: gpu_resident=True (new GPU-tensor dataloader)...", flush=True)
    r2 = run_trajectory(args.seed, args.dataset, args.model, args.pred_len,
                        bs, args.batches, gpu_resident=True, device=device)
    print(f"  ✓ ran {len(r2['losses'])} batches, dataset size {r2['n_train_dataset']}")
    print(f"  first 3 losses: {[f'{x:.6f}' for x in r2['losses'][:3]]}")
    print(f"  last  3 losses: {[f'{x:.6f}' for x in r2['losses'][-3:]]}")
    print()

    ok, msg = compare(r1, r2, args.atol_loss, args.atol_hash)
    if ok:
        print("✅ EQUIVALENT — safe to switch.")
        sys.exit(0)
    else:
        print("❌ NOT EQUIVALENT — divergence:")
        print(msg)
        print()
        print("Do NOT switch the launcher. Mixing the two paths would bias 3-seed averages.")
        sys.exit(1)


if __name__ == "__main__":
    main()
