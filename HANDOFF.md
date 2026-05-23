# HANDOFF — MetaTSF supplementary experiments, PART 2

**Owner of part 1**: Jay (running on main workstation, GPU 4)
**Owner of part 2**: you (this hand-off)
**Branch**: `main` (no PR — supplementary work merges via raw logs)
**Created**: 2026-05-22

---

## TL;DR

We're rerunning baselines under each model's "fair" recipe (closer to its original paper) and comparing to the main results in `docs/results_table.md`. The full set is 7 baselines × 7 datasets × 4 horizons = 196 runs on 1 GPU. Jay is taking the 3 heaviest models on his machine (PART 1). **You are taking the 4 lighter ones (PART 2): TimeMixer + iTransformer + LSTM + GRU — 112 runs, expected ~20 h on 1 GPU.**

Datasets used: `etth1 etth2 ettm1 ettm2 weather electricity traffic` (f1weather intentionally excluded — no original-paper baseline to compare to).
Horizons: `{96, 192, 336, 720}`. Seed: `2021` only.

---

## What's in scope for you

| Model | Epochs | Patience | LR | LR schedule | Why these settings |
|---|---:|---:|---|---|---|
| **LSTM** | 10 | 3 | **1e-3** | type1 | TSlib default; RNNs are LR-sensitive |
| **GRU** | 10 | 3 | **1e-3** | type1 | Same as LSTM |
| **iTransformer** | 10 | 3 | **5e-4** | type1 | Paper default LR for most datasets |
| **TimeMixer** | 10 | 5 | **1e-2** | type3 (cosine) | Paper default; 100× our main-run LR |

These differ from `configs/train/default.yaml` (which is `lr=1e-4, lradj=type1, patience=3`) — the launcher overrides them per model via Hydra CLI args.

Run count: 4 models × 7 datasets × 4 horizons × 1 seed = **112 runs**.

The other 3 models (PatchTST, ModernTCN, SegRNN — PART 1) are running on the main machine and are **not your responsibility**. PART 1 logs will collide-free with PART 2 because filenames are `<dataset>_<model>_h<H>_s2021.log` and we have disjoint model sets.

---

## Quick start

### 1. Clone / sync the repo

```bash
git clone <repo-url> VGLG && cd VGLG
git checkout main
git pull
```

The launcher and config changes are already on `main` (commits up through `1f74618`). The trainer was patched today (2026-05-22) to handle non-10-epoch cosine schedules correctly — verify your `src/train/trainer.py:30-40` shows `_adjust_lr(..., total_epochs: int = 10)` with `np.pi * epoch / total_epochs` in the type3 arm. If you don't see that, `git pull` again.

### 2. Set up the env

We use a `vglg` conda env. If you don't have it:

```bash
conda env create -f environment.yml -n vglg     # if a file exists
# or replicate from CLAUDE.md and the imports in src/
conda activate vglg
```

`scripts/check_env.py` is a 1-minute sanity check (PyTorch + CUDA + Chronos):

```bash
python scripts/check_env.py
```

### 3. Get the data

```bash
bash scripts/download_data.sh    # ~355 MB into data/
```

You do **not** need f1weather, so you can skip `scripts/build_f1_weather.py`.

### 4. Pick your GPU

```bash
nvidia-smi
```

Pick whichever GPU has > 24 GB free and is not in use. The launcher pins it via `CUDA_VISIBLE_DEVICES`. We use 32-batch by default; if you OOM on Traffic, drop to `train.batch_size=8`.

### 5. Launch

```bash
mkdir -p logs/supp_method_b
nohup env GPU=<your-gpu-index> bash scripts/run_supp_method_b_part2.sh \
    > logs/supp_method_b/nohup_part2.out 2>&1 &
echo "Launched PID=$!"
```

To run a single model only (useful for pre-flight or recovery):

```bash
GPU=0 ONLY=timemixer bash scripts/run_supp_method_b_part2.sh
```

Valid `ONLY` values: `lstm`, `gru`, `itransformer`, `timemixer`.

### 6. Monitor

```bash
# overall progress (count of OK + SKIP lines)
grep -c "^\[.*\] OK \|^\[.*\] SKIP " logs/supp_method_b/master.log

# recent events
tail -20 logs/supp_method_b/master.log

# what's running now
ps -p <PID>; nvidia-smi -i <GPU>
```

The launcher is **idempotent**: if it dies/crashes, restart with the same command — runs that already wrote `Test | mse=...` to their log file are SKIPped.

---

## Pre-flight (recommended, ~3 min)

Before launching the full sweep, smoke-test TimeMixer at the LR-1e-2 setting. We bumped this LR from 1e-4 to 1e-2 (100×) and it might NaN on some datasets.

```bash
GPU=<your-gpu> /data/xinkaiz/conda_envs/vglg/bin/python -m src.train.trainer \
    model=timemixer data=etth1 train.pred_len=96 \
    train.train_epochs=3 train.learning_rate=0.01 train.lradj=type3 \
    seed=2021 tag=smoke_timemixer
```

You should see `train_loss` and `val_mse` decrease smoothly. If you see `NaN` or wild divergence, fall back to `learning_rate: 0.005` and ping Jay. Edit `scripts/run_supp_method_b_part2.sh` line `run_one timemixer ... 0.01 type3` → `0.005`.

---

## Estimated wall-time

Derived from PART 1's measured per-epoch costs scaled down for these 4 (lighter) models:

| Model | Per-run avg | × 28 runs | Notes |
|---|---:|---:|---|
| LSTM | ~5–10 min | ~3–5 h | Small model, 10 ep |
| GRU | ~5–10 min | ~3–5 h | Same as LSTM |
| iTransformer | ~8–15 min | ~5–7 h | d_model=256 |
| TimeMixer | ~5–10 min | ~3–5 h | Risk of NaN — see pre-flight |
| **Total** | | **~15–22 h** | 1 GPU |

---

## Output expected from you

Two paths:

1. **Per-run logs** at `logs/supp_method_b/<dataset>_<model>_h<H>_s2021.log` (112 files). The trainer writes the final test line as `Test | mse=... mae=... rmse=...`.
2. **Master log** at `logs/supp_method_b/master.log` (your part will append; don't worry about merging — line-based).

When you're done (or partially done, if you need to hand back early), send back a tarball:

```bash
tar -czf supp_method_b_part2.tar.gz logs/supp_method_b/ checkpoints/supp_method_b/
# or just the logs if checkpoints aren't needed:
tar -czf supp_method_b_part2_logs.tar.gz logs/supp_method_b/*_{lstm,gru,itransformer,timemixer}_*.log
```

We'll merge into our `logs/supp_method_b/` (no filename collision since model names are disjoint) and re-run the aggregation.

---

## Things that might bite you

| Issue | Symptom | Fix |
|---|---|---|
| **TimeMixer NaN at LR=1e-2** | `loss=NaN` in log within first epoch | Fall back to `lr=0.005`; if still NaN, `lr=0.001` |
| **OOM on Traffic (N=862 vars)** | CUDA OOM in dataloader | Override `train.batch_size=8` in `run_supp_method_b_part2.sh` |
| **`_adjust_lr` errors out** | `TypeError: missing argument total_epochs` | You're on an old commit — `git pull` |
| **`gpu_resident_data` complaints** | dataloader workers fight CUDA | Set `train.gpu_resident_data=false` (already default) |
| **Empty `Test \| mse=` line in some log** | run crashed before final eval | Delete that log file → restart launcher → it'll redo |

---

## Coordination

- **Don't touch** PART 1 model files: `*_patchtst_*.log`, `*_moderntcn_*.log`, `*_segrnn_*.log` will be filled by Jay.
- **Same TAG (`supp_method_b`)** is intentional — it puts all 196 logs into one directory for easier aggregation.
- **Different SEEDS or DATASETS**: don't change. The main-results table was built with `seed=2021` and our 7-dataset list; deviating breaks the comparison.

---

## What the analysis will do with your numbers

After both PART 1 and PART 2 finish, the aggregation script (`scripts/aggregate_supp.py` — to be written) will:

1. Parse `Test | mse=...` from each of 196 logs
2. Join against the main-results table in `docs/results_table.md`
3. Produce a per-(model, dataset, horizon) Δ table
4. Decide whether the "0.55–0.59 MSE cluster" we observe in main results is a recipe artifact or a real architectural plateau

So all we need from you is **valid `Test | mse=...` lines in each of your 112 log files**.

---

## Contact

Ping Jay (`x9zou@ucsd.edu`) if you hit any blocker. For non-urgent issues, just leave a note in the master log via:

```bash
echo "[$(date '+%F %T')] NOTE  <your message>" >> logs/supp_method_b/master.log
```
