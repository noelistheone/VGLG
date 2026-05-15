# Distillation Handoff: Electricity + Traffic on 3× A6000

You're taking over the heavy datasets for the VGLG+KD distillation rows in Table 1.
P1's 4090 is handling ETTh1, ETTh2, f1weather, Weather concurrently.

## What to do

**Pull latest**:
```bash
cd VGLG  # or wherever your clone is
git pull origin main
conda activate vglg
```

**Run cache + train for Electricity + Traffic**. fp32 is fine on 48 GB; you
don't need `--fp16` (that was a 31 GB RAM workaround on the 4090 box).

The dispatchers are resume-aware: if anything dies, just re-run the same
command and it'll skip completed items.

### Option A — split across 3 GPUs in parallel (recommended, ~5-8h total)

Each A6000 takes one slice. Run in 3 separate tmux/screen panes:

```bash
# A6000-0: Electricity
CUDA_VISIBLE_DEVICES=0 nohup \
  python scripts/cache_teacher_predictions.py \
    --datasets electricity --horizons 96 192 336 720 \
    --splits train --batch-size 32 \
  > /tmp/cache_elec.log 2>&1 &
wait
CUDA_VISIBLE_DEVICES=0 python scripts/run_distill.py \
    --datasets electricity --horizons 96 192 336 720
```

```bash
# A6000-1: Traffic h=96, h=192
CUDA_VISIBLE_DEVICES=1 nohup \
  python scripts/cache_teacher_predictions.py \
    --datasets traffic --horizons 96 192 \
    --splits train --batch-size 16 \
  > /tmp/cache_traf_short.log 2>&1 &
wait
CUDA_VISIBLE_DEVICES=1 python scripts/run_distill.py \
    --datasets traffic --horizons 96 192
```

```bash
# A6000-2: Traffic h=336, h=720
CUDA_VISIBLE_DEVICES=2 nohup \
  python scripts/cache_teacher_predictions.py \
    --datasets traffic --horizons 336 720 \
    --splits train --batch-size 8 \
  > /tmp/cache_traf_long.log 2>&1 &
wait
CUDA_VISIBLE_DEVICES=2 python scripts/run_distill.py \
    --datasets traffic --horizons 336 720
```

### Option B — sequential on one A6000 (~16h total)

```bash
python scripts/cache_teacher_predictions.py \
    --datasets electricity traffic --horizons 96 192 336 720 \
    --splits train --batch-size 32
python scripts/run_distill.py \
    --datasets electricity traffic --horizons 96 192 336 720
```

## What gets produced

- `cache/teacher/electricity_h*_train.pt` (~1-30 GB each, kept local; gitignored)
- `cache/teacher/traffic_h*_train.pt` (similar)
- `logs/distill/electricity_metatsf_vglg_kd_h*_s*.log` (24 files)
- `logs/distill/traffic_metatsf_vglg_kd_h*_s*.log` (24 files)

The log files are tiny (~5 KB each). They're under `logs/` which is
`.gitignore`'d, so commit them with `-f`:

```bash
git add -f logs/distill/electricity_*.log logs/distill/traffic_*.log
git commit -m "P2: distillation logs for Electricity + Traffic"
git push origin main
```

P1 will pull, run `python scripts/update_results_table.py`, and merge into
the final table.

## Quick sanity check (run BEFORE the full sweep)

```bash
# One run to make sure your env + caches work end-to-end (~5 min)
python scripts/cache_teacher_predictions.py \
    --datasets electricity --horizons 96 --splits train --batch-size 32
python scripts/run_distill.py \
    --datasets electricity --horizons 96 --seeds 2021
ls -la logs/distill/electricity_metatsf_vglg_kd_h96_s2021.log
```

The last file should end with a line like `Test | mse=X mae=Y rmse=Z`.

## Coordination

Ping P1 once you're done so we can pull + merge before pushing the final
table. If anything crashes, the dispatcher is resume-aware; just re-run.
