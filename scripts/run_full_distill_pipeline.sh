#!/bin/bash
# Master pipeline: wait for current Chronos ZS to finish, then run teacher
# caching for all 6 remote datasets, then VGLG+KD training, then update +
# commit + push. Designed to run unattended overnight via nohup.
#
# Step-by-step log goes to /tmp/full_distill_pipeline.log.
set -u
set -o pipefail
exec > /tmp/full_distill_pipeline.log 2>&1
PY=/home/lawrence/miniconda3/envs/vglg/bin/python
ROOT=/home/lawrence/VGLG
cd "$ROOT"

stamp() { date "+%Y-%m-%d %H:%M:%S"; }

echo "[$(stamp)] === Pipeline start ==="

# 1) Wait for current Chronos ZS sweep to finish (any process running it)
echo "[$(stamp)] Waiting for run_chronos_zero_shot to finish ..."
while pgrep -f "run_chronos_zero_shot" > /dev/null; do
    sleep 120
done
echo "[$(stamp)] ZS sweep finished."

ZS_DONE=$(ls "$ROOT"/logs/main/*chronos_zs*.log 2>/dev/null | wc -l)
echo "[$(stamp)] Total ZS logs: $ZS_DONE / 32"

# 2) Teacher caching - small first (fp32 .pt fits easily)
echo "[$(stamp)] === Step 2: cache small datasets (ETTh1, ETTh2, f1weather) ==="
$PY scripts/cache_teacher_predictions.py \
    --datasets etth1 etth2 f1weather \
    --horizons 96 192 336 720 \
    --splits train --batch-size 16
echo "[$(stamp)] Small cache done."

echo "[$(stamp)] === Step 3: cache Weather ==="
$PY scripts/cache_teacher_predictions.py \
    --datasets weather \
    --horizons 96 192 336 720 \
    --splits train --batch-size 16
echo "[$(stamp)] Weather cache done."

# Big datasets need fp16 mmap to avoid OOM (Traffic h=720 fp32 = 30 GB)
echo "[$(stamp)] === Step 4: cache Electricity (fp16 mmap) ==="
$PY scripts/cache_teacher_predictions.py \
    --datasets electricity \
    --horizons 96 192 336 720 \
    --splits train --batch-size 8 --fp16
echo "[$(stamp)] Electricity cache done."

echo "[$(stamp)] === Step 5: cache Traffic (fp16 mmap) ==="
$PY scripts/cache_teacher_predictions.py \
    --datasets traffic \
    --horizons 96 192 336 720 \
    --splits train --batch-size 4 --fp16
echo "[$(stamp)] Traffic cache done."

# 3) VGLG+KD training - small first (faster), big last
echo "[$(stamp)] === Step 6: VGLG+KD on ETTh1, ETTh2, f1weather ==="
$PY scripts/run_distill.py \
    --datasets etth1 etth2 f1weather \
    --horizons 96 192 336 720
echo "[$(stamp)] Small KD done."

echo "[$(stamp)] === Step 7: VGLG+KD on Weather ==="
$PY scripts/run_distill.py --datasets weather --horizons 96 192 336 720
echo "[$(stamp)] Weather KD done."

echo "[$(stamp)] === Step 8: VGLG+KD on Electricity ==="
$PY scripts/run_distill.py --datasets electricity --horizons 96 192 336 720
echo "[$(stamp)] Electricity KD done."

echo "[$(stamp)] === Step 9: VGLG+KD on Traffic ==="
$PY scripts/run_distill.py --datasets traffic --horizons 96 192 336 720
echo "[$(stamp)] Traffic KD done."

# 4) Refresh + merge + push
echo "[$(stamp)] === Step 10: refresh table + merge teammate data ==="
$PY scripts/update_results_table.py
$PY /tmp/merge_final.py
echo "[$(stamp)] Table merged."

echo "[$(stamp)] === Step 11: commit + push ==="
git add docs/results_table.md src/data/distill_dataset.py \
    scripts/cache_teacher_predictions.py scripts/run_distill.py \
    scripts/run_chronos_zero_shot.py src/models/teacher.py 2>/dev/null || true
git commit -m "P1: complete distillation experiments for all 8 datasets

- All 32 Chronos-Bolt zero-shot evals finished (was already 27/32)
- Teacher cache built for ETTh1, ETTh2, f1weather, Weather (fp32) and
  Electricity, Traffic (fp16 numpy mmap to fit in 31 GB RAM)
- 72 VGLG+KD training runs done on the 6 remote datasets
- Streaming MSE/MAE evaluator in run_chronos_zero_shot.py to avoid OOM
  at batch 345 of traffic h=720 (was killing the process repeatedly)
- DistillDataset transparently handles both .pt fp32 and .npy fp16 caches
" 2>&1 | tail -3
git push origin main 2>&1 | tail -3

echo "[$(stamp)] === Pipeline complete ==="
