#!/usr/bin/env bash
# Single-(dataset, horizon) variant. Caches teacher (skip if exists) then runs distill (3 seeds).
# Usage: bash run_gpu_one.sh <gpu> <dataset> <horizon> <cache_bs>
set -e
GPU=$1
DS=$2
H=$3
BS=$4

ROOT=/home/xinkaiz/VGLG
PY=/data/xinkaiz/conda_envs/vglg/bin/python
cd "$ROOT"

echo "=== $(date '+%F %T') GPU=$GPU $DS h=$H (cache_bs=$BS) ==="

echo "--- cache $DS h=$H (bs=$BS) ---"
CUDA_VISIBLE_DEVICES=$GPU $PY scripts/cache_teacher_predictions.py \
  --datasets "$DS" --horizons "$H" --splits train --batch-size "$BS"

echo "--- distill $DS h=$H (3 seeds) ---"
CUDA_VISIBLE_DEVICES=$GPU $PY scripts/run_distill.py \
  --datasets "$DS" --horizons "$H"

echo "=== $(date '+%F %T') GPU=$GPU $DS h=$H DONE ==="
