#!/usr/bin/env bash
# Usage: bash run_gpu.sh <gpu> <dataset> <h1> <h2> <cache_bs1> <cache_bs2>
#   e.g.: bash run_gpu.sh 4 electricity 336 720 32 32
#         bash run_gpu.sh 2 traffic 336 720 8 8
# Resume-aware: cache + run_distill both skip already-done items.
set -e
GPU=$1
DS=$2
H1=$3
H2=$4
BS1=$5
BS2=$6

ROOT=/home/xinkaiz/VGLG
PY=/data/xinkaiz/conda_envs/vglg/bin/python
cd "$ROOT"

echo "=== $(date '+%F %T') GPU=$GPU $DS h=$H1,$H2 (cache_bs=$BS1,$BS2) ==="

echo "--- cache $DS h=$H1 (bs=$BS1) ---"
CUDA_VISIBLE_DEVICES=$GPU $PY scripts/cache_teacher_predictions.py \
  --datasets "$DS" --horizons "$H1" --splits train --batch-size "$BS1"

echo "--- cache $DS h=$H2 (bs=$BS2) ---"
CUDA_VISIBLE_DEVICES=$GPU $PY scripts/cache_teacher_predictions.py \
  --datasets "$DS" --horizons "$H2" --splits train --batch-size "$BS2"

echo "--- distill $DS h=$H1,$H2 (3 seeds each) ---"
CUDA_VISIBLE_DEVICES=$GPU $PY scripts/run_distill.py \
  --datasets "$DS" --horizons "$H1" "$H2"

echo "=== $(date '+%F %T') GPU=$GPU $DS h=$H1,$H2 DONE ==="
