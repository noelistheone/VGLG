#!/usr/bin/env bash
# Distill-only (cache assumed to exist). Used when another GPU already cached.
# Usage: bash run_distill_only.sh <gpu> <dataset> <horizon>
set -e
GPU=$1
DS=$2
H=$3

ROOT=/home/xinkaiz/VGLG
PY=/data/xinkaiz/conda_envs/vglg/bin/python
cd "$ROOT"

echo "=== $(date '+%F %T') GPU=$GPU distill-only $DS h=$H ==="
CUDA_VISIBLE_DEVICES=$GPU $PY scripts/run_distill.py \
  --datasets "$DS" --horizons "$H"
echo "=== $(date '+%F %T') GPU=$GPU distill-only $DS h=$H DONE ==="
