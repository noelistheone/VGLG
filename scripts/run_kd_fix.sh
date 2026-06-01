#!/bin/bash
# KD fix verification: no-KD vs fixed-KD (raw-input teacher cache) on the one
# dataset where Chronos is competent (electricity). Identical student/recipe/data;
# the only difference is whether the (fixed, ortho-normalized, soft-target) KD
# loss against a RAW-fed Chronos teacher is added.
#
# Recipe = main (10ep / lr1e-4 / type1 / pat3, batch 16) for comparability with
# the documented no-KD VGLG baseline.
#
# Usage: GPU=6 HORIZONS="96 192" SEEDS="2021" bash scripts/run_kd_fix.sh

set -u
cd "$(dirname "$0")/.."
GPU=${GPU:-6}
PY=/data/xinkaiz/conda_envs/vglg/bin/python
TAG=kd_fix
HORIZONS=${HORIZONS:-"96 192"}
SEEDS=${SEEDS:-"2021"}
RAW_CACHE=cache/teacher_raw
export CUDA_VISIBLE_DEVICES=$GPU
mkdir -p logs/${TAG}
MASTER=logs/${TAG}/master_gpu${GPU}.log
echo "[$(date '+%F %T')] BEGIN kd_fix GPU $GPU h='${HORIZONS}' seeds='${SEEDS}'" | tee -a $MASTER

done_ok() { [ -f "$1" ] && grep -q "^Test | mse=" "$1"; }

run_nokd() {
    local h=$1 s=$2
    local logf=logs/${TAG}/electricity_vglg_nokd_h${h}_s${s}.log
    if done_ok "$logf"; then echo "[$(date '+%F %T')] SKIP nokd h$h s$s" | tee -a $MASTER; return; fi
    echo "[$(date '+%F %T')] START nokd h$h s$s" | tee -a $MASTER
    $PY -m src.train.trainer model=metatsf_vglg data=electricity \
        train.pred_len=$h train.batch_size=16 train.num_workers=4 \
        train.train_epochs=10 seed=$s tag=${TAG} \
        run_name=electricity_vglg_nokd_h${h}_s${s} > "$logf" 2>&1
    local mse=$(grep "^Test | mse=" "$logf" | tail -1 | sed 's/^Test | mse=\([0-9.]*\) .*/\1/')
    echo "[$(date '+%F %T')] OK   nokd h$h s$s mse=${mse}" | tee -a $MASTER
}

run_kd() {
    local h=$1 s=$2
    local logf=logs/${TAG}/electricity_metatsf_vglg_kd_h${h}_s${s}.log
    if done_ok "$logf"; then echo "[$(date '+%F %T')] SKIP kd h$h s$s" | tee -a $MASTER; return; fi
    echo "[$(date '+%F %T')] START kd(raw) h$h s$s" | tee -a $MASTER
    $PY -m src.train.distill_trainer --config-name=distill_default data=electricity \
        train.pred_len=$h train.batch_size=16 train.num_workers=4 \
        train.train_epochs=10 train.teacher_cache_dir=${RAW_CACHE} \
        seed=$s tag=${TAG} run_name=electricity_metatsf_vglg_kd_h${h}_s${s} > "$logf" 2>&1
    local mse=$(grep "^Test | mse=" "$logf" | tail -1 | sed 's/^Test | mse=\([0-9.]*\) .*/\1/')
    echo "[$(date '+%F %T')] OK   kd(raw) h$h s$s mse=${mse}" | tee -a $MASTER
}

for h in $HORIZONS; do
  for s in $SEEDS; do
    run_nokd $h $s
    run_kd $h $s
  done
done
echo "[$(date '+%F %T')] END kd_fix GPU $GPU" | tee -a $MASTER
