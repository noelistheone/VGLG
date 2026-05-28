#!/bin/bash
# Phase 2 — MetaTSF full sweep under tuned recipe R3.
# Recipe: 30 ep, lr=1e-3, cosine, patience=8, seed=2021
# Scope:  4 mixers x 7 datasets x 4 horizons = 112 runs
#
# Usage:  GPU=2 bash scripts/run_metatsf_tuned.sh

set -u
cd "$(dirname "$0")/.."

GPU=${GPU:-2}
TAG=metatsf_tuned
PY=/data/xinkaiz/conda_envs/vglg/bin/python

export CUDA_VISIBLE_DEVICES=$GPU
mkdir -p logs/${TAG} checkpoints/${TAG}

MIXERS=(metatsf_mlp metatsf_conv metatsf_attn metatsf_vglg)
# Order: small → large so quick coverage first; Traffic last (biggest variate count).
DATASETS=(etth1 etth2 ettm1 ettm2 weather electricity traffic)
HORIZONS=(96 192 336 720)
SEED=2021

MASTER=logs/${TAG}/master.log
echo "[$(date '+%F %T')] BEGIN metatsf_tuned on GPU $GPU" | tee -a $MASTER

run_one() {
    local model=$1 data=$2 h=$3
    local run_name=${data}_${model}_h${h}_s${SEED}
    local logf=logs/${TAG}/${run_name}.log
    if [ -f "$logf" ] && grep -q "^Test | mse=" "$logf"; then
        echo "[$(date '+%F %T')] SKIP  $run_name" | tee -a $MASTER
        return
    fi
    echo "[$(date '+%F %T')] START $run_name (R3: 30ep lr=1e-3 cosine pat=8)" | tee -a $MASTER
    local t0=$(date +%s)
    $PY -m src.train.trainer \
        model=$model data=$data \
        train.pred_len=$h \
        train.train_epochs=30 \
        train.patience=8 \
        train.learning_rate=0.001 \
        train.lradj=type3 \
        seed=$SEED tag=${TAG} \
        > "$logf" 2>&1
    local rc=$?
    local secs=$(( $(date +%s) - t0 ))
    if [ $rc -eq 0 ]; then
        local mse=$(grep "^Test | mse=" "$logf" | tail -1 | sed 's/^Test | mse=\([0-9.]*\) .*/\1/')
        echo "[$(date '+%F %T')] OK    $run_name (${secs}s) mse=${mse}" | tee -a $MASTER
    else
        echo "[$(date '+%F %T')] FAIL  $run_name rc=$rc (${secs}s)" | tee -a $MASTER
    fi
}

for data in "${DATASETS[@]}"; do
    for h in "${HORIZONS[@]}"; do
        for model in "${MIXERS[@]}"; do
            run_one $model $data $h
        done
    done
done

echo "[$(date '+%F %T')] END metatsf_tuned" | tee -a $MASTER
