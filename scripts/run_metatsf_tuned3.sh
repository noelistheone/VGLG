#!/bin/bash
# Phase 3 rollout — horizon-aware weight decay for the long horizons.
#
# Final MetaTSF recipe: 30ep / lr1e-3 / cosine / pat8, d_model=128, dropout=0.1,
# shared across all 4 mixers, with a HORIZON-AWARE weight decay:
#   h96  -> 1e-3   h192 -> 1e-3   (live in logs/metatsf_tuned2/)
#   h336 -> 5e-4   h720 -> 1e-4   (this script, logs/metatsf_tuned3/)
# chosen from the h720 scout (4-point wd curves; longer horizon = less wd).
#
# Runs all 4 mixers at h336/h720; VGLG cells are pre-seeded from the scout and
# skipped idempotently, so this re-runs MLP/Conv/Attn (42 runs).
#
# Usage: GPU=3 WORKLIST="ettm1:336 ettm1:720" bash scripts/run_metatsf_tuned3.sh

set -u
cd "$(dirname "$0")/.."

GPU=${GPU:-3}
TAG=metatsf_tuned3
PY=/data/xinkaiz/conda_envs/vglg/bin/python
WORKLIST=${WORKLIST:?set WORKLIST="data:h ..."}

export CUDA_VISIBLE_DEVICES=$GPU
mkdir -p logs/${TAG} checkpoints/${TAG}
MIXERS=(metatsf_mlp metatsf_conv metatsf_attn metatsf_vglg)
SEED=2021
MASTER=logs/${TAG}/master_gpu${GPU}.log
echo "[$(date '+%F %T')] BEGIN tuned3 GPU $GPU work='${WORKLIST}'" | tee -a $MASTER

wd_for_h() { case "$1" in 336) echo 0.0005;; 720) echo 0.0001;; *) echo 0.0001;; esac; }

run_one() {
    local model=$1 data=$2 h=$3 wd=$4
    local run_name=${data}_${model}_h${h}_s${SEED}
    local logf=logs/${TAG}/${run_name}.log
    if [ -f "$logf" ] && grep -q "^Test | mse=" "$logf"; then
        echo "[$(date '+%F %T')] SKIP  $run_name" | tee -a $MASTER; return
    fi
    echo "[$(date '+%F %T')] START $run_name (wd=$wd)" | tee -a $MASTER
    local t0=$(date +%s)
    $PY -m src.train.trainer \
        model=$model data=$data \
        train.pred_len=$h train.train_epochs=30 train.patience=8 \
        train.learning_rate=0.001 train.lradj=type3 train.weight_decay=$wd \
        seed=$SEED tag=${TAG} \
        > "$logf" 2>&1
    local rc=$?; local secs=$(( $(date +%s) - t0 ))
    if [ $rc -eq 0 ]; then
        local mse=$(grep "^Test | mse=" "$logf" | tail -1 | sed 's/^Test | mse=\([0-9.]*\) .*/\1/')
        echo "[$(date '+%F %T')] OK    $run_name (${secs}s) mse=${mse}" | tee -a $MASTER
    else
        echo "[$(date '+%F %T')] FAIL  $run_name rc=$rc (${secs}s)" | tee -a $MASTER
    fi
}

for token in $WORKLIST; do
    data=${token%%:*}; h=${token##*:}; wd=$(wd_for_h "$h")
    for model in "${MIXERS[@]}"; do run_one "$model" "$data" "$h" "$wd"; done
done
echo "[$(date '+%F %T')] END tuned3 GPU $GPU" | tee -a $MASTER
