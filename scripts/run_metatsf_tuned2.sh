#!/bin/bash
# Phase 2 (round 2) — MetaTSF full sweep under tuned recipe R3 + weight_decay.
#
# Round-1 recipe (metatsf_tuned): 30 ep / lr 1e-3 / cosine / pat 8 / wd 0.0.
# Round-2 change (this script): add weight_decay=1e-3 — the sole winner of the
# reg-scout (Traffic h96 0.566 -> 0.501, -11.5%; electricity -4.1%; ETT flat).
# Everything else identical: d_model=128 n_layers=2 dropout=0.1 cmm=2, shared
# across all 4 mixers (preserves the same-backbone thesis).
#
# Scope: 4 mixers x (data:horizon worklist). 112 runs total across all GPUs.
#
# Usage:  GPU=3 WORKLIST="ettm1:96 ettm1:192" bash scripts/run_metatsf_tuned2.sh
#   WORKLIST = space-separated data:horizon tokens; all 4 mixers run per token.

set -u
cd "$(dirname "$0")/.."

GPU=${GPU:-3}
TAG=metatsf_tuned2
PY=/data/xinkaiz/conda_envs/vglg/bin/python
WD=${WD:-0.001}
WORKLIST=${WORKLIST:?must set WORKLIST="data:h data:h ..."}

export CUDA_VISIBLE_DEVICES=$GPU
mkdir -p logs/${TAG} checkpoints/${TAG}

MIXERS=(metatsf_mlp metatsf_conv metatsf_attn metatsf_vglg)
SEED=2021

MASTER=logs/${TAG}/master_gpu${GPU}.log
echo "[$(date '+%F %T')] BEGIN tuned2 on GPU $GPU wd=$WD worklist='${WORKLIST}'" | tee -a $MASTER

run_one() {
    local model=$1 data=$2 h=$3
    local run_name=${data}_${model}_h${h}_s${SEED}
    local logf=logs/${TAG}/${run_name}.log
    if [ -f "$logf" ] && grep -q "^Test | mse=" "$logf"; then
        echo "[$(date '+%F %T')] SKIP  $run_name" | tee -a $MASTER
        return
    fi
    echo "[$(date '+%F %T')] START $run_name (R3+wd=$WD)" | tee -a $MASTER
    local t0=$(date +%s)
    $PY -m src.train.trainer \
        model=$model data=$data \
        train.pred_len=$h \
        train.train_epochs=30 \
        train.patience=8 \
        train.learning_rate=0.001 \
        train.lradj=type3 \
        train.weight_decay=$WD \
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

for token in $WORKLIST; do
    data=${token%%:*}
    h=${token##*:}
    for model in "${MIXERS[@]}"; do
        run_one $model $data $h
    done
done

echo "[$(date '+%F %T')] END tuned2 on GPU $GPU" | tee -a $MASTER
