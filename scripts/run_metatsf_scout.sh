#!/bin/bash
# Phase 1 — MetaTSF recipe scouting on ETTh1 + Electricity, h=96 only.
# 4 mixers x 3 recipes x 2 datasets = 24 runs.
#
# Usage:  GPU=2 bash scripts/run_metatsf_scout.sh

set -u
cd "$(dirname "$0")/.."

GPU=${GPU:-2}
TAG=metatsf_scout
PY=/data/xinkaiz/conda_envs/vglg/bin/python

export CUDA_VISIBLE_DEVICES=$GPU
mkdir -p logs/${TAG} checkpoints/${TAG}

MIXERS=(metatsf_mlp metatsf_conv metatsf_attn metatsf_vglg)
DATASETS=(etth1 electricity)
SEED=2021
H=96

MASTER=logs/${TAG}/master.log
echo "[$(date '+%F %T')] BEGIN scout on GPU $GPU" | tee -a $MASTER

run_recipe() {
    local label=$1 epochs=$2 pat=$3 lr=$4 lradj=$5
    for model in "${MIXERS[@]}"; do
        for data in "${DATASETS[@]}"; do
            local run_name=${label}_${data}_${model}_h${H}_s${SEED}
            local logf=logs/${TAG}/${run_name}.log
            if [ -f "$logf" ] && grep -q "^Test | mse=" "$logf"; then
                echo "[$(date '+%F %T')] SKIP  $run_name (done)" | tee -a $MASTER
                continue
            fi
            echo "[$(date '+%F %T')] START $run_name (ep=$epochs lr=$lr lradj=$lradj)" | tee -a $MASTER
            local t0=$(date +%s)
            $PY -m src.train.trainer \
                model=$model data=$data \
                train.pred_len=$H \
                train.train_epochs=$epochs \
                train.patience=$pat \
                train.learning_rate=$lr \
                train.lradj=$lradj \
                seed=$SEED tag=${TAG} \
                run_name=$run_name \
                > "$logf" 2>&1
            local rc=$?
            local secs=$(( $(date +%s) - t0 ))
            if [ $rc -eq 0 ]; then
                local mse=$(grep "^Test | mse=" "$logf" | tail -1 | sed 's/^Test | mse=\([0-9.]*\) .*/\1/')
                echo "[$(date '+%F %T')] OK    $run_name (${secs}s) mse=${mse}" | tee -a $MASTER
            else
                echo "[$(date '+%F %T')] FAIL  $run_name rc=$rc (${secs}s)" | tee -a $MASTER
            fi
        done
    done
}

# R1 — PatchTST-clone (epochs 30, lr 1e-4, cosine, pat 8)
run_recipe R1 30 8 0.0001 type3
# R2 — mid LR
run_recipe R2 30 8 0.0005 type3
# R3 — aggressive LR
run_recipe R3 30 8 0.001  type3

echo "[$(date '+%F %T')] END scout" | tee -a $MASTER
