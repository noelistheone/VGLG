#!/bin/bash
# Method B supplementary — PART 1 (this machine, GPU 4).
# Models: PatchTST + ModernTCN + SegRNN (the heavy Tier 1 models).
# PART 2 (TimeMixer + iTransformer + LSTM + GRU) is run elsewhere — see HANDOFF.md.
#
# Idempotent: skips runs whose log already contains "Test | mse=".
#
# Usage:
#   GPU=4 bash scripts/run_supp_method_b_part1.sh

set -u
cd "$(dirname "$0")/.."

GPU=${GPU:-4}
TAG=supp_method_b
PY=/data/xinkaiz/conda_envs/vglg/bin/python

export CUDA_VISIBLE_DEVICES=$GPU
mkdir -p logs/${TAG} checkpoints/${TAG}

DATASETS=(etth1 etth2 ettm1 ettm2 weather electricity traffic)
HORIZONS=(96 192 336 720)
SEED=2021

MASTER_LOG=logs/${TAG}/master.log
echo "[$(date '+%F %T')] BEGIN method-b PART 1 on GPU $GPU" | tee -a $MASTER_LOG

run_one() {
    local model=$1 epochs=$2 pat=$3 lr=$4 lradj=$5
    for data in "${DATASETS[@]}"; do
        for h in "${HORIZONS[@]}"; do
            local run_name=${data}_${model}_h${h}_s${SEED}
            local logf=logs/${TAG}/${run_name}.log
            if [ -f "$logf" ] && grep -q "^Test | mse=" "$logf"; then
                echo "[$(date '+%F %T')] SKIP  $run_name (already done)" | tee -a $MASTER_LOG
                continue
            fi
            echo "[$(date '+%F %T')] START $run_name (ep=$epochs pat=$pat lr=$lr lradj=$lradj)" | tee -a $MASTER_LOG
            local t0=$(date +%s)
            $PY -m src.train.trainer \
                model=$model data=$data \
                train.pred_len=$h \
                train.train_epochs=$epochs \
                train.patience=$pat \
                train.learning_rate=$lr \
                train.lradj=$lradj \
                seed=$SEED tag=${TAG} \
                > "$logf" 2>&1
            local rc=$?
            local secs=$(( $(date +%s) - t0 ))
            if [ $rc -ne 0 ]; then
                echo "[$(date '+%F %T')] FAIL  $run_name rc=$rc (${secs}s) — see $logf" | tee -a $MASTER_LOG
            else
                local mse=$(grep "^Test | mse=" "$logf" | tail -1 | sed 's/^Test | mse=\([0-9.]*\) .*/\1/')
                echo "[$(date '+%F %T')] OK    $run_name (${secs}s) mse=${mse}" | tee -a $MASTER_LOG
            fi
        done
    done
}

# === PART 1: the three Tier 1 heavies ===
# PatchTST already 26/28 done; finishes traffic h=336, h=720 then exits.
run_one patchtst    30  8 0.0001 type3
run_one moderntcn   50 10 0.001  type3
run_one segrnn      30  8 0.001  type3

echo "[$(date '+%F %T')] END method-b PART 1" | tee -a $MASTER_LOG
