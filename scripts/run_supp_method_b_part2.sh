#!/bin/bash
# Method B supplementary — PART 2 (handed off to another machine).
# Models: TimeMixer + iTransformer + LSTM + GRU.
# PART 1 (PatchTST + ModernTCN + SegRNN) is run on the main machine.
# See HANDOFF.md for full briefing.
#
# Idempotent: skips runs whose log already contains "Test | mse=".
#
# Usage on the handoff machine:
#   GPU=<your-free-gpu-index> bash scripts/run_supp_method_b_part2.sh
#   # or to run a single model only:
#   GPU=0 ONLY=timemixer bash scripts/run_supp_method_b_part2.sh

set -u
cd "$(dirname "$0")/.."

GPU=${GPU:-0}
ONLY=${ONLY:-}
TAG=supp_method_b
# Override PY if your conda env lives elsewhere.
PY=${PY:-/data/xinkaiz/conda_envs/vglg/bin/python}

export CUDA_VISIBLE_DEVICES=$GPU
mkdir -p logs/${TAG} checkpoints/${TAG}

DATASETS=(etth1 etth2 ettm1 ettm2 weather electricity traffic)
HORIZONS=(96 192 336 720)
SEED=2021

MASTER_LOG=logs/${TAG}/master.log
echo "[$(date '+%F %T')] BEGIN method-b PART 2 on GPU $GPU | only=${ONLY:-<all>}" | tee -a $MASTER_LOG

run_one() {
    local model=$1 epochs=$2 pat=$3 lr=$4 lradj=$5
    if [ -n "$ONLY" ] && [ "$ONLY" != "$model" ]; then
        return
    fi
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

# === PART 2: 4 lighter models (no `lr_adj` schedule change for Tier 2). ===
# Order: fastest-first so partial-run handoffs still yield clean per-model data.
# Note for TimeMixer: lr=1e-2 is paper default but 100x ours — pre-flight one
# small run before the full sweep in case of NaN; if so fall back to 5e-3.
run_one lstm         10 3 0.001  type1
run_one gru          10 3 0.001  type1
run_one itransformer 10 3 0.0005 type1
run_one timemixer    10 5 0.01   type3

echo "[$(date '+%F %T')] END method-b PART 2" | tee -a $MASTER_LOG
