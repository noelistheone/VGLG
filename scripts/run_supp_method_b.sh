#!/bin/bash
# Supplementary experiments — Method B (model-by-model fair recipes).
# Runs 7 baselines x 4 datasets x 4 horizons x 1 seed = 112 runs on 1 GPU.
# See supp_experiments_plan.md for full rationale.
#
# Usage:
#   GPU=4 bash scripts/run_supp_method_b.sh                 # all tiers
#   GPU=4 TIER=1 bash scripts/run_supp_method_b.sh          # tier 1 only
#   GPU=4 ONLY=patchtst bash scripts/run_supp_method_b.sh   # single model
#
# Idempotent: skips runs whose log already contains a final "Test | mse=" line.

set -u
cd "$(dirname "$0")/.."

GPU=${GPU:-4}
TIER=${TIER:-all}     # 1 | 2 | all
ONLY=${ONLY:-}        # exact model name to restrict to (empty = no restriction)
TAG=supp_method_b
PY=/data/xinkaiz/conda_envs/vglg/bin/python

export CUDA_VISIBLE_DEVICES=$GPU
mkdir -p logs/${TAG} checkpoints/${TAG}

# Datasets ordered small → large so faster runs fill in coverage first.
# ETTh* (~17k samples), ETTm* (~70k), Weather (~53k, 21 vars),
# Electricity (~26k, 321 vars), Traffic (~17k, 862 vars).
# f1weather intentionally excluded — no original-paper baseline to compare to.
DATASETS=(etth1 etth2 ettm1 ettm2 weather electricity traffic)
HORIZONS=(96 192 336 720)
SEED=2021

MASTER_LOG=logs/${TAG}/master.log
echo "[$(date '+%F %T')] BEGIN method-b on GPU $GPU | tier=$TIER only=${ONLY:-<all>}" | tee -a $MASTER_LOG

run_one() {
    # args: model epochs patience lr lradj
    local model=$1 epochs=$2 pat=$3 lr=$4 lradj=$5
    if [ -n "$ONLY" ] && [ "$ONLY" != "$model" ]; then
        return
    fi
    for data in "${DATASETS[@]}"; do
        for h in "${HORIZONS[@]}"; do
            local run_name=${data}_${model}_h${h}_s${SEED}
            local logf=logs/${TAG}/${run_name}.log
            # Idempotent skip if previous run completed
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
            local t1=$(date +%s)
            local secs=$((t1 - t0))
            if [ $rc -ne 0 ]; then
                echo "[$(date '+%F %T')] FAIL  $run_name rc=$rc (${secs}s) — see $logf" | tee -a $MASTER_LOG
            else
                local mse=$(grep "^Test | mse=" "$logf" | tail -1 | sed 's/^Test | mse=\([0-9.]*\) .*/\1/')
                echo "[$(date '+%F %T')] OK    $run_name (${secs}s) mse=${mse}" | tee -a $MASTER_LOG
            fi
        done
    done
}

# Run order: TimeMixer is risky (LR 100x change vs main run) and may NaN, so it
# runs *last* — overnight survival of the other 96 runs is guaranteed even if
# TimeMixer crashes. Within the rest, sort by descending expected impact.
if [ "$TIER" = "all" ] || [ "$TIER" = "1" ]; then
    echo "[$(date '+%F %T')] === TIER 1 (high impact, low risk) ===" | tee -a $MASTER_LOG
    run_one patchtst    30  8 0.0001 type3
    run_one moderntcn   50 10 0.001  type3
    run_one segrnn      30  8 0.001  type3
fi

if [ "$TIER" = "all" ] || [ "$TIER" = "2" ]; then
    echo "[$(date '+%F %T')] === TIER 2 (low impact) ===" | tee -a $MASTER_LOG
    run_one itransformer 10 3 0.0005 type1
    run_one lstm         10 3 0.001  type1
    run_one gru          10 3 0.001  type1
fi

if [ "$TIER" = "all" ] || [ "$TIER" = "1" ]; then
    echo "[$(date '+%F %T')] === TIER 1 tail (TimeMixer, highest LR-change risk) ===" | tee -a $MASTER_LOG
    run_one timemixer   10  5 0.01   type3
fi

echo "[$(date '+%F %T')] END method-b" | tee -a $MASTER_LOG
