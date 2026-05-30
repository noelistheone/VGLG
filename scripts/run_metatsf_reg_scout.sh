#!/bin/bash
# Phase 1 (round 2) — MetaTSF regularization scout.
#
# Finding from metatsf_tuned logs: every MetaTSF run OVERFITS (train loss
# crashes, val plateaus by epoch 1-4). Traffic is the worst (train 0.08 /
# val 0.46) and is the entire Avg(7) gap vs iTransformer. weight_decay is
# currently 0.0. This scout sweeps regularization (wd + dropout) plus a
# capacity sanity check, on the 3 most diagnostic datasets, h=96, VGLG only.
#
# Baseline (= existing metatsf_tuned, NOT rerun here): d_model=128 n_layers=2
# dropout=0.1 wd=0.0 cmm=2 lr=1e-3 30ep cosine pat8.
#   etth1 h96 = 0.390 | electricity h96 = 0.169 | traffic h96 = 0.566
#
# Usage:  GPU=2 ONLY="S1_wd1e4 S2_wd1e3 ..." bash scripts/run_metatsf_reg_scout.sh
#         (ONLY unset = run all configs)

set -u
cd "$(dirname "$0")/.."

GPU=${GPU:-2}
TAG=metatsf_tuned2_scout
PY=/data/xinkaiz/conda_envs/vglg/bin/python
ONLY=${ONLY:-}

export CUDA_VISIBLE_DEVICES=$GPU
mkdir -p logs/${TAG} checkpoints/${TAG}

MODEL=metatsf_vglg
DATASETS=(etth1 electricity traffic)
SEED=2021
H=96

# label | extra hydra overrides (on top of the shared 30ep/1e-3/cosine/pat8 recipe)
declare -A CFG
CFG[S1_wd1e4]="train.weight_decay=0.0001"
CFG[S2_wd1e3]="train.weight_decay=0.001"
CFG[S3_wd1e2]="train.weight_decay=0.01"
CFG[S4_do2]="model.dropout=0.2 model.mixer.dropout=0.2"
CFG[S5_do3]="model.dropout=0.3 model.mixer.dropout=0.3"
CFG[S6_do2_wd1e3]="model.dropout=0.2 model.mixer.dropout=0.2 train.weight_decay=0.001"
CFG[S7_do3_wd1e2]="model.dropout=0.3 model.mixer.dropout=0.3 train.weight_decay=0.01"
CFG[S8_dm256]="model.d_model=256"
CFG[S9_lr5e4_do2]="train.learning_rate=0.0005 model.dropout=0.2 model.mixer.dropout=0.2"
CFG[S10_cmm1]="model.channel_mlp_mult=1"

LABELS=(S1_wd1e4 S2_wd1e3 S3_wd1e2 S4_do2 S5_do3 S6_do2_wd1e3 S7_do3_wd1e2 S8_dm256 S9_lr5e4_do2 S10_cmm1)

MASTER=logs/${TAG}/master_gpu${GPU}.log
echo "[$(date '+%F %T')] BEGIN reg-scout on GPU $GPU (ONLY='${ONLY}')" | tee -a $MASTER

run_one() {
    local label=$1 data=$2
    local overrides=${CFG[$label]}
    local run_name=${label}_${data}_${MODEL}_h${H}_s${SEED}
    local logf=logs/${TAG}/${run_name}.log
    if [ -f "$logf" ] && grep -q "^Test | mse=" "$logf"; then
        echo "[$(date '+%F %T')] SKIP  $run_name" | tee -a $MASTER
        return
    fi
    echo "[$(date '+%F %T')] START $run_name :: $overrides" | tee -a $MASTER
    local t0=$(date +%s)
    $PY -m src.train.trainer \
        model=$MODEL data=$data \
        train.pred_len=$H \
        train.train_epochs=30 \
        train.patience=8 \
        train.learning_rate=0.001 \
        train.lradj=type3 \
        $overrides \
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
}

for label in "${LABELS[@]}"; do
    if [ -n "$ONLY" ] && [[ " $ONLY " != *" $label "* ]]; then continue; fi
    for data in "${DATASETS[@]}"; do
        run_one $label $data
    done
done

echo "[$(date '+%F %T')] END reg-scout on GPU $GPU" | tee -a $MASTER
