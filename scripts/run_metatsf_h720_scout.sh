#!/bin/bash
# Phase 3 — horizon-aware weight-decay scout for the long horizons.
#
# Round-2 (wd=1e-3, all horizons) over-regularizes at h336/h720: 7 cells
# regress vs main, almost all at h720 (ETTh2/Weather/ETTm1/ETTm2). We already
# have the two endpoints of the wd curve at every cell:
#   wd=0     -> logs/metatsf_tuned/   (round 1)
#   wd=1e-3  -> logs/metatsf_tuned2/  (round 2)
# This scout fills the INTERMEDIATE wd values at the long horizons only, so we
# can pick a per-horizon wd (dataset-independent) that recovers the regressions
# without losing the Traffic/short-horizon gains. VGLG only (representative).
#
# Usage: GPU=3 WDLIST="0.0001 0.0005" WORKLIST="ettm1:336 ettm1:720" \
#          bash scripts/run_metatsf_h720_scout.sh

set -u
cd "$(dirname "$0")/.."

GPU=${GPU:-3}
TAG=metatsf_h720_scout
PY=/data/xinkaiz/conda_envs/vglg/bin/python
MIXER=${MIXER:-metatsf_vglg}
WDLIST=${WDLIST:?set WDLIST="0.0001 0.0005"}
WORKLIST=${WORKLIST:?set WORKLIST="data:h ..."}

export CUDA_VISIBLE_DEVICES=$GPU
mkdir -p logs/${TAG} checkpoints/${TAG}
SEED=2021
MASTER=logs/${TAG}/master_gpu${GPU}.log
echo "[$(date '+%F %T')] BEGIN h720-scout GPU $GPU wd='${WDLIST}' work='${WORKLIST}'" | tee -a $MASTER

run_one() {
    local data=$1 h=$2 wd=$3
    local wdtag=${wd//./p}
    local run_name=${data}_${MIXER}_h${h}_wd${wdtag}_s${SEED}
    local logf=logs/${TAG}/${run_name}.log
    if [ -f "$logf" ] && grep -q "^Test | mse=" "$logf"; then
        echo "[$(date '+%F %T')] SKIP  $run_name" | tee -a $MASTER; return
    fi
    echo "[$(date '+%F %T')] START $run_name (wd=$wd)" | tee -a $MASTER
    local t0=$(date +%s)
    $PY -m src.train.trainer \
        model=$MIXER data=$data \
        train.pred_len=$h train.train_epochs=30 train.patience=8 \
        train.learning_rate=0.001 train.lradj=type3 train.weight_decay=$wd \
        seed=$SEED tag=${TAG} run_name=$run_name \
        > "$logf" 2>&1
    local rc=$?; local secs=$(( $(date +%s) - t0 ))
    if [ $rc -eq 0 ]; then
        local mse=$(grep "^Test | mse=" "$logf" | tail -1 | sed 's/^Test | mse=\([0-9.]*\) .*/\1/')
        echo "[$(date '+%F %T')] OK    $run_name (${secs}s) mse=${mse}" | tee -a $MASTER
    else
        echo "[$(date '+%F %T')] FAIL  $run_name rc=$rc (${secs}s)" | tee -a $MASTER
    fi
}

for wd in $WDLIST; do
    for token in $WORKLIST; do
        run_one "${token%%:*}" "${token##*:}" "$wd"
    done
done
echo "[$(date '+%F %T')] END h720-scout GPU $GPU" | tee -a $MASTER
