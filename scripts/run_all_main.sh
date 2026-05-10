#!/bin/bash
# Week 5-6: full main results matrix. Runs every (model x dataset x horizon x seed)
# combination, distributing across the 3 GPUs visible (set CUDA_VISIBLE_DEVICES if
# you only have 1).
#
# This is a placeholder that prints the commands; pipe into `parallel` once
# baselines are ported in.
set -e

cd "$(dirname "$0")/.."

DATASETS=(etth1 etth2 ettm1 ettm2 weather electricity traffic)
HORIZONS=(96 192 336 720)
MODELS=(dlinear vglg_mlp vglg_cnn vglg_transformer)
SEEDS=(2021 2022 2023)

mkdir -p logs/main

for data in "${DATASETS[@]}"; do
    for h in "${HORIZONS[@]}"; do
        for m in "${MODELS[@]}"; do
            for s in "${SEEDS[@]}"; do
                echo "python -m src.train.trainer model=$m data=$data \
                    train.pred_len=$h seed=$s tag=main \
                    > logs/main/${data}_${m}_h${h}_s${s}.log 2>&1"
            done
        done
    done
done
