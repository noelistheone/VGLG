#!/bin/bash
# Quick end-to-end smoke test: train each model 1-2 epochs on ETTh1 to make
# sure the pipeline is healthy before launching real experiments.
set -e

cd "$(dirname "$0")/.."

MODELS=(dlinear timemixer itransformer patchtst moderntcn vglg_mlp vglg_cnn vglg_transformer)

for m in "${MODELS[@]}"; do
    echo "=== $m ==="
    python -m src.train.trainer model=$m data=etth1 \
        train.train_epochs=2 train.batch_size=32 train.num_workers=2 \
        tag=smoke
done

echo ""
echo "All smoke tests passed."
