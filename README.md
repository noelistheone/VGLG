# VGLG-TSF: Variate-Gated Local-Global Mixer for Time Series Forecasting

Course project skeleton implementing a unified backbone-agnostic time-series mixer
(VGLG) with MLP / CNN / Transformer wrappers, evaluated against standard baselines
(DLinear, TimeMixer, iTransformer, PatchTST, ModernTCN) and Chronos-Bolt distillation.

## Quick start

```bash
# 1. Activate env (created by conda create -n vglg python=3.10)
conda activate vglg

# 2. Verify environment
python scripts/check_env.py

# 3. Download datasets (writes into data/, ~355MB)
bash scripts/download_data.sh

# 4. Sanity-check data + model shapes
pytest tests/ -v

# 5. Train any (model x dataset x horizon) combo
python -m src.train.trainer model=vglg_mlp data=etth1 train.pred_len=96 train.train_epochs=10

# 6. Run the full smoke test (every model, 2 epochs each)
bash scripts/smoke_test.sh
```

## Models

| Name             | Type        | File                                      | Source        |
|------------------|-------------|-------------------------------------------|---------------|
| `dlinear`        | linear      | `src/models/baselines/dlinear.py`         | AAAI 2023     |
| `timemixer`      | MLP         | `src/models/baselines/timemixer.py`       | ICLR 2024     |
| `itransformer`   | Transformer | `src/models/baselines/itransformer.py`    | ICLR 2024 Spt |
| `patchtst`       | Transformer | `src/models/baselines/patchtst.py`        | ICLR 2023     |
| `moderntcn`      | CNN         | `src/models/baselines/moderntcn.py`       | ICLR 2024 Spt |
| `vglg_mlp`       | MLP (ours)  | `src/models/vglg/mlp_wrapper.py`          | —             |
| `vglg_cnn`       | CNN (ours)  | `src/models/vglg/cnn_wrapper.py`          | —             |
| `vglg_transformer` | Transformer (ours) | `src/models/vglg/tf_wrapper.py`  | —             |

All baselines are self-contained ports from THUML/Time-Series-Library
(and the official ModernTCN repo for ModernTCN). The shared transformer
encoder lives in `src/models/layers/transformer.py`.

### Smoke-test results (ETTh1, horizon=96, 1 epoch, batch=32, no tuning)

| Model              | Params  | Test MSE | Test MAE |
|--------------------|--------:|---------:|---------:|
| dlinear            | 19k     | 0.527    | 0.496    |
| timemixer          | 75k     | 0.468    | 0.457    |
| itransformer       | 842k    | 0.400    | 0.412    |
| patchtst           | 547k    | 0.401    | 0.404    |
| moderntcn          | 213k    | 0.397    | 0.407    |
| vglg_mlp           | 30k     | 0.457    | 0.453    |
| vglg_cnn           | 30k     | 0.457    | 0.453    |
| vglg_transformer   | 295k    | 0.417    | 0.425    |

These are 1-epoch numbers, not tuned — they only confirm the pipeline runs
end-to-end. Full Week 1-2 reproductions require 10+ epochs and the published
hyperparameters (e.g. PatchTST uses input_len=336, not 96).

## Datasets

Downloaded from THUML/Time-Series-Library bundle into `data/`:

| Dataset        | Variates | Frequency | Split           |
|----------------|---------:|-----------|-----------------|
| ETTh1, ETTh2   | 7        | 1 hour    | 12/4/4 months   |
| ETTm1, ETTm2   | 7        | 15 min    | 12/4/4 months   |
| Weather        | 21       | 10 min    | 70/10/20        |
| Electricity    | 321      | 1 hour    | 70/10/20        |
| Traffic        | 862      | 1 hour    | 70/10/20        |

## Layout

```
configs/    Hydra YAML configs for data / model / train
src/
  data/        PyTorch Dataset + DataLoader factory
  models/
    baselines/   Baseline reproductions (DLinear, TimeMixer, iTransformer, PatchTST, ModernTCN)
    vglg/        VGLG block + 3 wrappers (MLP/CNN/Transformer)
    layers/      Shared layers (RevIN, Transformer encoder)
  losses/      Distillation losses (trend/freq/diff KD)
  utils/       Metrics, seeding, early stopping
  train/       Hydra-driven training entry point
scripts/    Top-level run scripts (download data, smoke test, batch experiments)
tests/      Unit tests (data shapes, model forward+backward)
notebooks/  Exploratory analysis
```

## Adding a new model

1. Drop a class into `src/models/baselines/<name>.py` (or `vglg/`). The
   class must accept `seq_len`, `pred_len`, `n_vars` as kwargs plus its own
   hyperparameters, and `forward(x_enc, x_mark_enc=None, x_dec=None,
   x_mark_dec=None) -> Tensor[B, pred_len, N]`.
2. Register it in `src/models/builder.py:MODEL_REGISTRY`.
3. Add a `configs/model/<name>.yaml` with at least `name: <name>` and the
   model's hyperparameters. Extra keys are tolerated (`**_unused`).

Then `python -m src.train.trainer model=<name> data=etth1` just works.

## Status

Week 0 + Week 1-2 baseline porting complete. Use `bash scripts/smoke_test.sh`
to verify the pipeline end-to-end.
