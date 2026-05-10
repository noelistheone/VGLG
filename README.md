# MetaTSF: A Unified TokenMixer Framework for Time-Series Forecasting

Course project implementing a **MetaFormer-style** time-series forecasting
framework where the only variable across our four model variants is the
TokenMixer module. We compare against eight strong baselines and add
Chronos-Bolt distillation as a knowledge-transfer ablation.

> **Core claim**: token mixing can be cleanly separated from the backbone.
> We propose a **VGLG (Variate-Gated Local-Global) TokenMixer** and compare
> it head-to-head against MLP / Conv / Attn mixers under a fixed backbone.

## Quick start

```bash
conda activate vglg

python scripts/check_env.py           # verify GPU / PyTorch / Chronos
bash   scripts/download_data.sh       # ~355MB, writes data/
pytest tests/ -v                      # 29 unit tests

# Train MetaTSF-VGLG on ETTh1, horizon 96, 10 epochs
python -m src.train.trainer model=metatsf_vglg data=etth1 train.train_epochs=10

# Verify every (model x dataset) combo runs end-to-end
python scripts/verify_all.py          # 12 models x 7 datasets x h=96
```

## Models

12 registered models across 6 families:

| Name              | Family             | File                                         |
|-------------------|--------------------|----------------------------------------------|
| `dlinear`         | Linear (no mixer)  | `src/models/baselines/dlinear.py`            |
| `lstm`            | RNN (classic)      | `src/models/baselines/lstm.py`               |
| `gru`             | RNN (classic)      | `src/models/baselines/gru.py`                |
| `segrnn`          | Modern RNN         | `src/models/baselines/segrnn.py`             |
| `timemixer`       | MLP (multi-scale)  | `src/models/baselines/timemixer.py`          |
| `moderntcn`       | CNN (large kernel) | `src/models/baselines/moderntcn.py`          |
| `patchtst`        | Transformer (patch) | `src/models/baselines/patchtst.py`          |
| `itransformer`    | Transformer (inv.) | `src/models/baselines/itransformer.py`       |
| `metatsf_mlp`     | **Ours / MLP mixer**  | `src/models/metatsf/` + `mixers/mlp_mixer.py`     |
| `metatsf_conv`    | **Ours / Conv mixer** | `src/models/metatsf/` + `mixers/conv_mixer.py`    |
| `metatsf_attn`    | **Ours / Attn mixer** | `src/models/metatsf/` + `mixers/attn_mixer.py`    |
| `metatsf_vglg`    | **Ours / VGLG mixer** (main contribution) | `src/models/metatsf/` + `mixers/vglg_mixer.py` |

The four `metatsf_*` variants share **identical** backbone, depth, width,
optimiser, and training schedule — the only difference is the TokenMixer
module. This is what makes the MLP / CNN / Transformer / Ours comparison
honest.

## MetaTSF design

```
MetaTSFBlock = LayerNorm -> TokenMixer(B, L, N) -> LayerNorm -> ChannelMLP
```

Each TokenMixer must obey one signature:

```python
class TokenMixer(nn.Module):
    def __init__(self, seq_len: int, n_vars: int, **kwargs): ...
    def forward(self, x: Tensor) -> Tensor:    # (B, L, N) -> (B, L, N)
```

To add a fifth mixer:
1. Drop a class into `src/models/metatsf/mixers/<name>_mixer.py`
2. Register it in `src/models/metatsf/mixers/__init__.py:MIXER_REGISTRY`
3. Add `configs/model/metatsf_<name>.yaml` with `mixer.type: <name>`

That's it — the backbone, trainer, and tests pick it up automatically.

## Datasets

Standard TSlib bundle plus a custom FastF1-derived dataset, downloaded into `data/`:

| Dataset          | Variates | Frequency | Split           | Source                                    |
|------------------|---------:|-----------|-----------------|-------------------------------------------|
| ETTh1, ETTh2     | 7        | 1 hour    | 12/4/4 months   | `scripts/download_data.sh`                |
| ETTm1, ETTm2     | 7        | 15 min    | 12/4/4 months   | `scripts/download_data.sh`                |
| Weather          | 21       | 10 min    | 70/10/20        | `scripts/download_data.sh`                |
| Electricity      | 321      | 1 hour    | 70/10/20        | `scripts/download_data.sh`                |
| Traffic          | 862      | 1 hour    | 70/10/20        | `scripts/download_data.sh`                |
| **f1weather**    | 7        | 1 min     | 70/10/20        | `python scripts/build_f1_weather.py --year 2023 2024` |

`f1weather` aggregates per-session weather telemetry (AirTemp, Humidity,
Pressure, Rainfall, TrackTemp, WindDirection, WindSpeed) from every Formula 1
session of one or more seasons via the [FastF1](https://github.com/theOehrly/Fast-F1)
API, concatenated along the time axis.

## Layout

```
configs/
  data/        per-dataset YAML
  model/       12 model configs (8 baselines + 4 metatsf_*)
  train/       default + distillation training configs
src/
  data/        Dataset + DataLoader
  models/
    baselines/   8 baseline reproductions
    metatsf/     unified MetaFormer backbone + mixer registry
      backbone.py, block.py
      mixers/    {mlp,conv,attn,vglg}_mixer.py + registry
    layers/      shared layers (RevIN, transformer encoder)
  losses/      distillation losses (KD trend / freq / diff)
  utils/       metrics, seeding, early stopping
  train/       Hydra-driven trainer with W&B integration
scripts/    download data, smoke test, verify_all, full sweep
tests/      29 tests: data shapes + mixer shapes/grads + backbone forward
```

## Experiments

Main matrix (Week 5-6): 7 datasets × 4 horizons × 12 models × 3 seeds = **1008 runs**.
Distillation (Week 7-8): 7 × 4 × 5 configs × 3 seeds = **420 runs**.
See `VGLG_Project_Roadmap_v2.md` for the week-by-week plan and GPU
allocation between the local 4090 and the cluster A6000s.

## Status

Week 0-4 complete:
- Environment + data + verified pipeline ✓
- 8 baselines ported and unit-tested ✓
- MetaTSF backbone + 4 mixers (MLP / Conv / Attn / VGLG) ✓
- W&B logging integrated ✓
- 12 models × 8 datasets (7 standard + f1weather) verified runnable end-to-end ✓
- 30 unit tests passing ✓
