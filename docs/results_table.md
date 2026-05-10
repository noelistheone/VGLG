# Final Results Tables

All experiments use the same trainer (`src/train/trainer.py`), data splits
(TSlib-standard 12/4/4 months for ETT, 70/10/20 elsewhere), and seed averaging
(seeds = 2021 / 2022 / 2023). Numbers are **MSE / MAE on the test set**,
mean across 3 seeds. Lower is better. **Bold** = best in column. <u>Underline</u>
= second best.

Reproducible from W&B with `tags=main`:
```bash
python notebooks/build_results_table.py --tag main --out docs/results_table.md
```

---

## Table 1 — Main results, average MSE across horizons {96, 192, 336, 720}

The headline table. Each cell is the mean MSE across 4 horizons and 3 seeds.

| Model              | Family             |  #Params | ETTh1 | ETTh2 | ETTm1 | ETTm2 | Weather | Electricity | Traffic | f1weather | **Avg** |
|--------------------|--------------------|---------:|------:|------:|------:|------:|--------:|------------:|--------:|----------:|--------:|
| DLinear            | Linear (no mixer)  |    `—`   |  `—`  |  `—`  |  `—`  |  `—`  |   `—`   |     `—`     |   `—`   |    `—`    |   `—`   |
| LSTM               | RNN (classic)      |    `—`   |  `—`  |  `—`  |  `—`  |  `—`  |   `—`   |     `—`     |   `—`   |    `—`    |   `—`   |
| GRU                | RNN (classic)      |    `—`   |  `—`  |  `—`  |  `—`  |  `—`  |   `—`   |     `—`     |   `—`   |    `—`    |   `—`   |
| SegRNN             | Modern RNN         |    `—`   |  `—`  |  `—`  |  `—`  |  `—`  |   `—`   |     `—`     |   `—`   |    `—`    |   `—`   |
| TimeMixer          | MLP (multi-scale)  |    `—`   |  `—`  |  `—`  |  `—`  |  `—`  |   `—`   |     `—`     |   `—`   |    `—`    |   `—`   |
| ModernTCN          | CNN (large kernel) |    `—`   |  `—`  |  `—`  |  `—`  |  `—`  |   `—`   |     `—`     |   `—`   |    `—`    |   `—`   |
| iTransformer       | Trf. (inverted)    |    `—`   |  `—`  |  `—`  |  `—`  |  `—`  |   `—`   |     `—`     |   `—`   |    `—`    |   `—`   |
| PatchTST           | Trf. (patch)       |    `—`   |  `—`  |  `—`  |  `—`  |  `—`  |   `—`   |     `—`     |   `—`   |    `—`    |   `—`   |
| **MetaTSF-MLP**    | **Ours / MLP**     |    `—`   |  `—`  |  `—`  |  `—`  |  `—`  |   `—`   |     `—`     |   `—`   |    `—`    |   `—`   |
| **MetaTSF-Conv**   | **Ours / Conv**    |    `—`   |  `—`  |  `—`  |  `—`  |  `—`  |   `—`   |     `—`     |   `—`   |    `—`    |   `—`   |
| **MetaTSF-Attn**   | **Ours / Attn**    |    `—`   |  `—`  |  `—`  |  `—`  |  `—`  |   `—`   |     `—`     |   `—`   |    `—`    |   `—`   |
| **MetaTSF-VGLG**   | **Ours / VGLG**    |    `—`   |  `—`  |  `—`  |  `—`  |  `—`  |   `—`   |     `—`     |   `—`   |    `—`    |   `—`   |
| _Chronos-Bolt_ (zero-shot) | Foundation | 205M     |  `—`  |  `—`  |  `—`  |  `—`  |   `—`   |     `—`     |   `—`   |    `—`    |   `—`   |
| **MetaTSF-VGLG + KD**| **Ours + KD**    |    `—`   |  `—`  |  `—`  |  `—`  |  `—`  |   `—`   |     `—`     |   `—`   |    `—`    |   `—`   |

> **Reading guide**: the four MetaTSF rows share an identical backbone, depth,
> width, optimiser, and training schedule. The only difference is the TokenMixer
> module. This is the controlled comparison the paper centres on.

---

## Table 1a — ETTh1 (per-horizon detail)

| Model              | h=96 MSE | h=96 MAE | h=192 MSE | h=192 MAE | h=336 MSE | h=336 MAE | h=720 MSE | h=720 MAE |
|--------------------|---------:|---------:|----------:|----------:|----------:|----------:|----------:|----------:|
| DLinear            |   `—`    |   `—`    |    `—`    |    `—`    |    `—`    |    `—`    |    `—`    |    `—`    |
| LSTM               |   `—`    |   `—`    |    `—`    |    `—`    |    `—`    |    `—`    |    `—`    |    `—`    |
| GRU                |   `—`    |   `—`    |    `—`    |    `—`    |    `—`    |    `—`    |    `—`    |    `—`    |
| SegRNN             |   `—`    |   `—`    |    `—`    |    `—`    |    `—`    |    `—`    |    `—`    |    `—`    |
| TimeMixer          |   `—`    |   `—`    |    `—`    |    `—`    |    `—`    |    `—`    |    `—`    |    `—`    |
| ModernTCN          |   `—`    |   `—`    |    `—`    |    `—`    |    `—`    |    `—`    |    `—`    |    `—`    |
| iTransformer       |   `—`    |   `—`    |    `—`    |    `—`    |    `—`    |    `—`    |    `—`    |    `—`    |
| PatchTST           |   `—`    |   `—`    |    `—`    |    `—`    |    `—`    |    `—`    |    `—`    |    `—`    |
| **MetaTSF-MLP**    |   `—`    |   `—`    |    `—`    |    `—`    |    `—`    |    `—`    |    `—`    |    `—`    |
| **MetaTSF-Conv**   |   `—`    |   `—`    |    `—`    |    `—`    |    `—`    |    `—`    |    `—`    |    `—`    |
| **MetaTSF-Attn**   |   `—`    |   `—`    |    `—`    |    `—`    |    `—`    |    `—`    |    `—`    |    `—`    |
| **MetaTSF-VGLG**   |   `—`    |   `—`    |    `—`    |    `—`    |    `—`    |    `—`    |    `—`    |    `—`    |
| Chronos zero-shot  |   `—`    |   `—`    |    `—`    |    `—`    |    `—`    |    `—`    |    `—`    |    `—`    |
| **VGLG + KD**      |   `—`    |   `—`    |    `—`    |    `—`    |    `—`    |    `—`    |    `—`    |    `—`    |

## Table 1b — ETTh2 (per-horizon detail)

| Model              | h=96 MSE | h=96 MAE | h=192 MSE | h=192 MAE | h=336 MSE | h=336 MAE | h=720 MSE | h=720 MAE |
|--------------------|---------:|---------:|----------:|----------:|----------:|----------:|----------:|----------:|
| DLinear            |   `—`    |   `—`    |    `—`    |    `—`    |    `—`    |    `—`    |    `—`    |    `—`    |
| LSTM               |   `—`    |   `—`    |    `—`    |    `—`    |    `—`    |    `—`    |    `—`    |    `—`    |
| GRU                |   `—`    |   `—`    |    `—`    |    `—`    |    `—`    |    `—`    |    `—`    |    `—`    |
| SegRNN             |   `—`    |   `—`    |    `—`    |    `—`    |    `—`    |    `—`    |    `—`    |    `—`    |
| TimeMixer          |   `—`    |   `—`    |    `—`    |    `—`    |    `—`    |    `—`    |    `—`    |    `—`    |
| ModernTCN          |   `—`    |   `—`    |    `—`    |    `—`    |    `—`    |    `—`    |    `—`    |    `—`    |
| iTransformer       |   `—`    |   `—`    |    `—`    |    `—`    |    `—`    |    `—`    |    `—`    |    `—`    |
| PatchTST           |   `—`    |   `—`    |    `—`    |    `—`    |    `—`    |    `—`    |    `—`    |    `—`    |
| **MetaTSF-MLP**    |   `—`    |   `—`    |    `—`    |    `—`    |    `—`    |    `—`    |    `—`    |    `—`    |
| **MetaTSF-Conv**   |   `—`    |   `—`    |    `—`    |    `—`    |    `—`    |    `—`    |    `—`    |    `—`    |
| **MetaTSF-Attn**   |   `—`    |   `—`    |    `—`    |    `—`    |    `—`    |    `—`    |    `—`    |    `—`    |
| **MetaTSF-VGLG**   |   `—`    |   `—`    |    `—`    |    `—`    |    `—`    |    `—`    |    `—`    |    `—`    |
| Chronos zero-shot  |   `—`    |   `—`    |    `—`    |    `—`    |    `—`    |    `—`    |    `—`    |    `—`    |
| **VGLG + KD**      |   `—`    |   `—`    |    `—`    |    `—`    |    `—`    |    `—`    |    `—`    |    `—`    |

## Table 1c — ETTm1 (per-horizon detail)

| Model              | h=96 MSE | h=96 MAE | h=192 MSE | h=192 MAE | h=336 MSE | h=336 MAE | h=720 MSE | h=720 MAE |
|--------------------|---------:|---------:|----------:|----------:|----------:|----------:|----------:|----------:|
| (rows omitted; same 14 rows as Table 1a) | | | | | | | | |

## Table 1d — ETTm2

(same structure as Table 1a)

## Table 1e — Weather

(same structure as Table 1a)

## Table 1f — Electricity

(same structure as Table 1a)

## Table 1g — Traffic

(same structure as Table 1a)

## Table 1h — f1weather (FastF1)

| Model              | h=96 MSE | h=96 MAE | h=192 MSE | h=192 MAE | h=336 MSE | h=336 MAE | h=720 MSE | h=720 MAE |
|--------------------|---------:|---------:|----------:|----------:|----------:|----------:|----------:|----------:|
| DLinear            |   `—`    |   `—`    |    `—`    |    `—`    |    `—`    |    `—`    |    `—`    |    `—`    |
| LSTM               |   `—`    |   `—`    |    `—`    |    `—`    |    `—`    |    `—`    |    `—`    |    `—`    |
| GRU                |   `—`    |   `—`    |    `—`    |    `—`    |    `—`    |    `—`    |    `—`    |    `—`    |
| SegRNN             |   `—`    |   `—`    |    `—`    |    `—`    |    `—`    |    `—`    |    `—`    |    `—`    |
| TimeMixer          |   `—`    |   `—`    |    `—`    |    `—`    |    `—`    |    `—`    |    `—`    |    `—`    |
| ModernTCN          |   `—`    |   `—`    |    `—`    |    `—`    |    `—`    |    `—`    |    `—`    |    `—`    |
| iTransformer       |   `—`    |   `—`    |    `—`    |    `—`    |    `—`    |    `—`    |    `—`    |    `—`    |
| PatchTST           |   `—`    |   `—`    |    `—`    |    `—`    |    `—`    |    `—`    |    `—`    |    `—`    |
| **MetaTSF-MLP**    |   `—`    |   `—`    |    `—`    |    `—`    |    `—`    |    `—`    |    `—`    |    `—`    |
| **MetaTSF-Conv**   |   `—`    |   `—`    |    `—`    |    `—`    |    `—`    |    `—`    |    `—`    |    `—`    |
| **MetaTSF-Attn**   |   `—`    |   `—`    |    `—`    |    `—`    |    `—`    |    `—`    |    `—`    |    `—`    |
| **MetaTSF-VGLG**   |   `—`    |   `—`    |    `—`    |    `—`    |    `—`    |    `—`    |    `—`    |    `—`    |
| Chronos zero-shot  |   `—`    |   `—`    |    `—`    |    `—`    |    `—`    |    `—`    |    `—`    |    `—`    |
| **VGLG + KD**      |   `—`    |   `—`    |    `—`    |    `—`    |    `—`    |    `—`    |    `—`    |    `—`    |

---

## Table 2 — Ablation of VGLG TokenMixer (3 datasets × 3 horizons)

Run on ETTh1, Weather, Electricity at horizons {96, 336, 720}, 3 seeds. Reported
as MSE averaged across all 9 (dataset, horizon) cells.

| Variant                          | ETTh1 avg | Weather avg | Elec. avg | Overall avg | Δ vs full |
|----------------------------------|----------:|------------:|----------:|------------:|----------:|
| **MetaTSF-VGLG** (full)          |   `—`     |    `—`      |    `—`    |     `—`     |   `0.000` |
| – fixed gate at g=0.5            |   `—`     |    `—`      |    `—`    |     `—`     |   `—`     |
| – local path only (g=1)          |   `—`     |    `—`      |    `—`    |     `—`     |   `—`     |
| – global path only (g=0)         |   `—`     |    `—`      |    `—`    |     `—`     |   `—`     |
| kernel_size = 15 (default 31)    |   `—`     |    `—`      |    `—`    |     `—`     |   `—`     |
| kernel_size = 51                 |   `—`     |    `—`      |    `—`    |     `—`     |   `—`     |
| rank = 4 (default 8)             |   `—`     |    `—`      |    `—`    |     `—`     |   `—`     |
| rank = 16                        |   `—`     |    `—`      |    `—`    |     `—`     |   `—`     |
| rank = 32                        |   `—`     |    `—`      |    `—`    |     `—`     |   `—`     |
| no RevIN                         |   `—`     |    `—`      |    `—`    |     `—`     |   `—`     |

---

## Table 3 — Distillation comparison (Δ vs no-KD, MSE only)

For each of the 8 datasets at horizon=96, compare the **same student model**
trained with vs without Chronos-Bolt distillation. Negative Δ = KD helps.

| Student                | ETTh1 | ETTh2 | ETTm1 | ETTm2 | Weather | Elec. | Traffic | f1weather |
|------------------------|------:|------:|------:|------:|--------:|------:|--------:|----------:|
| MetaTSF-MLP, no KD     |  `—`  |  `—`  |  `—`  |  `—`  |   `—`   |  `—`  |   `—`   |    `—`    |
| MetaTSF-MLP, with KD   |  `—`  |  `—`  |  `—`  |  `—`  |   `—`   |  `—`  |   `—`   |    `—`    |
| Δ (KD – noKD)          |  `—`  |  `—`  |  `—`  |  `—`  |   `—`   |  `—`  |   `—`   |    `—`    |
| MetaTSF-VGLG, no KD    |  `—`  |  `—`  |  `—`  |  `—`  |   `—`   |  `—`  |   `—`   |    `—`    |
| MetaTSF-VGLG, with KD  |  `—`  |  `—`  |  `—`  |  `—`  |   `—`   |  `—`  |   `—`   |    `—`    |
| Δ (KD – noKD)          |  `—`  |  `—`  |  `—`  |  `—`  |   `—`   |  `—`  |   `—`   |    `—`    |

---

## Run accounting

Total runs used to populate every table above:

| Block                  | Datasets | Horizons | Models | Seeds | Total runs |
|------------------------|---------:|---------:|-------:|------:|-----------:|
| Table 1 (main matrix)  |    8     |    4     |   12   |   3   |   **1152** |
| Table 2 (ablation)     |    3     |    3     |    9   |   3   |     **243** |
| Table 3 (distillation) |    8     |    4     |    5   |   3   |     **480** |
| Chronos zero-shot      |    8     |    4     |    1   |   1   |      **32** |
| **Grand total**        |          |          |        |       |  **≈1907** |

Wall-clock estimate at 5 min/run averaged across (model, dataset, horizon),
distributed over 4 GPUs (1× 4090 + 3× A6000): **~40 hours**, i.e. one long
weekend.
