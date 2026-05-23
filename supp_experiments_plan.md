# MetaTSF — Supplementary Experiments Plan (Method B, split across 2 machines)

**Owner**: Jay · **Last updated**: 2026-05-22 · **Status**: PART 1 in progress; PART 2 handoff in `HANDOFF.md`

## Motivation

Main results use a single uniform training recipe (`epochs=10, lr=1e-4, lradj=type1, patience=3`) across all 12 models. This is fair within the MetaTSF family (4 variants share a backbone and were designed under this recipe), but it likely undertrains several baselines whose original papers use 30–100 epochs with cosine/OneCycle schedules and higher LRs. The goal of this supplementary run is to **rerun each baseline under a recipe that respects its original paper**, then compare to main results.

This is not a global re-tune — we change only `train_epochs`, `patience`, `learning_rate`, and `lradj`. Architecture-internal hyperparameters (d_model, depth, kernels, …) stay at each model's existing config. The 4 MetaTSF variants are not retrained: their recipe is already the one we designed them for.

---

## Scope

**Datasets** (7, dropping f1weather): `etth1 etth2 ettm1 ettm2 weather electricity traffic`
**Horizons** (4): `{96, 192, 336, 720}`
**Seeds** (1): `2021` only — direction-of-change study; variance already nailed by 3-seed main matrix.
**Models** (7 baselines, MetaTSF × 4 + DLinear unchanged): PatchTST, ModernTCN, SegRNN, TimeMixer, iTransformer, LSTM, GRU.

**Total**: 7 × 7 × 4 × 1 = **196 runs**.

### Why 7 datasets, not 4

Initial plan was 4 datasets (ETTh1 / Weather / Electricity / Traffic). After PatchTST showed Traffic h=96 crossing the plan's 10 % Δ threshold (−11.1 % vs main), Jay invoked the dataset expansion clause to add **ETTh2 / ETTm1 / ETTm2** to all models. f1weather remains excluded — no original-paper baseline exists to compare against.

---

## Split across machines (added 2026-05-22)

The single-GPU budget for 196 runs was projected at 60–90 h; splitting roughly in half across two machines gets results in 1–1.5 days.

| Part | Owner | Machine | GPU | Models | Runs | Est. time |
|------|-------|---------|-----|--------|-----:|----------:|
| **PART 1** | Jay (this repo) | main workstation | 4 | PatchTST, ModernTCN, SegRNN | 84 (26 already done) | ~30 h |
| **PART 2** | handoff (see `HANDOFF.md`) | TBD | TBD | TimeMixer, iTransformer, LSTM, GRU | 112 | ~15–22 h |

Launchers:
- `scripts/run_supp_method_b_part1.sh` — heavy Tier 1 models (PART 1)
- `scripts/run_supp_method_b_part2.sh` — light models (PART 2, can be copied to handoff machine as-is)
- `scripts/run_supp_method_b.sh` — original combined version, kept for reference

Both parts write to the same `logs/supp_method_b/` directory using `tag=supp_method_b`. File names are `<dataset>_<model>_h<H>_s2021.log`; since the two parts have disjoint model sets, there is **no collision risk**. PART 2's handoff machine produces a tarball of its 112 logs which we untar into our `logs/supp_method_b/` for aggregation.

---

## Pre-flight code change (done 2026-05-22)

`_adjust_lr` in `src/train/trainer.py` and `src/train/distill_trainer.py` hard-coded the cosine period to `/10`. For any `train_epochs ≠ 10` the curve wrapped to negative cosine values and the LR reflected back up — wrong for our 30/50-epoch budgets.

Fix shipped today:
- Added `total_epochs: int = 10` parameter; cosine arm uses it.
- Added a `constant` scheme for future ablations.
- Caller passes `cfg.train.train_epochs`.
- Default arg preserves the existing 10-epoch behaviour bit-for-bit — previously logged numbers are unaffected.

Smoke-tested with `model=patchtst data=etth1 train.train_epochs=3 train.lradj=type3`.

---

## Bug found mid-run (2026-05-22)

The original launcher's `master.log` "OK" lines printed an `mse=...` value via `sed 's/.*mse=\([0-9.]*\).*/\1/'`. Greedy `.*` matched through to the *last* `mse=` in the trainer's `Test | mse=X mae=Y rmse=Z` line — so the logged value was actually **RMSE**, not MSE. Caused a panic early on where PatchTST looked +60 % worse than main; correcting the regex showed −5 % better.

Fix: anchored sed to `^Test | mse=\([0-9.]*\) ` in both split launchers. The per-run log files were always correct — only the summary line in `master.log` was misleading.

---

## Per-model recipe changes

The recipe deltas below are unchanged from the 2026-05-21 plan; only the dataset list expanded.

### Tier 1 — large gap from original paper (PART 1)

| Model | epochs | patience | lr | lradj | Reason |
|---|---:|---:|---|---|---|
| **PatchTST** | 30 | 8 | 1e-4 | type3 | Paper uses 100ep + OneCycle; cosine is closest in our trainer |
| **ModernTCN** | 50 | 10 | 1e-3 | type3 | Paper uses 50–100ep + cosine/OneCycle; lr 10× higher than ours |
| **SegRNN** | 30 | 8 | 1e-3 | type3 | Paper uses 30ep + cosine; lr 10× higher than ours |

### Tier 1 — large gap, but lighter compute (PART 2)

| Model | epochs | patience | lr | lradj | Reason |
|---|---:|---:|---|---|---|
| **TimeMixer** | 10 | 5 | **1e-2** | type3 | Paper default lr is 100× ours — biggest single change |

### Tier 2 — smaller gap (PART 2)

| Model | epochs | patience | lr | lradj | Reason |
|---|---:|---:|---|---|---|
| **iTransformer** | 10 | 3 | 5e-4 | type1 | Paper default lr; otherwise close to our recipe |
| **LSTM** | 10 | 3 | 1e-3 | type1 | TSlib default; RNNs are lr-sensitive |
| **GRU** | 10 | 3 | 1e-3 | type1 | Same as LSTM |

### Tier 3 — no rerun

- **DLinear**: linear model, converges in ≤ 5 epochs at any sane lr; current config is already a fair fight.
- **MetaTSF × 4** (MLP / Conv / Attn / VGLG): the recipe we designed them for; rerunning would invalidate the 4-mixer ablation comparison.

---

## Progress as of 2026-05-22 14:42

| Part | Model | Status |
|------|-------|--------|
| PART 1 | PatchTST | 26/28 done; running `traffic_h336` then `traffic_h720` |
| PART 1 | ModernTCN | not started — queued after PatchTST |
| PART 1 | SegRNN | not started — queued after ModernTCN |
| PART 2 | TimeMixer / iTransformer / LSTM / GRU | **handed off — see HANDOFF.md** |

### Early signal from the 26 done PatchTST runs

13 of 13 runs across ETTh1 / Weather / Electricity / Traffic-h=96 show:
- 11/13 better than main, 2/13 worse
- Average Δ = **−5.2 %** MSE
- Largest improvement: Traffic h=96 **−11.1 %**
- Largest regression: ETTh1 h=720 **+9.4 %**

The "−5 % across the board with one notable outlier" pattern is consistent with the hypothesis that fair training recipe makes PatchTST modestly better but not transformatively. Full numbers are in `logs/supp_method_b/*.log` and will be aggregated after both parts finish.

---

## Output

```
logs/supp_method_b/<dataset>_<model>_h<H>_s2021.log         # one per run, 196 total
logs/supp_method_b/master.log                                # tee of START/OK/SKIP/FAIL events
checkpoints/supp_method_b/<dataset>_<model>_h<H>_s2021.pt    # best-val checkpoint per run
```

Final deliverable: `docs/supp_method_b_results.md` containing per-model Δ table, updated leaderboard with each baseline reported under its rerun recipe, and the answer to the "0.55-0.59 cluster: real or artifact?" question.

Aggregation script `scripts/aggregate_supp.py` (to write after both parts finish) — parses the 196 logs, joins against main numbers in `docs/results_table.md`, emits the deliverable markdown.

---

## Risks

| Risk | Mitigation |
|------|---|
| PART 2 handoff machine has different conda env path | `HANDOFF.md` allows `PY=` override |
| TimeMixer at lr=1e-2 NaNs on some dataset | `HANDOFF.md` pre-flight smoke test + 5e-3 fallback path |
| Traffic + ModernTCN OOM at 50 epoch (memory cache pressure) | Fall back to `train.batch_size=4` |
| Filename collision between PART 1 / PART 2 | None possible — disjoint model sets |
| Either machine pre-empted by another user | Idempotent restart picks up where it stopped |

---

## Out of scope

- LR / d_model grid sweeps per model (deferred; the epoch + lr change alone settles the recipe question)
- Repeat across seeds 2022 / 2023 — only if final results merit publication-grade averages
- f1weather (custom dataset; no comparable baseline)
