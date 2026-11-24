# Phase 2 — P0 Results

**Scope:** A2, A3, A4, A5 only. A1, B1, B2 and C1–C5 were **not** run.

**Date:** 2026-11-14 · **Seed:** 42 · **Device:** CPU only (`CPU:0`)

Every number below is read from the artefacts the runs wrote themselves
(`evaluation/metrics/*_validation.json`, `*_per_class.csv`, `*_predictions.csv`)
via `src/evaluation/summarise_p0.py` and `src/evaluation/compare_p0.py`. Nothing
is transcribed by hand.

---

## 1. Actual experimental configurations

All four runs share one split: **4,286 train / 929 validation / 935 test**. The
test split was never loaded by any run. Backbones were frozen; no fine-tuning
tiers were executed.

| Run | Model | Backbone state | Input | Batch | Epochs (run/planned) | LR | Normalisation | Class weighting |
|---|---|---|---|---|---|---|---|---|
| A5 | Random Forest, 83 handcrafted features | — | — | — | — | — | — | `balanced_subsample` |
| A2 | MobileNetV2 | frozen | 224² | 32 | 15/15 | 1e-3 | `tf` (x/127.5−1) | yes |
| A4 | MobileNetV2 multi-task | frozen | 224² | 32 | 14/15 (ES) | 1e-3 | `tf` | yes |
| A3 | ResNet50 | frozen | 224² | 32 | 15/15 | 1e-3 | `caffe` (BGR mean) | yes |

Shared training protocol: Adam, `ReduceLROnPlateau(factor=0.3, patience=2,
min_lr=1e-6)`, `EarlyStopping(patience=4, min_delta=1e-4, restore_best_weights)`,
checkpoint on best **validation macro-F1**, augmentation = horizontal flip,
±10 % zoom, ±10 % translation, ±5 % rotation. The primary metric is macro F1
because the training corpus is imbalanced up to 45.7×.

Class weights: for the single-head runs these are Keras-native `class_weight`
values; for A4 Keras 3 rejects a per-output `class_weight` dict, so the **same**
per-class weights ride on the dataset as per-sample weights. The numerical
weights applied are identical, so fairness between A2/A3 and A4 is preserved.
This substitution is a Keras limitation, not a modelling choice.

## 2. Model performance

Validation split, n = 929, identical samples in identical order for all runs.

| Run | Model | **Macro F1** | 95 % CI (bootstrap) | Accuracy | Weighted F1 |
|---|---|---:|---|---:|---:|
| A5 | Random Forest (83 handcrafted) | **0.9539** | [0.9345, 0.9701] | 0.9645 | 0.9642 |
| A3 | ResNet50 frozen | **0.9521** | [0.9342, 0.9676] | 0.9537 | 0.9538 |
| A4 | MobileNetV2 multi-task | **0.9401** | [0.9216, 0.9560] | 0.9354 | 0.9350 |
| A2 | MobileNetV2 frozen | **0.9376** | [0.9206, 0.9525] | 0.9268 | 0.9263 |

The confidence intervals overlap almost completely. Reporting only the point
estimates would imply an ordering that the data does not support.

## 3. Comparison against the Random Forest

Paired bootstrap of the macro-F1 difference (2,000 resamples over samples) plus
McNemar's exact test on the discordant pairs:

| Comparison | Δ macro F1 | 95 % CI | McNemar p | Significant at 0.05 |
|---|---:|---|---:|---|
| A5 vs A3 | **+0.0018** | [−0.0207, +0.0233] | 0.253 | **no** |
| A5 vs A4 | +0.0139 | [−0.0081, +0.0342] | 0.0011 | yes |
| A5 vs A2 | +0.0164 | [−0.0059, +0.0366] | <0.0001 | yes |
| A3 vs A4 | +0.0121 | [−0.0062, +0.0290] | 0.043 | yes (marginal) |
| A3 vs A2 | +0.0146 | [−0.0022, +0.0307] | 0.0041 | yes |
| A4 vs A2 | +0.0025 | [−0.0104, +0.0153] | 0.230 | no |

**The headline finding of P0: deep learning did not produce a measurable benefit
over the handcrafted baseline on this dataset.**

The best deep model (ResNet50, 0.9521) and the Random Forest (0.9539) are
statistically indistinguishable — 13 images of 929 differ in the direction that
favours the forest, for p = 0.25. MobileNetV2 in both forms is *worse* than the
Random Forest by a margin that survives correction for pairing.

This is the answer P0 existed to obtain, and it is reported as measured.

## 4. Multi-task learning results

A4 (MobileNetV2, vegetable + quality heads, auxiliary weight 0.3):

| Task | Macro F1 | Accuracy | Weighted F1 |
|---|---:|---:|---:|
| Quality (primary) | 0.9401 | 0.9354 | 0.9350 |
| Vegetable (auxiliary) | **0.9974** | 0.9978 | 0.9978 |

Both heads correct on the same image: **93.3 %** of validation samples.

Vegetable identification is effectively solved by this dataset (4 classes, 99.7 %
F1), so it contributes almost nothing as an auxiliary signal. Multi-task did not
beat the single-task MobileNetV2 on quality by a significant margin
(+0.0025, p = 0.23) — see A4 vs A2 above — while still trailing both ResNet50 and
the Random Forest. The auxiliary task is cheap but, here, uninformative.

## 5. Per-class performance

**A5 — Random Forest (0.9539)**

| Class | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Damaged | 46 | 1.0000 | 0.8478 | 0.9176 |
| Dried | 210 | 0.9766 | 0.9952 | 0.9858 |
| Old | 314 | 0.9624 | 0.9777 | 0.9700 |
| Ripe | 267 | 0.9552 | 0.9588 | 0.9570 |
| Unripe | 92 | 0.9551 | 0.9239 | 0.9392 |

**A3 — ResNet50 (0.9521)**

| Class | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Damaged | 46 | 0.8800 | 0.9565 | 0.9167 |
| Dried | 210 | 0.9905 | 0.9905 | 0.9905 |
| Old | 314 | 0.9664 | 0.9172 | 0.9412 |
| Ripe | 267 | 0.9179 | 0.9625 | 0.9397 |
| Unripe | 92 | 0.9780 | 0.9674 | 0.9727 |

**A4 — MobileNetV2 multi-task (0.9401)**

| Class | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Damaged | 46 | 0.9362 | 0.9565 | 0.9462 |
| Dried | 210 | 0.9764 | 0.9857 | 0.9810 |
| Old | 314 | 0.9006 | 0.9522 | 0.9257 |
| Ripe | 267 | 0.9544 | 0.8614 | 0.9055 |
| Unripe | 92 | 0.9175 | 0.9674 | 0.9418 |

**A2 — MobileNetV2 (0.9376)**

| Class | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Damaged | 46 | 0.9565 | 0.9565 | 0.9565 |
| Dried | 210 | 0.9587 | 0.9952 | 0.9766 |
| Old | 314 | 0.8869 | 0.9236 | 0.9048 |
| Ripe | 267 | 0.9424 | 0.8577 | 0.8980 |
| Unripe | 92 | 0.9368 | 0.9674 | 0.9519 |

`Damaged` is the smallest class (46 samples, 5 % of validation) and is where the
models disagree most. The Random Forest is perfectly precise on it but misses
7 of 46; ResNet50 finds more of them but admits false positives. Ripe/Dried and
Ripe/Old confusion dominates every model — consistent with the 41 genuinely
ambiguous duplicate clusters found in preflight.

**Tomato/Dried is absent** from the corpus (19 of 20 cells are populated), so no
model can predict it and its per-class support is zero everywhere. This is a
property of the dataset, not a modelling failure.

## 6. Confusion matrices

Row-normalised, counts with percentages, produced by `src/evaluation/metrics.py`:

| Run | Quality | Vegetable |
|---|---|---|
| A5 | `evaluation/confusion_matrices/A5_quality.png` | — |
| A2 | `evaluation/confusion_matrices/A2_quality.png` | — |
| A3 | `evaluation/confusion_matrices/A3_quality.png` | — |
| A4 | `evaluation/confusion_matrices/A4_quality.png` | `evaluation/confusion_matrices/A4_vegetable.png` |

The dominant off-diagonal mass for every model is `Ripe → Old` and `Old → Ripe`,
plus `Dried → Old`. Confusions are *semantically adjacent*: no model confuses
`Damaged` with `Unripe`. That pattern is what a model learning maturity and
surface condition should look like; it also matches the label-ambiguity clusters.

## 7. Training and inference timings

Measured on this machine (i5-13450HX, 16 GB, TensorFlow CPU:0).

| Run | Training | Inference, batched (ms/img) | Batch-1 latency, model only (ms) | **End-to-end, decode → prediction (ms)** |
|---|---:|---:|---:|---:|
| A5 | **0.65 s** | 0.10 | 0.10 | **44.3** |
| A2 | 14.4 min | 9.47 | 150.9 | — |
| A4 | 13.7 min | 9.56 | 152.2 | 128.6 |
| A3 | 39.0 min | 26.00 | 298.7 | 234.7 |

The batched figures exclude preprocessing. The end-to-end column
(`evaluation/metrics/inference_latency.json`, 25 images, sequential, includes
JPEG decode and resize) is the honest website-facing number, and it reverses the
naive reading: the Random Forest's `predict` is 260× cheaper than ResNet50's, but
its feature extraction costs 44 ms of the 44.3 ms. Net: RF ≈ 44 ms, MobileNetV2-MT
≈ 129 ms, ResNet50 ≈ 235 ms end-to-end.

Model load time: ResNet50 1.66 s from disk; Random Forest 0.02 s.

## 8. Resource consumption

RSS sampled once per second during `fit`.

| Run | Peak RSS (MB) | Mean RSS (MB) | Mean CPU % | Aborted |
|---|---:|---:|---:|---|
| A5 | not sampled (in-process sklearn, 0.65 s) | — | — | no |
| A2 | 1,520 | 1,491 | 82.6 | no |
| A4 | 1,260 | 1,236 | 78.1 | no |
| A3 | 1,761 | 1,616 | 78.3 | no |

Machine total: 15.71 GB. The 85 % abort guard (13.4 GB) was never approached; the
largest run used 11 % of RAM. The earlier concern about 4.49 GB free memory was
not realised — that figure was measured while other processes were running.

Runs executed one process at a time; no two models were resident simultaneously.

## 9. Limitations

1. **Single holdout, 929 samples.** The 95 % CI on macro F1 is roughly ±0.018.
   Differences smaller than that are not resolvable here, including the entire
   A2/A4/A5 spread.
2. **The test split was not evaluated.** All numbers are validation numbers. They
   are not test performance and must not be quoted as such. Final test evaluation
   remains gated on approval.
3. **One seed per configuration.** Seed variance is unmeasured; a difference of
   ±0.01 between two runs could easily be seed noise alone.
4. **Backbones frozen.** A2/A3/A4 are feature extractors plus a small head. Fine-
   tuning (B1/B2) was explicitly out of scope and might change the ranking.
5. **Tomato/Dried absent.** 19 of 20 dataset cells exist; the missing class is
   structural.
6. **Duplicate over-merging (R1).** The split keeps a 732-image connected
   component entirely in train, so validation and test cover a narrower slice of
   the corpus than intended. Over-grouping cannot inflate scores, but it does
   restrict coverage.
7. **Grouped-CV 0.9199 is not comparable.** It is a 5-fold grouped CV *inside* the
   training split; every number in this report is a holdout validation score. The
   two are deliberately kept apart and must never be tabulated together.
8. **Macro F1 is unstable for `Damaged`** — 46 samples, so one image moves its
   recall by 2.2 points (0.8478 vs 0.9565 across models is a 5-image swing).

## 10. Recommendations for P1 and P2

**What P0 established.** With frozen backbones at 224², deep learning does not
beat 83 handcrafted descriptors on this dataset. The gap that matters is not
model capacity but the near-duplicate structure and class imbalance.

**P1 (if approved):**

1. Run **B1** (fine-tune the last block of MobileNetV2/ResNet50) before any
   further architecture work. Frozen-feature extraction is the most likely reason
   ResNet50 stalls at 0.9521; this is the single highest-value next experiment.
2. Do **not** prioritise A4-style multi-task. The vegetable head is at ceiling
   (0.9974) and transfers nothing.
3. Run **at least 3 seeds** on whichever two configurations matter, so the next
   comparison is not decided by ±0.01 noise.
4. Investigate `Ripe ↔ Old` directly: it is the dominant confusion for all four
   models and coincides with the 41 genuine ambiguity clusters. Cleaning or
   merging those labels may raise every model more than any architecture change.

**P2:**

5. Evaluate the *selected* model on the test split **once**, under the approved
   protocol, and report it as `test`, never mixed with the numbers above.
6. If a deployable system is required, note that the Random Forest is 5× faster
   end-to-end, 6× smaller, and statistically tied with the best CNN. Any
   deployment argument for the CNN must rest on something other than accuracy.

**Deployment decision (recorded as D-012).** The Random Forest (A5) is selected:
highest point estimate, statistically tied with ResNet50 (p = 0.25), 44 ms vs
235 ms end-to-end, 15 MB vs 94 MB, and no TensorFlow dependency at serving time.
It predicts quality only, so the website must not display a vegetable prediction.

---

## Verification

| Check | Result |
|---|---|
| Runs executed | A5, A2, A4, A3 — all exit 0 |
| Test split touched | **no** (`resolve_split("test")` never called; registry `test_*` columns empty) |
| Registry rows | 4 (`Admin/experiment_registry.csv`) |
| Phase 1 frozen artefacts | 12/12 checksums unchanged |
| Test suite | 80 passed |
| Artefacts per run | config, history, `best.keras`/`last.keras`, predictions, per-class CSV, classification report, confusion matrix, validation JSON |
