# Phase 2 Preflight Report

**Date:** 2026-11-14
**Status:** Preflight complete. **No training run has been launched.**
**Scope:** Stage 1 (preflight), Stage 2 (benchmark), Stage 3 (matrix). Stages 4–5
await approval.

---

## 1. Hardware benchmark results

Measured by `src/training/benchmark.py` — 90 s of real TensorFlow work, not
specification extrapolation.

### Environment

| Item | Value |
|---|---|
| Platform | Windows 11, native (no WSL) |
| CPU | Intel Core i5-13450HX — **10 physical / 16 logical** cores |
| RAM | 15.71 GB total, **4.49 GB available at benchmark time** |
| Disk free | 40.1 GB of 475 GB |
| Python | 3.12.16 (project-local, provisioned with uv) |
| TensorFlow | 2.21.0 (`tensorflow-cpu`) |
| Devices visible to TF | `CPU:0` only — **no GPU** |
| Mixed precision | `float32` (CPU) |

### Input pipeline throughput

| Pipeline | Images/s | Implied cost per epoch |
|---|---:|---:|
| `tf_native` (TF ops, on-the-fly) | **3,891** | ~1.1 s |
| `enhanced` (OpenCV preset via `numpy_function`) | **718** | ~6.0 s |

Both are an order of magnitude faster than the slowest model step, so **data
loading is not the bottleneck** and the ablation can afford the OpenCV presets.

### Model timings

| Model | Params (M) | Step (s) | Train img/s | **min/epoch** | Inference ms/img |
|---|---:|---:|---:|---:|---:|
| `custom_cnn_single` @224 | 0.42 | 1.285 | 24.9 | **2.87** | 8.47 |
| `mobilenetv2_single_frozen` @224 | 2.42 | 0.361 | 88.6 | **0.81** | 19.39 |
| `mobilenetv2_single_finetune` (last 30) @224 | 2.42 | 0.410 | 78.0 | **0.92** | 18.46 |
| `mobilenetv2_single_full` @224 | 2.42 | 1.843 | 17.4 | **4.12** | 15.95 |
| `resnet50_single_frozen` @224 | 23.85 | 0.723 | 44.3 | **1.61** | 37.40 |
| `resnet50_single_frozen` @160 | 23.85 | 0.408 | 78.3 | **0.91** | 24.95 |
| `mobilenetv2_multitask_frozen` @224 | 2.59 | 0.257 | 124.6 | **0.57** | 15.62 |

**Unexpected result worth stating:** the *from-scratch* custom CNN is **3.5× more
expensive per epoch than a frozen ResNet50** and 5× more than a frozen
MobileNetV2. A frozen backbone only back-propagates through a small head, so the
pretrained model is not the expensive option here — the deep untrained network is.
This inverts the usual intuition and is why A1 gets the largest epoch budget.

**Second finding:** MobileNetV2 fine-tuning the last 30 layers costs 0.92 min/epoch
versus 4.12 for full fine-tuning — a **4.5× saving** for a mode that is usually
the better-behaved one on small datasets.

Run-to-run variance was measured at ~7% (custom CNN: 3.07 then 2.87 min/epoch
across two runs); the table reports the later measurement.

**Memory.** Peak process RSS reached 6.2 GB, but this is *cumulative across the
seven models benchmarked in one process* — TensorFlow does not return memory to
the OS. Per-model deltas are much smaller and are recorded in
`evaluation/reports/hardware_benchmark.json`. Available RAM (4.49 GB) is the real
constraint and is listed as risk R3.

---

## 2. Feature-selection leakage audit

`src/features/leakage_audit.py` → `evaluation/reports/leakage_audit.json`.

| Audit | Result |
|---|---|
| Group integrity across CV folds | **0 shared groups** across all 5 folds |
| Scaling fitted per fold vs globally | 0.9199 vs 0.9199 (Δ **+0.0000**) |
| Nested vs flat feature selection | 0.7635 vs 0.7528 (Δ **−0.0107**) |
| Phase 1 headline reproduced exactly | **Yes** (0.9199 identical to 4 dp) |
| Phase 1 artefacts unchanged | **Yes** — all 12 checksummed files identical |
| Test split featurised | **No** — only `features_train.*` exists |

**Verdict: the Phase 1 procedure is sound.**

Three points deserve honesty rather than a clean bill of health:

1. **The scaling-leakage delta is exactly zero**, not "small". A `StandardScaler`
   is an affine, label-independent transform and the downstream estimator is a
   Random Forest, which is invariant to monotone per-feature rescaling. The
   measured delta therefore shows the *mechanism was absent from the score*, not
   that the experiment proved the pipeline safe. The pipeline is correct by
   construction (`StandardScaler` is inside the `Pipeline`, re-fitted per fold);
   the audit confirms the reported number is reproduced by that correct path.
2. **The selection-leakage delta is negative (−0.0107).** A flat selection of the
   top-20 features by F-score did *not* inflate the estimate. This is not evidence
   that selection leakage is harmless: k=20 discards most of the signal (both
   arms score ~0.76 against 0.92 for all 83 features), and the variance between
   folds dominates the 1-point difference. The honest reading is *"no measurable
   inflation at this k"*, not *"selection leakage does not exist"*.
3. **The Phase 1 headline used no feature selection at all.** All 83 features were
   cross-validated with groups intact, so neither leak mechanism was in the
   reporting path. `forest_importance` is fitted on the full training split, but
   only to *rank features for the report*; it never selects the columns that are
   scored. That distinction is what keeps the number honest, and it is now
   documented in code.

---

## 3. Duplicate-label investigation

`src/data/label_conflicts.py` and `src/data/cluster_diagnostics.py`.

### The headline finding: Phase 1's clustering over-merged badly

Investigating the 42 "label-conflicting groups" uncovered a flaw in Phase 1's
duplicate detection that is more consequential than the label conflicts themselves.

Union-find computes the **transitive closure** of "dHash distance ≤ 6". On a corpus
of near-identical white-background studio photographs, that chains unrelated
images into one component:

| Threshold | Components | Largest component | Largest **diameter** | Distinct labels in largest | Label-pure share |
|---:|---:|---:|---:|---:|---:|
| 0 | 6 | 2 | 0 | 1 | 100.0% |
| 2 | 118 | 5 | 4 | 1 | 98.3% |
| 4 | 369 | 25 | 18 | 5 | 94.3% |
| **6 (used)** | **572** | **732** | **39** | **18** | 92.7% |
| 8 | 490 | 2,349 | 45 | 20 | 92.5% |

A genuine duplicate burst cannot have a diameter (39) six times its own threshold
(6). At threshold 6, **one component absorbs 732 images — 11.9% of the corpus —
spanning all four vegetables and 18 labels.** That is an artefact, not a burst.

### Consequence for the label-conflict count

The Phase 1 claim of "42 groups / 866 images with conflicting labels" is
**materially misleading**, and is corrected here:

* **732 of those 866 images** belong to the single chained component. They are not
  label noise; they are unrelated images that were over-merged.
* The remaining **41 genuine small clusters involve 134 images — 2.2% of the
  corpus.** That is the real label-ambiguity figure.

Genuine conflicts concentrate on semantically adjacent classes:
`Dried ↔ Ripe` (22 groups), `Dried ↔ Old` (21), `Old ↔ Ripe` (21). This matches
the visual inspection in Phase 1 and is expected: `Old`, `Dried` and `Ripe`
describe overlapping visible states.

### Consequence for the split

The 732-image component landed **entirely in the training split** (0 in
validation, 0 in test), as the grouping rule requires. Rebuilding the split at
threshold 2 moves **2,813 of 6,150 images (45.8%)** into different splits.

**This is a real methodological risk and is recorded as R1.** Per the instruction
to keep the splits unchanged, Phase 2 proceeds on the existing split. The
over-grouping is *conservative* for leakage — it cannot inflate test scores — but
it means 11.9% of the corpus is absent from validation and test, so those splits
sample a somewhat narrower slice of the data than a threshold-2 split would.
A threshold-2 split is pre-specified as an optional robustness check; it is **not**
run unless approved, because it would invalidate comparability with any Phase 2
result obtained on the current split.

---

## 4. Proposed model configurations

Full detail in `Admin/experiment_matrix.md`. Summary:

| ID | Model | Head | Input | Epochs | LR | Pretrained |
|---|---|---|---|---:|---:|---|
| A1 | Custom CNN, 4 conv blocks, 0.42 M params | quality | 224 | 20 | 1e-3 | no |
| A2 | MobileNetV2 frozen | quality | 224 | 15 | 1e-3 | ImageNet |
| A3 | ResNet50 frozen | quality | 224 | 15 | 1e-3 | ImageNet |
| A4 | MobileNetV2 multitask | veg + quality | 224 | 15 | 1e-3 | ImageNet |
| A5 | Handcrafted RF (83 features) | quality | — | — | — | — |
| B1 | MobileNetV2, last 30 layers | quality | 224 | 10 | 1e-4 | ImageNet |
| B2 | MobileNetV2, full fine-tune | quality | 224 | 8 | 1e-4 | ImageNet |
| C1–C5 | Ablations of the winner | quality | 224 | 15 | as winner | — |

**Multi-task justification.** Every image carries exactly one vegetable label and
one quality label, so a two-head model with a shared trunk is supported by the
annotations. It is *not* a 20-way joint classifier — `Tomato/Dried` has no data,
so the joint label space has 19 cells. The vegetable head is an auxiliary task
(weight 0.3 vs 1.0 for quality): quality is the project's primary question, and
the auxiliary head exists to stop the trunk collapsing onto a single cue. The
custom CNN is also built for both heads so the architecture comparison is
like-for-like.

**Pretrained-input correctness.** MobileNetV2 and ResNet50 expect different input
distributions (`x/127.5−1` and Caffe-style BGR mean subtraction respectively).
Both are implemented explicitly in the data pipeline
(`pipeline.normalize` modes `tf` and `caffe`) rather than with `Lambda` layers, so
saved models load without `custom_objects`. The mode is recorded in model metadata.

---

## 5. Estimated training time

Derived from the measured per-epoch costs above.

| Stage | Runs | Worst case | Realistic (with early stopping) |
|---|---:|---:|---:|
| A — primary comparison | 5 | 102 min | 65–80 min |
| B — transfer depth | 2 | 42 min | 30–40 min |
| C — ablations | 5 | 65 min | 45–55 min |
| **Total** | **12** | **≈ 3 h 30 min** | **≈ 2–2.5 h** |

The single largest item is A1 (custom CNN, 57 min worst case). If time is short,
A2–A4 plus A5 alone (~45 min) already answer the primary question; A1 can follow.

---

## 6. Experiment priority

| Priority | Runs | Rationale |
|---|---|---|
| **P0 — must run** | A2, A3, A4, A5 | Answers "does transfer learning beat 83 classical features on this corpus?" — the central question. |
| **P1 — should run** | A1, C5 | A1 is the honest from-scratch baseline; C5 (no augmentation) tests whether augmentation helps at all at this dataset size. |
| **P2 — valuable** | C1, C2, C3, C4 | Class weighting and the preprocessing presets. Directly tests Phase 1's measured claims on *classification* rather than image statistics. |
| **P3 — optional** | B1, B2 | Fine-tuning depth. B1 is cheap (9 min); B2 (33 min) only if B1 shows promise. |

If only P0 is approved, the result is still a complete, fair comparison against
the handcrafted baseline, which is the project's stated bar.

---

## 7. Resource requirements

| Resource | Requirement | Headroom |
|---|---|---|
| Wall clock | 2–3.5 h for the full matrix | Stage C is optional |
| Disk | < 0.5 GB checkpoints + ~0.4 GB data | 40.1 GB free |
| RAM | ~2–4 GB peak per model (single process) | 4.5 GB available — **tight, see R3** |
| GPU | none used | — |
| Network | one-off ImageNet weight download (~110 MB) | already cached |
| CPU | full 16 threads during training | machine will be busy |

---

## 8. Identified risks

| # | Risk | Severity | Mitigation |
|---|---|---|---|
| **R1** | **Split is distorted by duplicate over-merging** — one 732-image chained component (11.9% of corpus) sits entirely in train; a threshold-2 split would relocate 45.8% of images. | **High** | Proceed on the unchanged split as instructed and report it. Pre-specified threshold-2 re-split available as a robustness check on approval. All Phase 2 conclusions are explicitly conditioned on this split. |
| **R2** | **The handcrafted baseline is strong (0.920 grouped macro F1).** CNNs may not beat it on 4.3k images. | **High** | This is a legitimate possible outcome and will be reported as such. It is the reason A5 runs on the identical split with the identical metric. |
| **R3** | Available RAM was only 4.49 GB at benchmark time; full fine-tuning of ResNet50 could approach the limit. | Medium | ResNet50 is only run frozen (2.4 GB observed). B2 (full fine-tune) is MobileNetV2 at 3.9 GB peak. Monitor RSS during runs. |
| **R4** | Validation/test sets are small (929 / 935 images, ~4 images in the smallest cell), so macro F1 has wide confidence intervals. | Medium | Report per-class support and bootstrap intervals; never quote a bare point estimate for `Tomato/Damaged`. |
| **R5** | CPU-only training with no GPU; run-to-run variance ≈7%. | Low | Fixed seeds; report single runs and label them as such. |
| **R6** | Pretrained weights depend on network access at first use. | Low | Already downloaded during benchmarking and cached in `~/.keras`. |
| **R7** | Real residual near-duplicate correlation may exceed the dHash threshold, so even grouped metrics are mildly optimistic. | Medium | Stated in `reports/limitations.md`; carried into the Phase 2 report. |
| **R8** | Multi-task loss weighting (0.3/1.0) is a judgement call, not tuned. | Low | Documented as a fixed choice; not presented as optimal. |

---

## Compliance statement

* Phase 1 artefacts: **unchanged** — verified by SHA-256 for 12 files.
* Test split: **still untouched**. No model has seen it.
* No training has been launched. `Admin/experiment_registry.csv` contains a header
  only, with no rows, because no experiment has been executed.
