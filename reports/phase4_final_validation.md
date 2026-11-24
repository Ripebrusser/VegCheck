# Phase 4 — Final Validation Report

**Vegetable Quality and Freshness Detection Using Image Processing and Deep Learning**

**Date:** 2026-11-15 · **Deployed model:** A5 Random Forest (83 handcrafted features)

---

## 1. Project overview

The project classifies a photograph of a vegetable into one of five visible quality
conditions — `Damaged`, `Dried`, `Old`, `Ripe`, `Unripe` — using classical image
processing and machine learning, and ships a Streamlit application that runs the
selected model.

The project was run in four gated phases: dataset audit and splitting (Phase 1),
controlled model comparison (Phase 2), application delivery (Phase 3), and this
final validation and release freeze (Phase 4). Every reported figure comes from a
saved artefact; none is transcribed by hand.

**The central empirical result:** three deep-learning configurations were trained and
evaluated under an identical protocol and **none demonstrated a statistically
meaningful improvement** over a Random Forest on 83 handcrafted descriptors. The
forest was deployed because it matched the best network on accuracy at a fraction of
the inference cost.

## 2. Dataset

| Property | Value |
|---|---|
| Source | New VegNet, Hugging Face redistribution of Suryawanshi et al. (2022), CC BY 4.0 |
| Images | 6,150 JPEG, RGB, all 256×256 |
| Size on disk | 125.5 MiB |
| Vegetables | Bell Pepper, Chile Pepper, New Mexico Green Chile, Tomato |
| Quality classes | Damaged, Dried, Old, Ripe, Unripe |
| Populated cells | 19 of 20 — `Tomato/Dried` is empty |
| Corrupt files | 0 |
| Exact duplicates | 0 |
| Near-duplicate clusters | 572 groups, 2,182 images at dHash ≤ 6 |

Split, grouped by duplicate cluster so no specimen crosses a boundary:

| Split | Images | Share |
|---|---:|---:|
| Train | 4,286 | 69.69 % |
| Validation | 929 | 15.11 % |
| Test | 935 | 15.21 % |

Verified in Phase 4: **0** group ids and **0** SHA-256 values span splits; **0** test
group ids or hashes appear in train or validation.

Class imbalance is severe: 2,044 `Old` versus 317 `Damaged` across the corpus (6.4×
between quality classes, 45.7× between populated vegetable × quality cells).

## 3. Dataset limitations

1. **Smaller, downsampled redistribution.** The publication reports 6,850 images;
   this corpus has 6,150, all already reduced to 256×256. Published benchmark
   numbers are therefore not comparable.
2. **One class absent.** `Tomato/Dried` has zero images, so no model can predict it
   and no metric is reported for it.
3. **Label ambiguity.** 41 clusters (134 images, 2.2 %) carry genuinely conflicting
   quality labels. An earlier "42 clusters / 866 images" figure was a transitive-
   chaining artefact of dHash union-find (one component of 732 images with a
   diameter of 39); corrected in Phase 1 documentation.
4. **Studio imagery.** Plain backgrounds, even lighting, one subject, fixed
   resolution. Performance on ordinary photographs is addressed in §13 and is
   materially worse.
5. **Retained split.** The duplicate over-grouping was not corrected; a 732-image
   block sits entirely in train. Over-grouping cannot inflate scores, but it narrows
   the population that validation and test represent.
6. **`Ripe` / `Unripe` are maturity labels**, not spoilage measurements.

## 4. Preprocessing methodology

The deployed model applies exactly four steps and nothing else:

1. decode the upload to RGB uint8;
2. resize to **224 × 224** with `INTER_AREA` (`config.IMAGE_SIZE`);
3. foreground segmentation — LAB distance from the border-estimated background,
   discarded when implausible (`area_fraction` outside a sanity band);
4. feature extraction → 83 descriptors → `StandardScaler`.

CLAHE, bilateral/median/Gaussian filtering and ImageNet normalisation exist in the
project as **evaluated alternatives** (Phase 1 study, ablations C2–C4) but are not
part of the deployed pipeline, because the deployed model never saw them.

> **Phase 4 correction.** The Phase 3 application resized to 256 px while training
> used 224 px. This was found by the integrity audit, corrected, and verified to
> produce bit-identical features to the training extraction
> (`reports/phase4_integrity_audit.md` §2). No model was retrained.

## 5. Handcrafted feature extraction

83 descriptors in three families (`src/features/handcrafted.py`):

| Family | Count | Contents |
|---|---:|---|
| Colour | 30 | RGB mean/std/skew (9), circular hue mean via sin/cos (2), saturation and value statistics (4), 12-bin hue histogram, Hasler–Süsstrunk colourfulness, excess-green, excess-red |
| Texture | 39 | GLCM contrast, dissimilarity, homogeneity, energy, correlation, ASM at distances 1/2/3 (18); uniform LBP histograms at (P=8, R=1) and (P=16, R=2) (21) |
| Shape | 7 | area fraction, normalised perimeter, circularity, aspect ratio, solidity, extent, normalised equivalent diameter |

Hue is handled circularly — averaging hue degrees would place red at both 0° and
180° and average to cyan. Texture is measured on the mask bounding box, not on a
background-filled image, which would manufacture a strong artificial edge.

## 6. Models evaluated

| Run | Model | Head | Trainable parameters |
|---|---|---|---|
| A2 | MobileNetV2, frozen backbone | quality (5) | 329 k head on 2.26 M frozen |
| A3 | ResNet50, frozen backbone | quality (5) | 10 k head on 23.6 M frozen |
| A4 | MobileNetV2, frozen backbone | vegetable (4) + quality (5) | 329 k, quality weight 1.0, vegetable 0.3 |
| A5 | Random Forest, 200 trees, `balanced_subsample` | quality (5) | 83 features, no gradient training |

Frozen backbones were a deliberate scope decision for P0: it isolates *whether the
representation transfers* from *whether fine-tuning helps*, and it fits the
available hardware. Fine-tuning tiers (B1/B2) were never authorised or run.

## 7. Experimental protocol

* One split for all four models: the Phase 1 grouped split. The test split was not
  touched during Phases 2–3.
* Identical validation population (929 images, same order) for every model.
* Primary metric **macro F1** — with 6.4× class imbalance, accuracy alone rewards
  ignoring `Damaged`.
* Early stopping, LR scheduling and checkpoint selection all on **validation** macro
  F1; the test set was never consulted.
* Fixed seed 42 for every run.
* Class weighting applied to all models (Keras `class_weight` for single-head CNNs,
  identical per-sample weights on the dataset for the multi-task model, because
  Keras 3 rejects a per-output `class_weight` dict).
* Uncertainty quantified with a percentile bootstrap over samples (1,000–2,000
  resamples) and McNemar's exact test on discordant pairs.

## 8. Phase 2 validation comparison

| Run | Model | Macro F1 | 95 % CI | Accuracy | Weighted F1 |
|---|---|---:|---|---:|---:|
| **A5** | **Random Forest** | **0.9539** | [0.9345, 0.9701] | 0.9645 | 0.9642 |
| A3 | ResNet50 | 0.9521 | [0.9342, 0.9676] | 0.9537 | 0.9538 |
| A4 | MobileNetV2 multi-task | 0.9401 | [0.9216, 0.9560] | 0.9354 | 0.9350 |
| A2 | MobileNetV2 | 0.9376 | [0.9206, 0.9525] | 0.9268 | 0.9263 |

Paired comparisons:

| Pair | Δ macro F1 | 95 % CI | McNemar p | Significant |
|---|---:|---|---:|---|
| A5 vs A3 | +0.0018 | [−0.0207, +0.0233] | 0.253 | no |
| A5 vs A4 | +0.0139 | [−0.0081, +0.0342] | 0.0011 | yes |
| A5 vs A2 | +0.0164 | [−0.0059, +0.0366] | <0.0001 | yes |
| A4 vs A2 | +0.0025 | [−0.0104, +0.0153] | 0.230 | no |

Every confidence interval overlaps. A5 and A3 are indistinguishable; the two
MobileNetV2 variants are significantly worse than the forest.

## 9. Why A5 was selected

1. **Highest point estimate** of the four.
2. **Statistically tied** with the best network (Δ +0.0018, CI spanning zero,
   p = 0.25), so accuracy cannot separate them.
3. **Cheaper**: 19.7 ms end-to-end versus 172.8 ms (A4) and 302.8 ms (A3); 15.7 MB
   versus 13–94 MB; no TensorFlow at serving time; 2.1 s cold start.
4. **Reproducible**: fits in 0.65 s, so the deployed artefact can be regenerated
   trivially.

Choosing ResNet50 because the project title says "deep learning" would have been
motivated reasoning. The networks remain fully reported; they did not win.

## 10. Final test-set results

The test split was evaluated **once**, in Phase 4, with the frozen artefact
(`sha256 53db2b14…dd83d3`) under the frozen protocol. No model, hyperparameter or
threshold was changed afterwards.

| Metric | Value |
|---|---:|
| **Macro F1 (primary)** | **0.9350** |
| 95 % CI (bootstrap) | [0.9137, 0.9553] |
| Accuracy | 0.9497 |
| Macro precision | 0.9573 |
| Macro recall | 0.9178 |
| Weighted precision | 0.9503 |
| Weighted recall | 0.9497 |
| Weighted F1 | 0.9492 |
| **Images evaluated** | **935** |
| Correct / incorrect | 888 / 47 |

**Validation versus test, clearly separated:**

| Scheme | Macro F1 | Meaning |
|---|---:|---|
| Grouped 5-fold CV (Phase 1, training split) | 0.9199 | historical benchmark, different population |
| Holdout validation (Phase 2) | 0.9539 | model selection |
| **Official test (Phase 4)** | **0.9350** | reported once, never used for selection |

The test score is 0.0189 below validation. That gap is well inside the overlap of the
two confidence intervals and is the expected outcome of a single 935-image draw —
not evidence of overfitting to validation, and not a number to be tuned against.

## 11. Per-class performance (test)

| Class | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Damaged | 49 | 0.9744 | **0.7755** | 0.8636 |
| Dried | 217 | 0.9727 | 0.9862 | 0.9794 |
| Old | 304 | 0.9470 | 0.9408 | 0.9439 |
| Ripe | 274 | 0.9271 | 0.9745 | 0.9502 |
| Unripe | 91 | 0.9651 | 0.9121 | 0.9379 |

`Damaged` has the lowest F1 (0.8636) despite the highest precision (0.9744): the
model almost never calls a healthy vegetable damaged, but misses 11 of 49 damaged
specimens. It is also the smallest class (49 images), so each error moves recall by
2 points.

All five classes have test support. No class is reported from zero examples, and
`Tomato/Dried` is absent from the corpus rather than merely unevaluated.

## 12. Confusion matrix interpretation

```
              pred:  Damaged  Dried   Old  Ripe  Unripe
true Damaged            38      0     9     2       0
     Dried               0    214     2     0       1
     Old                 0      5   286    12       1
     Ripe                1      1     4   267       1
     Unripe              0      0     1     7      83
```

47 errors in 935 images. The structure is informative:

* **`Damaged → Old` (9 of 49, 18.4 %)** is the single largest leak. Visibly damaged
  produce that has also aged is genuinely both; the label hierarchy is not clean.
* **`Old ↔ Ripe` (12 + 4)** is the second theme — the same boundary that dominated
  validation and coincides with the 41 ambiguous duplicate clusters.
* **`Unripe → Ripe` (7 of 91, 7.7 %)** is the opposite direction: young specimens
  that already look mature.
* `Dried` is almost perfect (0.9794 F1), and no class collapses entirely.

Errors are concentrated in *semantically adjacent* categories. The model is not
confusing a tomato with a pepper's condition; it is disputing a boundary that human
annotators also dispute.

**Confidence tracks accuracy on this in-distribution test set:**

| Confidence band | n | Accuracy |
|---|---:|---:|
| ≤ 0.50 | 46 | 0.674 |
| 0.50–0.60 | 64 | 0.797 |
| 0.60–0.70 | 94 | 0.883 |
| 0.70–0.80 | 141 | 0.957 |
| 0.80–0.90 | 210 | 0.990 |
| 0.90–0.95 | 179 | 1.000 |
| 0.95–1.00 | 201 | 1.000 |

This monotonic relationship holds **in distribution**. It does **not** survive the
robustness study (§13), where confidence stayed high while accuracy collapsed. The
scores remain uncalibrated vote shares, not probabilities of correctness.

## 13. Real-world robustness

No independent real photographs exist in this project, so a controlled degradation
study was run instead: 61 held-out test images × 10 conditions = 610 trials, with
labels inherited from the source image. It measures sensitivity to transformation,
not field performance.

| Condition | Accuracy | Mean confidence |
|---|---:|---:|
| control | 0.902 | 0.795 |
| JPEG q30 | 0.869 | 0.643 |
| rotated 15° | 0.787 | 0.713 |
| brightness × 1.45 | 0.721 | 0.516 |
| central 70 % crop | 0.607 | 0.604 |
| brightness × 0.55 | 0.574 | 0.510 |
| contrast 55 % | 0.557 | 0.510 |
| vignette | 0.475 | 0.581 |
| gaussian blur | 0.443 | 0.615 |
| downscale 96 px | 0.426 | 0.631 |

Accuracy falls by up to 48 points. Resolution and focus destroy the texture
descriptors; illumination shifts every colour descriptor; a vignette moves the
background estimate that segmentation depends on. The most confident failures are
`Old → Ripe` at 0.96–0.97 under blur and downscaling — **confidence is not a
reliability signal out of distribution.**

Full detail: `reports/phase4_real_world_robustness.md`.

## 14. Application architecture

```
website/
  app.py                     upload → preprocess → predict → history
  pages/                     dataset · model performance · about
  components/                upload, preprocessing view, results, chart
  utils/                     validation, image_processor, model_loader,
                             artefacts, history, samples, cache, paths
  styles/style.css
```

The application calls the project's own `src/` pipeline — `segment_vegetable` and
`extract_features` — rather than re-implementing it, and loads the model through a
loader that refuses to serve an artefact whose feature list or class order has
drifted. Metrics are read from `p0_comparison.json` and
`phase4_test/metrics.json`; no figure is hard-coded (three literals were removed
during the Phase 4 audit).

## 15. Inference performance

Measured on the deployment machine (i5-13450HX, 16 GB, CPU only), 50 repetitions:

| Stage | Median |
|---|---:|
| Validation | 0.01 ms |
| Decode | 0.32 ms |
| Preprocessing (resize + segmentation + 83 features) | 19.10 ms |
| Model inference | 7.54 ms |
| **End-to-end** | **27.19 ms** (mean 29.20, sd 4.99) |

Cold start (fresh process to usable model): **2.06 s**, paid once. 200 consecutive
predictions: 0 errors, +5.1 MB resident memory. A 12 MP (14.06 MB) upload is handled
in 255 ms.

## 16. Testing results

| Metric | Value |
|---|---:|
| Total tests | **178** |
| Passed | **178** |
| Failed | 0 |
| Skipped | 0 |
| New in Phase 4 | 32 |

The 32 new tests cover the test-evaluation guard (a second run must refuse to
overwrite), label encoding, split-verification failure detection, the completeness
and internal consistency of the saved test predictions, determinism of every
robustness transform, the integrity-audit result, and a regression guard pinning the
serving input size to 224.

Phase 1 integrity was re-verified after all work: **12/12 checksums unchanged**.

## 17. Limitations

1. **Single split, single model, one test pass.** No repeated holdout, no
   cross-validation of the final model, no seed variance for any configuration.
2. **Test score has a wide interval** — [0.9137, 0.9553]. Differences smaller than
   ~0.04 are not resolvable with 935 images.
3. **Frozen backbones only.** Fine-tuning was never run; it is the most plausible
   way a CNN could close the gap, and remains untested.
4. **Studio-domain only.** Robustness study shows large degradations outside it
   (§13). No field data.
5. **Uncalibrated scores.** No temperature scaling or isotonic calibration was
   performed; the confidence-vs-accuracy table is descriptive, not a guarantee.
6. **Label ambiguity is a ceiling.** The 41 ambiguous clusters bound achievable
   accuracy from above.
7. **`Damaged` is fragile** — 49 test images.
8. **One dataset family** — four pepper/tomato types from one redistribution.

## 18. Future improvements

1. **Fine-tune** the last block of ResNet50/MobileNetV2 (B1) — the single highest-
   value untested experiment.
2. **Resolve the label boundary** between `Old` and `Ripe` and re-evaluate; this may
   raise every model more than any architecture change.
3. **Multi-seed evaluation** so future comparisons are not decided by ±0.01 noise.
4. **Calibrate** the outputs and report expected calibration error.
5. **Collect 100+ field photographs** with verified labels and report performance on
   them, instead of a degradation proxy.
6. **Correct the duplicate clustering** at threshold 2 (largest component 5,
   98.3 % label-pure) and rebuild the split, if the coverage trade-off is accepted.
7. **Domain adaptation** for background and illumination, e.g. segmentation-guided
   normalisation, evaluated on the degradation suite as a target.

## Research conclusion

On this dataset, under this protocol, the three deep-learning configurations tested
— MobileNetV2 (frozen, single and multi-task) and ResNet50 (frozen) — **did not
demonstrate a statistically meaningful improvement** over a Random Forest trained on
83 handcrafted colour, texture and shape descriptors. The best network was
indistinguishable from the forest (Δ +0.0018, p = 0.25); the MobileNetV2 variants
were significantly worse (p ≤ 0.0011).

The Random Forest was selected because it offered **comparable predictive
performance at substantially lower inference cost** — roughly 15× faster end to end
and 6× smaller — and because the difference that would justify the more complex model
was not present in the data.

This is **not** a claim that deep learning is unsuitable for vegetable grading. It is
a claim about four specific configurations at 224 px on a 6,150-image, studio-style
corpus with frozen backbones. Fine-tuning, larger datasets and field imagery are all
untested here and could reverse the conclusion.

Final official test performance of the deployed system: **macro F1 0.9350, accuracy
0.9497 on 935 held-out images**, with the weakest class (`Damaged`) at 0.8636 F1.

The system does not detect internal spoilage, does not assess food safety, and its
confidence score should not be read as a probability of correctness outside the
training distribution.
