# Results — Phase 1

Every number below was produced by executing the code in this repository on
2026-11-14. No value is estimated, projected or copied from a paper.

Reproduce with:

```bash
.venv/Scripts/python.exe -m src.data.audit
.venv/Scripts/python.exe -m src.data.verify_raw_copy
.venv/Scripts/python.exe -m src.data.splits
.venv/Scripts/python.exe -m src.features.screening
.venv/Scripts/python.exe -m src.preprocessing.evaluate
.venv/Scripts/python.exe -m pytest tests/ -q
```

---

## 1. Dataset inventory

| Property | Value |
|---|---|
| Images | **6,150** |
| Total size | **131,638,293 bytes (125.5 MiB)** |
| Mean file size | 21,404 bytes |
| Format | JPEG, RGB, 3-channel |
| Dimensions | **256×256 for all 6,150 files** (single distinct dimension) |
| Unreadable / corrupt | **0** |
| Byte-identical duplicates | **0** |

### Per vegetable

| Vegetable | Images | Share |
|---|---:|---:|
| Bell Pepper | 1,176 | 19.1% |
| Chile Pepper | 1,286 | 20.9% |
| New Mexico Green Chile | 1,327 | 21.6% |
| Tomato | 2,361 | 38.4% |

### Per quality class

| Quality | Images | Share |
|---|---:|---:|
| Damaged | 317 | 5.2% |
| Dried | 1,389 | 22.6% |
| Old | 2,044 | 33.2% |
| Ripe | 1,787 | 29.1% |
| Unripe | 613 | 10.0% |

### Joint (vegetable × quality) cells

| | Damaged | Dried | Old | Ripe | Unripe |
|---|---:|---:|---:|---:|---:|
| Bell Pepper | 31 | 296 | 349 | 448 | 52 |
| Chile Pepper | 121 | 593 | 200 | 183 | 189 |
| New Mexico Green Chile | 138 | 500 | 261 | 201 | 227 |
| Tomato | 27 | **0** | 1,234 | 955 | 145 |

* Populated cells: **19 of 20**.
* Largest cell: `Tomato/Old`, **1,234**.
* Smallest non-empty cell: `Tomato/Damaged`, **27**.
* Imbalance (largest ÷ smallest non-empty): **45.7×**.

## 2. Duplicates

| Kind | Groups | Images |
|---|---:|---:|
| Exact (SHA-256) | 0 | 0 |
| Near-duplicate (dHash, Hamming ≤ 6) | **572** | **2,182** |
| Near-duplicate groups spanning **different labels** (raw) | **42** | **866** |

2,182 of 6,150 images (35.5%) belong to a perceptual near-duplicate cluster, so a
naive random split would place the same specimen in both train and test for a third
of the corpus. This is the single most important finding of the audit.

### Correction (Phase 2 preflight)

The "42 groups / 866 images carry conflicting labels" row above is the **raw
threshold-6 output** and must not be read as label noise. Transitive chaining
merged unrelated images: a single component holds **732 images (11.9% of the
corpus) spanning all four vegetables, with a diameter of 39 against a threshold of
6** — impossible for a genuine duplicate burst, since every member of a true burst
is within the threshold of every other.

| Kind | Groups | Images |
|---|---:|---:|
| Raw threshold-6 output | 42 | 866 |
| Chained artefact (one component) | 1 | **732** |
| **Genuine label ambiguity** | **41** | **134 (2.2%)** |

The genuine conflicts concentrate on semantically adjacent classes — `Dried↔Ripe`
(22 groups), `Dried↔Old` (21), `Old↔Ripe` (21) — which is expected, because those
three describe overlapping visible states rather than distinct conditions.

Evidence: `evaluation/reports/cluster_diagnostics.json` (threshold sweep) and
`evaluation/metrics/cluster_threshold_sweep.csv`.

The split was **not** rebuilt. Over-grouping is conservative for leakage — it
cannot inflate test scores — but the 732-image block sits entirely in train, so
validation and test sample a narrower slice of the corpus than intended. This is
recorded as risk R1 in `reports/phase2_preflight.md`.

*Frozen-artefact note:* the printed tables in
`notebooks/01_dataset_audit.ipynb` and `datasets/metadata/audit_summary.json` still
show the raw 42 / 866 figures. They are Phase 1 artefacts preserved unchanged for
provenance; this section supersedes their **interpretation**, not their existence.

## 3. Capture-condition statistics

| Metric | mean | std | p05 | median | p95 | max |
|---|---:|---:|---:|---:|---:|---:|
| Brightness (HSV V) | 178.51 | 12.47 | 156.68 | 179.24 | 197.33 | 228.36 |
| LAB-L contrast (σ) | 41.00 | 9.19 | 25.70 | 41.80 | 55.11 | 65.83 |
| Sharpness (Laplacian var) | 139.33 | 132.10 | 12.95 | 92.21 | 431.80 | 801.57 |
| Saturation (HSV S) | 56.38 | 26.03 | 18.56 | 51.76 | 100.69 | 214.00 |
| Colourfulness | 50.89 | 22.47 | 17.37 | 49.09 | 89.16 | 142.52 |

Brightness is tightly controlled (studio photography on white backgrounds), but
sharpness spans two orders of magnitude — motion blur and defocus are real
sources of variation.

## 4. Split

| Split | Images | Fraction | Requested |
|---|---:|---:|---:|
| train | 4,286 | 69.69% | 70% |
| validation | 929 | 15.11% | 15% |
| test | 935 | 15.20% | 15% |

Leakage checks:

| Check | Result |
|---|---|
| Duplicate groups in corpus | 4,540 |
| Groups spanning splits | **0** |
| SHA-256 values spanning splits | **0** |
| Splits missing a quality class | **0** |
| Splits missing a vegetable | **0** |
| Populated cells absent from train | **0** |
| Populated cells absent from test | **0** |

All five quality classes and all four vegetables appear in every split, including
the 27-image `Tomato/Damaged` cell.

## 5. Preprocessing operation study

190 training images (10 per populated cell), 224×224, single CPU.

| Operation | noise σ | noise ↓ | sharpness | sharpness kept | contrast Δ | SSIM vs raw | ms/img |
|---|---:|---:|---:|---:|---:|---:|---:|
| `identity` | 0.95 | 0.0% | 106.7 | 100.0% | +0.0% | 1.000 | 0.00 |
| `gaussian_3` | 0.72 | 23.8% | 26.0 | 24.4% | −0.9% | 0.990 | 0.19 |
| `gaussian_5` | 0.67 | 29.2% | 14.2 | 13.3% | −1.4% | 0.980 | 0.21 |
| `median_3` | 0.82 | 13.8% | 65.3 | 61.2% | −0.3% | 0.990 | 0.10 |
| `median_5` | 0.68 | 27.9% | 45.4 | 42.6% | −0.9% | 0.972 | 0.37 |
| **`bilateral`** | 0.66 | **30.9%** | 100.7 | **94.4%** | −0.3% | 0.989 | 0.52 |
| `clahe` | 2.47 | −160.4% | 276.4 | 259.0% | **+8.6%** | 0.895 | 0.95 |
| `median3+clahe` | 2.16 | −127.5% | 152.3 | 142.7% | +8.5% | 0.903 | 0.98 |
| `gaussian3+clahe` | 2.06 | −116.8% | 77.1 | 72.3% | +8.6% | 0.901 | 1.07 |

**Findings that contradicted the initial plan:**

1. **Bilateral filtering dominates the denoisers.** It removes 31% of the
   high-frequency noise while retaining 94% of the sharpness measure, at 0.52 ms
   per image — 5.4× the cost of median 3×3, but in absolute terms negligible.
2. **Gaussian blur is the worst denoiser here.** At 3×3 it retains only 24.4% of
   sharpness, versus 61.2% for the median filter at the same kernel size.
3. **CLAHE raises high-frequency energy by 160%**, of which some is genuine local
   contrast and some is amplified noise. The metric cannot separate them, so
   CLAHE's value is left to the Phase 2 ablation.

Segmentation: a plausible mask was produced for **99.5%** of the 190 sampled
images (1 failure, a `Dried` sample). Mean foreground area fraction 0.249
(p05 0.048, p95 0.550).

## 6. Handcrafted feature screening

4,286 training images × 83 features, 2,709 duplicate-safe groups.

| Feature group | Features | Grouped CV macro-F1 | Random CV macro-F1 |
|---|---:|---:|---:|
| Colour | 30 | **0.8966 ± 0.0158** | 0.9315 |
| Texture | 46 | 0.6991 ± 0.0656 | 0.7527 |
| Shape | 7 | 0.4062 ± 0.0283 | 0.5068 |
| **All** | **83** | **0.9199 ± 0.0333** | 0.9446 |

Same descriptors against **vegetable identity**, grouped CV:

| Group | macro-F1 |
|---|---:|
| Colour | 0.9124 ± 0.0148 |
| Texture | 0.8752 ± 0.0279 |
| Shape | 0.6746 ± 0.0144 |
| All | **0.9544 ± 0.0220** |

**Near-duplicate inflation** (random CV − grouped CV, all features):
**+0.0248 macro-F1**. That 2.5-point gap is the size of the leakage error the
grouped split design exists to prevent, measured rather than assumed.

**Interpretation.** This is the Phase 1 result that most affects Phase 2. A
Random Forest on 83 classical descriptors reaches **0.920 grouped macro-F1** on
quality classification. Any deep model must be judged against that bar. If a CNN
lands below 0.920, the honest conclusion is that extra capacity bought nothing on
this corpus — not that the CNN "works".

Colour alone accounts for most of the signal (0.897), consistent with the visual
inspection in notebook 02: ripeness and drying are largely chromatic phenomena in
this dataset. Shape is the weakest group, as expected for a *condition* task.

## 7. Test suite

```
73 passed in 1.53s
```

Covering: dataset presence and composition, audit consistency, zero cross-split
leakage, split reproducibility, split stratification coverage, processed-tree
integrity, colour-space contracts, filter behaviour, CLAHE, normalisation,
segmentation on synthetic images, pipeline stage contract, augmentation
determinism, and the handcrafted feature contract.

## 8. Notebooks executed

| Notebook | Cells | Embedded figures | Errors |
|---|---:|---:|---:|
| `01_dataset_audit.ipynb` | 19 | 0 | 0 |
| `02_exploratory_analysis.ipynb` | 17 | 5 | 0 |
| `03_image_preprocessing.ipynb` | 23 | 10 | 0 |

## 9. Not yet measured

Phase 2 and 3 results do not exist yet. There is no accuracy, F1, confusion matrix
or inference-time figure for any neural model in this document, because no neural
model has been trained. Any such number appearing elsewhere in the repository
before Phase 2 completes would be fabricated.
