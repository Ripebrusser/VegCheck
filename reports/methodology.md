# Methodology

**Project:** Vegetable Quality and Freshness Detection Using Image Processing and Deep Learning
**Phase covered:** 1 (data, image processing, features)
**Date:** 2026-11-14

This document describes *how* the work was done and *why* each choice was made, so
the study can be reproduced and criticised. Measured numbers live in
`reports/results.md`; the constraints that limit them live in
`reports/limitations.md`.

---

## 1. Environment

| Item | Value |
|---|---|
| OS | Windows 11 (native), Git Bash shell |
| CPU | Intel Core i5-13450HX, 16 logical cores |
| RAM | 15.7 GB |
| GPU | NVIDIA RTX 4050 Laptop, 6 GB (not used by TensorFlow — see below) |
| Python | 3.12.16, provisioned project-locally with `uv` |
| Venv | `.venv/` at the project root |
| Framework | TensorFlow 2.21.0 (CPU), OpenCV 5.0.0, scikit-image 0.26.0, scikit-learn 1.9.1 |

**Why Python 3.12 was provisioned.** The machine's default interpreters are
Python 3.14.3 and 3.15.2, for which TensorFlow publishes no wheels
(`pip index versions tensorflow-cpu` returns *No matching distribution* for both).
Rather than silently switching to a different framework, CPython 3.12.16 was
installed with `uv` and a project-local virtual environment was built from it.

**Why training runs on CPU despite a discrete GPU.** TensorFlow dropped native
Windows GPU support after 2.10; the `tensorflow[and-cuda]` extras are Linux/WSL2
only. Enabling the RTX 4050 would require a WSL2 + CUDA + cuDNN stack, which
moves the dataset to a different filesystem and adds multi-gigabyte,
administrator-level dependencies. The project therefore states plainly that
training is CPU-bound and sizes its experiments accordingly instead of implying
GPU acceleration that does not exist.

## 2. Data pipeline

### 2.1 Provenance and preservation

The corpus arrived pre-extracted. It is kept **byte-identical** in `Dataset/`
and copied to `datasets/raw/vegnet/New VegNet/` as the working copy. A SHA-256
manifest is written to `datasets/metadata/sha256_manifest.csv` and the two trees
are compared file-by-file (`src/data/verify_raw_copy.py`): 6,150 files, 0 missing,
0 extra, 0 content mismatches.

### 2.2 Audit

`src/data/audit.py` opens every image twice — Pillow `verify()` (catches truncated
JPEG payloads that decoders tolerate) and a full OpenCV decode — and records size,
dimensions, channels, mode, colour moments, brightness, LAB lightness spread,
Laplacian variance, saturation, colourfulness, clipping fractions, SHA-256 and a
64-bit dHash.

Duplicate detection runs in two passes:

1. **Exact**: group by SHA-256.
2. **Near-duplicate**: all-pairs Hamming distance over 64-bit dHash values,
   union-find clustering at threshold ≤ 6 bits.

The all-pairs computation is vectorised with a byte-plane popcount lookup. A naive
Python double loop over 6,150 images is 18.9 M comparisons (≈40 s); the vectorised
version completes in well under a second. This is documented because the naive
version is the obvious implementation and it is what made an all-pairs scan
practical.

### 2.3 Leakage-safe splitting

Near-duplicate clusters are **indivisible groups**. The splitter
(`src/data/splits.py`):

1. assigns every image to a group (cluster id, or a unique negative id);
2. stratifies on the joint (vegetable, quality) cell;
3. within each stratum, visits groups largest-first and assigns each to the split
   with the largest remaining deficit against its 70/15/15 target;
4. forces at least one group into every split so no class is missing from
   validation or test.

Randomness comes from a generator seeded with `(SEED, stratum)`, so the split is
reproducible and independent of iteration order. A test asserts that the same
seed reproduces the assignment exactly and a different seed does not.

Augmentation is **never** applied here. Only original images are split; augmented
copies exist only inside the training pipeline.

### 2.4 Preprocessing

`src/preprocessing/pipeline.py` implements each operation as an independent,
testable function: resize (squash / centre-crop / pad), Gaussian, median and
bilateral filtering, CLAHE on the LAB L channel, LAB-distance segmentation with
morphological cleanup and hole filling, and normalisation (`unit` or ImageNet).

Rather than stacking everything, operations are **measured**
(`src/preprocessing/evaluate.py`) on 190 training images across nine variants,
recording noise σ, Laplacian variance, LAB-L contrast, entropy, SSIM against the
raw image and per-image latency. Three conclusions follow from the numbers rather
than from convention, and they are stated in `evaluation/reports/preprocessing_study.md`.

**Metric limitation, stated up front:** the Laplacian-MAD noise estimator cannot
separate amplified sensor noise from legitimately added high-frequency detail, so
CLAHE raises it by design. No conclusion about CLAHE is drawn from that metric
alone; the Phase 2 ablation settles it.

### 2.5 Augmentation

Implemented in `src/preprocessing/augment.py` with OpenCV, using a **single affine
warp** for rotation + zoom + translation so the image is resampled once instead of
three times. Included: rotation ±12°, zoom ±10%, translation ±6%, brightness ±0.12,
contrast ±0.12, horizontal flip p=0.5.

Excluded deliberately: vertical flip (vegetables have a canonical orientation) and
hue rotation (ripeness is largely a hue cue here, so shifting hue would effectively
relabel the image).

### 2.6 Features

83 handcrafted descriptors in three groups:

* **Colour (30)** — RGB moments and skew, HSV statistics with hue handled
  *circularly* (mean sine/cosine rather than a degree average, which would place
  red at both 0° and 180° and average to cyan), a 12-bin hue histogram,
  Hasler–Süsstrunk colourfulness, excess-green and excess-red indices.
* **Texture (46)** — GLCM Haralick descriptors (contrast, dissimilarity,
  homogeneity, energy, correlation, ASM) at three distances, plus uniform-LBP
  histograms at two radii.
* **Shape (7)** — area, normalised perimeter, circularity, aspect ratio, solidity,
  extent, equivalent diameter from the segmented foreground.

Texture is measured on the **mask bounding box**, not by filling the background
with a constant, which would manufacture an artificial edge and inflate GLCM
contrast.

## 3. Feature screening design

The screening question is not "which features have high F-scores" but "how far do
these descriptors get on their own". A 5-fold cross-validated Random Forest is
fitted per feature group, under **two** schemes:

* **random stratified K-fold** — the obvious approach, which lets near-duplicate
  clusters straddle the fold boundary and therefore reports an optimistic score;
* **stratified group K-fold** using the duplicate-cluster ids — the honest
  estimate.

Both are reported so the size of the near-duplicate effect is visible rather than
hidden. Only the training split is used, so feature selection cannot leak.

## 4. Verification

`pytest` covers dataset integrity, audit consistency, split leakage, split
reproducibility, the geometry and colour-space contracts of every preprocessing
operation, CLAHE behaviour, normalisation, segmentation on synthetic images,
augmentation determinism, and the feature extraction contract. The full suite is
73 tests and runs in under two seconds because the data tests read manifests
rather than re-scanning the corpus.

Notebooks are generated from a builder (`tools/build_notebooks.py`) and executed
with `nbconvert`, so every stored output — including all embedded figures — is a
real execution result, not a screenshot.

## 5. What was *not* done, and why

* No model training. Phase 1 ends at the gate; Phase 2 is blocked on approval.
* No dataset substitution. The proposed dataset was used as-is once its
  limitations were documented.
* No attempt to reconstruct missing images. The 700-image gap and the empty
  `Tomato/Dried` cell are recorded as dataset properties, not repaired.
