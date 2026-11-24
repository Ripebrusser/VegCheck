# Limitations

Limitations are grouped by whether they come from the **dataset**, the
**method**, or the **compute environment**. Each is stated with its consequence,
because a limitation without a consequence is just a disclaimer.

---

## 1. Dataset limitations

### L1 — The corpus is a downsampled redistribution, not the published original

The local corpus has **6,150 images, all exactly 256×256**. Mendeley Data
(10.17632/6nxnjbn9w6.1) and the *Data in Brief* article both describe **6,850**
images. The version used here is the Hugging Face redistribution
(`Project-AgML/VegNet_quality_classification`).

**Consequences:**
* 700 images relative to the published description are absent.
* Every image has already been downsampled to 256×256, so **no resolution above
  256 px can add information**. Model input is capped accordingly and a
  higher-resolution study would be measuring interpolation, not detail.
* Reported metrics are **not numerically comparable** to published work on the
  original release. This project does not claim to beat or match any published
  benchmark.

**Not mitigated deliberately.** Downloading the Mendeley original to "top up" the
corpus would mix two differently preprocessed copies of the same dataset and
introduce near-duplicate leakage across splits. A single, describable corpus was
preferred over a larger, ambiguous one.

### L2 — One (vegetable, quality) cell is empty

`Tomato/Dried` contains **0** images. Only **19 of 20** cells are populated.

**Consequences:**
* A 20-way joint classifier is impossible without inventing data.
* Quality classification remains well-posed as a 5-class problem because all five
  qualities occur for the other three vegetables (`Dried` has 1,389 images overall).
* The web application must not present "Tomato, Dried" as a plausible outcome even
  if the model's softmax assigns it mass; the interface states which combinations
  are unsupported by training data.

### L3 — Severe class imbalance, and one very small cell

Cell counts range from **27** (`Tomato/Damaged`) to **1,234** (`Tomato/Old`) — a
**45.7×** ratio. Quality-class counts range from **317** (`Damaged`) to **2,044**
(`Old`).

**Consequences:**
* Accuracy is a misleading headline metric; this project reports **macro F1**.
* Any per-class statistic for `Tomato/Damaged` rests on **27 images** in total,
  i.e. roughly 4 in the test split. Confidence intervals there are enormous and
  are reported alongside support counts, never as a bare percentage.
* Class weighting and balanced sampling are required, and are ablation-tested in
  Phase 2 rather than assumed to help.

### L4 — 35% of images sit in perceptual near-duplicate clusters

2,182 of 6,150 images belong to 572 dHash clusters at a Hamming threshold of 6.
That clustering over-merged: union-find transitive closure produced **one component
of 732 images (11.9% of the corpus) with a diameter of 39**, spanning all four
vegetables — impossible for a genuine duplicate burst.

**Corrected label-ambiguity figure: 41 clusters / 134 images (2.2%)**, not the
42 clusters / 866 images originally reported. 732 of those 866 images were the
single chained artefact, not annotation noise.

**Consequences:**
* Any split that ignores cluster structure inflates test scores. The measured
  inflation on handcrafted features is **+0.025 macro F1** (0.9446 random CV vs
  0.9199 grouped CV).
* The 41 genuinely conflicting clusters mean the ceiling is below 100%: some
  images are genuinely ambiguous or mislabelled. Error analysis in Phase 2 must not
  present these as pure model failures.
* Over-grouping pushes a 732-image block entirely into train, so validation and
  test sample a narrower slice of the corpus than a correctly-clustered split
  would. The split is retained unchanged (risk R1), and this is a limitation of
  the *evaluation*, not of the leakage protection.
* **Residual risk:** dHash clustering with a Hamming threshold of 6 catches
  near-identical frames but not all same-session correlation. Two photographs of
  the same specimen from different angles may fall outside the threshold and land
  in different splits. Grouped CV numbers are therefore still mildly optimistic.
  This is the main reason the 0.92 handcrafted-feature result should be read as an
  upper bound rather than an expected field performance.

### L5 — Studio conditions, narrow capture distribution

Brightness is tightly controlled (mean 178.5, σ 12.5) and every image is a
single vegetable centred on a uniform, near-white background.

**Consequences:**
* Models trained here will not transfer to cluttered field photographs, mixed
  lighting, or multiple vegetables per frame. No such capability is claimed.
* Segmentation is easy on this corpus (99.5% usable masks) precisely because the
  background is trivial. That success does **not** generalise to real scenes.

### L6 — Labels conflate different phenomena under one word

"Quality" here mixes **maturity** (`Unripe`, `Ripe`), **senescence** (`Old`),
**desiccation** (`Dried`) and **physical injury** (`Damaged`). These are not points
on one scale.

**Consequence:** the five classes are treated as five distinct categories, never
as an ordinal freshness ranking. The project does not output "freshness 7/10".
`Old` and `Dried` overlap semantically and are the most likely confusion pair;
Phase 2 must quantify that rather than assume it away.

### L7 — No ground truth beyond the label

There is no independent measurement of firmness, moisture, sugar content or
shelf life.

**Consequence:** the system predicts the **assigned visual label**. It cannot
verify that label, and no claim is made that a predicted class corresponds to any
measurable physical property.

---

## 2. Methodological limitations

### L8 — External photographs only

The dataset contains only **external** images.

**Consequence:** this project detects **visible surface condition**. It explicitly
does **not** detect internal rot, internal deterioration, pest damage beneath the
skin, or chemical contamination, and must never be described as doing so. Doing so
would require imaging modalities (X-ray, hyperspectral, NIR) and evidence that this
project does not have.

### L9 — No food-safety or edibility claim

A predicted class is a description of appearance, not a safety judgement.

**Consequence:** the website states that predictions are informational, that
confidence is not a guarantee of correctness, and that the tool must not be used
to decide whether food is safe to eat. Misclassification of a spoiled item as
`Ripe` is a realistic failure mode and is called out in the UI.

### L10 — Handcrafted-feature baseline is a strong confound

Colour descriptors alone reach 0.897 grouped macro-F1; all 83 reach 0.920.

**Consequence:** a deep model that reports, say, 0.93 has demonstrated almost
nothing. Phase 2 must report the CNN-vs-features comparison explicitly, and a CNN
that fails to beat 0.920 will be reported as such rather than quietly omitted.

### L11 — Preprocessing "improvement" is measured on proxy metrics

The operation study measures noise σ, Laplacian variance, LAB-L contrast, entropy
and SSIM. These are proxies for image quality, not for task performance.

**Consequence:** the study can say bilateral filtering preserves detail better than
Gaussian blur, but it cannot say which preprocessing improves **classification**.
That question is deferred to the Phase 2 ablation and stated as open.

The noise estimator is additionally blind to the difference between amplified
noise and genuine high-frequency detail, which is why CLAHE's +160% reading is not
interpreted as harm.

---

## 3. Environment limitations

### L12 — CPU-only training

TensorFlow has no native Windows GPU support after 2.10. The available RTX 4050 is
therefore unused, and all training is CPU-bound.

**Consequences:**
* Experiment budget is limited; the Phase 2 matrix must be sized to measured
  throughput rather than ambition.
* Epoch counts, image size (224 px) and fine-tuning depth are constrained by
  wall-clock time, and this will be stated with the results rather than presented
  as a tuned optimum.
* Inference latency measured on this CPU will be slower than a GPU deployment;
  reported latencies are labelled with the hardware they were measured on.

### L13 — Storage budget

The dataset is small (125.5 MiB), duplicated once as a verified working copy, and
materialised once more as split directories. Total project storage stays far below
the 10 GB budget. This is a limitation only in the sense that the corpus is small:
6,150 images is modest for deep learning, which is another reason transfer learning
rather than from-scratch training is the sensible default.

---

## 4. Corrections recorded during Phase 2 preflight

**Correction to L4 (near-duplicate clusters).** The original text described the 572
clusters as "bursts of photographs of the same physical specimen" and quoted
"42 groups / 866 images carrying conflicting labels".

That framing was wrong in one important respect. `src/data/cluster_diagnostics.py`
showed that union-find transitive closure chained unrelated images together: at the
threshold used, **one component holds 732 images (11.9% of the corpus) spanning all
four vegetables, with a diameter of 39 against a threshold of 6.** A genuine
duplicate burst cannot exceed its own threshold by that margin.

Corrected figures:

* **732 of the 866** "label-conflicting" images are that single chained component —
  an artefact of over-merging, not annotation noise.
* The genuine label ambiguity is **41 clusters / 134 images (2.2% of the corpus)**,
  concentrated on the semantically adjacent pairs `Dried↔Ripe`, `Dried↔Old`,
  `Old↔Ripe`.

The clustering remains conservative for leakage (over-grouping cannot inflate test
scores), but it means the training split contains a rigid 732-image block that
validation and test do not sample. This is recorded as risk R1 in
`reports/phase2_preflight.md`; the threshold sweep and a proposed threshold-2
alternative are in `evaluation/reports/cluster_diagnostics.json`.

**Unchanged:** the leakage claims themselves. Zero duplicate groups and zero
SHA-256 values cross a split, which was re-verified during preflight.
