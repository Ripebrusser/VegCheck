# Phase 4 — Real-World Robustness

**Date:** 2026-11-15 · **Record:** `evaluation/phase4_robustness/summary.json`,
`trials.csv`, `manifest.csv`

---

## 1. What was and was not possible

The brief asks for 20–30 ordinary photographs covering varied lighting, backgrounds,
angles, distances and resolutions.

**The repository contains no such photographs.** The dataset is studio imagery: a
plain background, even illumination, one subject, 256×256. Nothing in this project
can supply kitchen or market photographs without inventing them, and labelling them
would be fabrication.

Rather than claim a real-world evaluation that was never performed, this stage runs
a **controlled degradation study**: 61 held-out test images are transformed in memory
to approximate the conditions an ordinary photograph introduces. Because each
derived image inherits a verified dataset label, the results are quantitative — but
they measure *sensitivity to transformations*, not performance on unseen scenes.

| Property | Value |
|---|---|
| Base images | 61, stratified random from the test split (seed 42) |
| Conditions | 10 (control + 9 transformations) |
| Trials | 610 |
| Labels | inherited from the unmodified source image |
| Real photographs | **none** |

Every row of `manifest.csv` records `real_photograph = no`,
`label_provenance = "dataset label of the unmodified source image"` and
`suitable_for_quantitative_evaluation = "yes (label inherited)"`.

**A successful prediction on an unlabelled photograph is not evidence of accuracy,
and none is reported here as such.**

## 2. Results

| Condition | Accuracy | Mean confidence | Correct when confident ≥ 0.7 |
|---|---:|---:|---:|
| control (unmodified) | **0.902** | 0.795 | 0.979 (n = 47) |
| JPEG quality 30 | 0.869 | 0.643 | 1.000 (n = 46) |
| rotated 15° | 0.787 | 0.713 | 0.971 (n = 34) |
| brightness × 1.45 | 0.721 | 0.516 | 1.000 (n = 22) |
| central 70 % crop | 0.607 | 0.604 | 0.882 (n = 34) |
| brightness × 0.55 | 0.574 | 0.510 | 1.000 (n = 17) |
| contrast compressed to 55 % | 0.557 | 0.510 | 1.000 (n = 15) |
| radial vignette | 0.475 | 0.581 | 0.769 (n = 26) |
| gaussian blur (σ 1.4) | 0.443 | 0.615 | 0.632 (n = 38) |
| downscale 96 px → upscale | 0.426 | 0.631 | 0.650 (n = 40) |

Overall across all 610 trials: **0.636** accuracy, mean confidence 0.635.

The control at 0.902 versus the official test score of 0.9497 is a 61-image sample;
its standard error is ≈ 0.04, so the gap is within sampling noise, not a
discrepancy. It confirms the sample is representative rather than cherry-picked.

## 3. What the numbers mean

**The pipeline is sensitive to photographic conditions.** Accuracy falls by
35–48 points between the studio control and the harshest degradations. Three
distinct causes are visible in the data:

1. **Resolution and focus destroy texture.** GLCM and LBP measure fine structure;
   downscaling to 96 px or blurring with σ 1.4 removes the very signal they encode.
   Accuracy 0.43–0.44 is barely above the 0.20 majority-class-free baseline.
2. **Illumination changes every colour descriptor.** Dimming or compressing contrast
   shifts the RGB/HSV moments and hue histogram that carry most of the model's
   discriminative power.
3. **Background and framing break segmentation.** A vignette darkens the border,
   which is exactly where `segment_vegetable` estimates the background colour from.
   Its output changes, and the seven shape features move with it.

**Confidence does not track correctness.** Under blur the model is 0.615 confident at
0.443 accuracy; under downscaling, 0.631 confident at 0.426. The Random Forest's vote
share reflects how similar a feature vector is to the training distribution, not how
likely the label is to be right.

## 4. Failure cases

222 of 610 trials were wrong. The highest-confidence failures are the informative
ones:

| Image | Condition | True | Predicted | Confidence |
|---|---|---|---:|---:|
| IMG_20220601_094922.jpg | downscaled | Old | Ripe | 0.970 |
| IMG_20220528_092117.jpg | downscaled | Old | Ripe | 0.960 |
| IMG_20220528_092117.jpg | blurred | Old | Ripe | 0.960 |
| IMG_20220601_094836.jpg | downscaled | Old | Ripe | 0.960 |
| IMG_20220601_094922.jpg | blurred | Old | Ripe | 0.955 |

Every one of the top failures is `Old → Ripe`. This is the same confusion that
dominated the validation and test confusion matrices and coincides with the 41
genuinely ambiguous duplicate clusters: the boundary between "old" and "ripe"
produce is not sharp, and degrading the image pushes borderline specimens across it
with high confidence.

## 5. Invalid input handling

| Input | Behaviour |
|---|---|
| Not an image | rejected at the magic-byte check with a readable message |
| Truncated JPEG | rejected by the decoder |
| 16 × 16 image | rejected (below the 64 px minimum) |
| 15 MB + 10 bytes | rejected with the size in the message |
| 12 MP photograph | accepted, 255 ms end-to-end (see performance audit) |
| Flat/featureless image | segmentation falls back, features are still produced, prediction returned |

## 6. Threats to validity

* Transformations are synthetic and derived from the same distribution as training
  data; real photographs differ in more ways than these nine operations capture
  (backgrounds that are not plain, multiple objects, occlusion, motion blur, mirror
  highlights, unusual colour temperature).
* 61 base images gives wide confidence intervals; a difference of 0.05 between
  adjacent conditions is not resolvable.
* Accuracy on a transformed image is not an estimate of field performance. It is an
  upper bound under the specific transformation applied, because the label is
  inherited from a studio photograph of the same specimen.

## 7. Recommendation

Before any claim of field usability, collect 100+ genuine photographs with verified
labels and re-run this analysis on them. Until then the honest statement is:

> The system performs well on the dataset it was trained for and degrades sharply
> under conditions that change resolution, lighting or background. Its confidence
> score should not be used as a reliability signal.
