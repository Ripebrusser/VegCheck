# Viva Preparation — 24 Questions with Evidence-Based Answers

Every figure below is from a saved artefact in this repository. Where a question has
a common wrong answer, it is noted.

---

**1. Why was a Random Forest selected over the deep models?**

It had the highest macro F1 on the shared validation split (0.9539 vs ResNet50's
0.9521) and was statistically indistinguishable from it: paired bootstrap
Δ = +0.0018, 95 % CI [−0.0207, +0.0233], McNemar p = 0.253. With accuracy unable to
separate them, cost decided it: 19.7 ms versus 302.8 ms end-to-end, 15.7 MB versus
94 MB, no TensorFlow at serving time, 0.65 s to retrain. Recorded as D-012.

*Wrong answer:* "because it is simpler" without the statistical test, or "because
deep learning failed" — it did not fail, it tied.

**2. Why is macro F1 the primary metric instead of accuracy?**

The corpus is imbalanced 6.4× between quality classes (2,044 `Old` vs 317 `Damaged`)
and 45.7× between populated cells. A model that never predicts `Damaged` still scores
high accuracy while being useless for the class a user most cares about. Macro F1
weights every class equally, so `Damaged` contributes a fifth of the score.

**3. What handcrafted features were extracted, and why those?**

83 descriptors in three families:

* **Colour (30)** — RGB moments, circular hue statistics (mean sine/cosine, not mean
  degrees), saturation/value statistics, 12-bin hue histogram, colourfulness,
  excess-green and excess-red. These capture ripeness and damage colouration.
* **Texture (39)** — GLCM Haralick properties at three distances and four angles, and
  uniform-LBP histograms at two scales. These capture shrivelling and surface damage.
* **Shape (7)** — area fraction, normalised perimeter, circularity, aspect ratio,
  solidity, extent, normalised equivalent diameter. These capture deformation.

**4. Why not feed raw pixels to the Random Forest?**

83 handcrafted values carry the discriminative structure of the image. Raw 224×224×3
= 150,528 dimensions with a forest would overfit far worse and be far slower. The
feature design is the "image processing" half of the project.

**5. Why did ResNet50 not outperform the Random Forest?**

Because with a frozen backbone it is a **linear probe on a generic ImageNet
representation**. ResNet50's features were trained to separate 1,000 object classes,
not to distinguish "old" from "ripe" produce. The handcrafted descriptors were
designed for exactly the cues this task needs — colour and texture — so a strong
classifier on top of them is competitive. Fine-tuning was never run and is the most
plausible way the gap could change.

**6. Why did MobileNetV2 perform worse than ResNet50?**

MobileNetV2 trades representation quality for efficiency: depthwise-separable
convolutions and 2.26 M parameters versus ResNet50's 23.6 M. On a frozen probe it
produced the weakest features of the four (A2 0.9376; A4 0.9401). Efficiency that
this deployment does not need is not a benefit.

**7. What did multi-task learning contribute?**

Almost nothing. The vegetable head reached 0.9974 macro F1 — effectively solved — so
it provided no useful auxiliary signal to the quality head. A4 (0.9401) beat A2
(0.9376) by +0.0025, which is not significant (p = 0.23). Both heads were correct on
93.3 % of validation images. The idea was tested and rejected on evidence, not on
principle.

**8. What does the confidence score mean?**

The Random Forest's vote share: the fraction of 200 trees voting for that class.
**It is not a calibrated probability** — no temperature scaling or isotonic
regression was applied. On the in-distribution test set it tracks accuracy well
(0.674 accuracy at ≤0.50 confidence, 1.000 at ≥0.90). Under image degradation it
does not: blur produced 0.615 mean confidence at 0.443 accuracy.

*Wrong answer:* "80 % means it is right 80 % of the time."

**9. How was data leakage prevented?**

Three mechanisms: (a) perceptual near-duplicate clusters from dHash were computed
first and treated as indivisible — every image in a cluster goes to one split;
(b) stratification was applied on the joint (vegetable, quality) cell; (c) the split
was verified: 0 group ids and 0 SHA-256 values span splits, and 0 test groups or
hashes appear in train or validation. Verified again immediately before the test
evaluation.

**10. Why was the test set not used earlier?**

Because using it for early stopping, checkpoint choice or model comparison turns it
into a second validation set and inflates the final number. Selection used validation
only; the test split was scored exactly once, in Phase 4, with a frozen artefact, and
the result was not fed back into any decision.

**11. How do you know the test evaluation was not influenced by the model?**

The artefact is byte-identical to the Phase 2 output
(`sha256 53db2b14…dd83d3`, 15,728,010 bytes). No retraining, no threshold change, no
hyperparameter change occurred between Phase 2 and Phase 4. The evaluator refuses to
overwrite an existing result without an explicit flag, and the run log records its
configuration and hash.

**12. What is the model's biggest limitation?**

Domain sensitivity. On studio-style held-out images it reaches 0.9350 test macro F1.
Under controlled degradation it drops to 0.43–0.79 for blur, low resolution, dim
light or a vignette. Segmentation depends on a plain background, and the texture
descriptors depend on resolution and focus.

**13. Why does the website not predict vegetable type?**

Because the deployed model cannot. A4 predicts vegetable at 0.9974 F1, but A4 was not
selected — it is significantly worse at the primary task (p = 0.0011). Displaying a
vegetable prediction from a quality-only model would be fabrication, which the brief
forbids.

**14. How does inference actually work, end to end?**

Upload → validate (extension, magic bytes, size, decode, dimensions) → decode to RGB
→ resize to 224×224 → LAB-distance segmentation → 83 features → NaN scrub →
`StandardScaler` → `RandomForestClassifier.predict_proba` → validate the output
(shape, finiteness, range, sums to 1) → map indices through the artefact's class list
→ render. Median 27.2 ms, of which 19.1 ms is preprocessing.

**15. How is the preprocessing pipeline reused rather than reimplemented?**

`website/utils/image_processor.py` calls `src.preprocessing.pipeline.segment_vegetable`
and `src.features.handcrafted.extract_features` — the same functions that produced
the training matrix. The Phase 4 audit proves equivalence: 20 images, 0 mismatches,
max absolute difference **0.0**.

**16. What did the near-duplicate investigation reveal?**

dHash at threshold 6 with union-find produced 42 "conflicting-label" groups covering
866 images. Investigation showed transitive chaining: one component held **732 images
with a diameter of 39** against a threshold of 6 — impossible for a genuine duplicate
burst. Genuine ambiguity is **41 clusters / 134 images (2.2 %)**. The split was kept
unchanged (over-grouping cannot inflate scores), and the limitation is documented
everywhere with the correction.

**17. What was the critical defect found in Phase 4?**

The Phase 3 application extracted features at 256 px while training used 224 px
(`config.IMAGE_SIZE`). Feature vectors differed from training by up to 0.908, with
1,562 of 1,660 values mismatched. It was found by the integrity audit, fixed in
`website/utils/paths.py`, and re-verified to 0.0 difference. The model itself was
never affected; the test evaluation uses the training extraction path.

**18. Why is the test score lower than validation?**

0.9350 versus 0.9539 — a gap of 0.0189. The 95 % intervals ([0.9137, 0.9553] and
[0.9345, 0.9701]) overlap substantially, and a single 935-image draw has a standard
error of roughly 0.01. The gap is consistent with sampling variation. It was not
"fixed" by re-running, re-splitting or tuning.

**19. How do you handle the missing `Tomato/Dried` class?**

It is absent from the corpus (19 of 20 cells populated), so no model can predict it.
It is reported as absent rather than presented as a zero score, no metric is claimed
for it, and the dataset page shows the empty cell explicitly.

**20. What does the per-class table say that the aggregate hides?**

`Damaged` has precision 0.9744 but recall 0.7755 on test — the model rarely raises a
false alarm but misses 11 of 49 damaged specimens. Aggregate accuracy of 0.9497 hides
this entirely, because `Damaged` is 5 % of the test set.

**21. Where do the numbers on the website come from?**

`evaluation/metrics/p0_comparison.json` for the four-model comparison and latency,
`evaluation/phase4_test/metrics.json` for test results,
`datasets/metadata/*.json` and `splits.csv` for dataset facts. The Phase 4 audit
found three hard-coded literals in prose and removed them; a regression check now
scans for metric literals in the app source.

**22. What is the measured inference performance, and how was it measured?**

50 repetitions per stage on an i5-13450HX, CPU only: validation 0.01 ms, decode
0.32 ms, preprocessing 19.10 ms, model 7.54 ms, end-to-end **27.19 ms median**
(sd 4.99). Cold start 2.06 s, paid once. 200 consecutive predictions: 0 errors,
+5.1 MB RSS. A 12 MP upload takes 255 ms.

**23. What would you do next, with more time?**

Fine-tune the last block of ResNet50 (untested and the most likely lever), resolve
the `Old`/`Ripe` label boundary, run multiple seeds so comparisons are not decided by
±0.01 noise, calibrate the outputs, and collect 100+ field photographs with verified
labels instead of using a degradation proxy.

**24. What does this project actually contribute?**

1. A leakage-controlled pipeline that quantifies the near-duplicate effect
   (+0.025 macro F1 inflation for a naive split) and corrects a documented
   duplicate-clustering error.
2. A like-for-like comparison of classical and deep approaches on one split with
   paired statistics — where deep learning did **not** win.
3. A deployed, tested application whose metrics are read from artefacts, whose
   failure modes are measured, and whose limitations are stated rather than hidden.
4. A reproducible record of one critical serving defect (224 vs 256 px) and its
   correction — evidence that the validation process works.

---

## Numbers to have memorised

| Fact | Value |
|---|---|
| Corpus | 6,150 images, 19/20 cells, 4 vegetables, 5 quality classes |
| Split | 4,286 / 929 / 935 |
| Validation macro F1 | A5 0.9539, A3 0.9521, A4 0.9401, A2 0.9376 |
| A5 vs A3 | Δ +0.0018, CI [−0.0207, +0.0233], p = 0.253 |
| **Test macro F1** | **0.9350**, CI [0.9137, 0.9553] |
| Test accuracy / weighted F1 | 0.9497 / 0.9492 |
| Test errors | 47 of 935 |
| Weakest class | Damaged, F1 0.8636 (support 49) |
| End-to-end latency | 27.19 ms median (19.7 ms in a lighter session) |
| Cold start | 2.06 s |
| Tests | 178 passed |
| Model hash | 53db2b148c0c1e3b9722e610c2c47c9dcc758859049b38ff434c04d10cdd83d3 |
| Genuine label ambiguity | 41 clusters / 134 images (2.2 %) |
