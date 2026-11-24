# Demonstration Checklist

Everything below was executed against the running application; the screenshots in
`reports/screenshots/phase4/` are captures of that session, not mock-ups.

---

## 0. Before the demonstration

```bash
# from the project root
cd website
../.venv/Scripts/python.exe -m streamlit run app.py
```

Open <http://localhost:8501>. First load takes ~2 s while the model loads once;
after that every interaction is served from the Streamlit cache.

**Sanity check:** the header should read
`Deployed model · Random Forest (83 handcrafted features) · validation macro-F1 0.9539`.

If the model is missing, the app still opens and shows a red banner explaining how to
regenerate it (`python -m src.training.train --run-id A5`). That failure mode is
worth demonstrating rather than hiding.

---

## 1. Landing page

**Screenshot:** `screenshots/phase4/01_home.png`

- Title, description, three badges: deployed model + validation macro-F1, the
  224×224 input size, five quality classes.
- Expand **"What this system does — and does not do"** — this is the first thing to
  read aloud: five visual categories, no internal inspection, no safety verdict,
  and no vegetable prediction.

## 2. Upload an image

Two paths, both real:

* **Upload my own image** — JPG/JPEG/PNG up to 15 MB, minimum 64×64.
* **Use a sample validation image** — one held-out photograph per class, with its
  true label shown to you but not to the model.

Demonstrate rejection first (it is more convincing than a success):

| Try | Expected |
|---|---|
| a `.txt` renamed to `.png` | "The file content is not a readable JPEG or PNG image…" |
| a 16×16 image | "Image is 16×16; the minimum size is 64×64." |
| a 20 MB file | rejected by the 15 MB limit shown on the uploader |

## 3. Preprocessing visualisation

**Screenshot:** `screenshots/phase4/02_prediction.png` (upper half)

Three panels — original, 224×224 resize, segmented foreground — plus segmentation
status, foreground fraction and mask pixel count. Open **Feature vector (83
descriptors)** to show the exact values handed to the Random Forest.

Say explicitly: *these are the only transformations applied; CLAHE, blur and
ImageNet normalisation belong to experiments that were not deployed.*

## 4. Prediction

**Screenshot:** `screenshots/phase4/02b_result_detail.png` (result + chart)

Click **Predict quality**. The result card shows:

* predicted class,
* top score,
* inference time (~8 ms model, ~27 ms end-to-end),
* runner-up and its score,
* the caveat that vote shares are not calibrated probabilities.

The explanation under the chart states that this is a visual classification, not a
food-safety assessment, and that *Ripe*/*Unripe* describe maturity.

## 5. Confidence chart

Bar chart of all five classes, predicted class highlighted in green, percentages on
every bar. Point out that `Damaged` and `Unripe` are the classes the model is least
sure about — consistent with the test per-class table.

## 6. Prediction history

**Screenshot:** `screenshots/phase4/02_prediction.png` (full page, lower half)

Each prediction appends a row: time, filename, quality, confidence, latency. Click
**Clear history** — it empties immediately and nothing was ever written to disk.

## 7. Dataset information page

**Screenshot:** `screenshots/phase4/03_dataset.png`

6,150 images, 4 vegetables, 5 quality classes, 45.7× cell imbalance; split table; the
19-of-20 populated matrix with `Tomato/Dried` visibly empty; and the six documented
limitations including the corrected duplicate figures (41 clusters / 134 images).

## 8. Model performance page

**Screenshot:** `screenshots/phase4/04_model_performance.png`

Deployed model card (macro F1 0.9539 validation, 95 % CI), the four-model comparison
table, per-class validation results, the confusion matrix image, and the
three-row table that separates grouped-CV / holdout / test. Then the test result:
**macro F1 0.9350 on 935 images**.

## 9. Status: limitations

State plainly:

1. 0.9350 is the **test** score; 0.9539 was validation, used for selection.
2. The test set was scored **once**, after the model was frozen.
3. Accuracy drops to 0.43–0.79 under blur, low resolution, dim light, or changed
   background — the robustness study.
4. Confidence is informative in-distribution, unreliable out of it.
5. No food-safety, spoilage or nutrition claims are made.

## 10. Final test results (to quote)

| Metric | Value |
|---|---:|
| Test macro F1 (primary) | **0.9350** |
| 95 % CI | [0.9137, 0.9553] |
| Test accuracy | 0.9497 |
| Weighted F1 | 0.9492 |
| Images | 935 |
| Weakest class | Damaged, F1 0.8636 (support 49) |

## 11. If asked to prove it is not hard-coded

```bash
# every number on screen comes from these files
cat evaluation/phase4_test/metrics.json
cat evaluation/metrics/p0_comparison.json

# the full suite
.venv/Scripts/python.exe -m pytest tests/ -q      # 178 passed

# integrity of the frozen Phase 1 artefacts
.venv/Scripts/python.exe -m src.evaluation.phase4_audit   # all checks pass
```

## Timing plan (10 minutes)

| Minutes | Content |
|---|---|
| 0–1 | Landing page, scope panel |
| 1–2 | Upload rejection cases |
| 2–4 | Preprocessing panels + feature inspector |
| 4–5 | Prediction, chart, explanation |
| 5–6 | History, clear, dataset page |
| 6–8 | Model performance, validation vs test |
| 8–10 | Robustness result, limitations, test metrics |
