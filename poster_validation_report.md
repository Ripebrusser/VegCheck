# Poster Validation Report — VegCheck

**Poster:** *Vegetable Quality and Freshness Detection Using Image Processing and Machine Learning*
**Files audited:** `IEEE_Conference_Poster.pptx`, `IEEE_Conference_Poster.pdf`, `poster_preview.png`
**Audit date:** 17 November 2026
**Repository state:** frozen results from `evaluation/phase4_test/`, `evaluation/metrics/p0_comparison.json`,
`evaluation/phase4_robustness/`, `datasets/metadata/splits.csv`

---

## 1. Dimensions and format

| Property | Required | Measured | Status |
|---|---|---|---|
| Slide width | A0 landscape | 46.811 in = **1189 mm** | pass |
| Slide height | A0 landscape | 33.110 in = **841 mm** | pass |
| PDF MediaBox | 1189 × 841 mm | 3370.3 × 2383.9 pt = **1189 × 841 mm** | pass |
| PDF pages | 1 | 1 | pass |
| PDF orientation | landscape | landscape | pass |
| PNG preview | whole poster | 3600 × 2546 px | pass |
| Layout | three columns | 3 columns, 14.9 in wide, 0.6 in margins | pass |
| Background | white / very light | `#FBFCFA` off-white, `#0E4A34` dark-green bands | pass |

No conference poster template or specification exists in the repository, so the A0 landscape default
from the brief was used. If a conference later supplies a template, only `W`/`H` in
`poster_source/build_poster.py` and the column geometry need to change.

---

## 2. Typography

Single family throughout (Segoe UI), one hierarchy, sizes chosen for 1–2 m viewing at print scale.

| Element | Size | Notes |
|---|---|---|
| Poster title | 50 pt | 2 lines, full width |
| Section headers | 28 pt | white on dark-green bands |
| Key-fact values (header pills, stat cards) | 26 pt | bold |
| Sub-headings (author block, "presented at") | 22–25 pt | |
| Abstract, introduction, objectives body | 20 pt | |
| Section body bullets | 17–18 pt | |
| Tables | 16 pt | icon-free, native PPTX tables |
| Figure captions | 15–16 pt | italic, muted |
| References | 14.5 pt | IEEE numeric style |
| Header pill labels, footer note | 13.5 pt | fine print |
| "PRESENTED AT" label | 12 pt | only sub-13.5 pt text on the poster |

Lowest content size on the poster is **13.5 pt** (footer provenance note); the single 12 pt run is the
two-word decorative label above the conference placeholder. Body is 17–20 pt, i.e. above the usual
"readable at 1.5–2 m" floor for A0.

Charts are drawn at their **final physical size** (`figsize` in inches = placed size) with real point
sizes, so nothing is down-scaled and label text prints at the size specified above.

---

## 3. Reference verification status

All eight references were checked against a publisher, DOI or dataset-platform record during
preparation. Numeric IEEE style is used, with DOIs where the poster format has room.

| # | Reference | Verified | Notes |
|---|---|---|---|
| [1] | Suryawanshi, Patil, Chumchu, "VegNet: Dataset of vegetable quality images for machine learning applications," *Data in Brief*, vol. 45, art. 108657, 2022 | Authors, journal, volume, article number, year, DOI `10.1016/j.dib.2022.108657` | Reports **6,850** images — the poster states this explicitly and separates it from the 6,150 used here |
| [2] | Project-AgML, "VegNet_quality_classification" [Data set], Hugging Face | URL resolves; card shows 6,150 rows, 256 × 256 images, 4 vegetable classes | The distribution actually used |
| [3] | Breiman, "Random forests," *Machine Learning*, vol. 45, no. 1, pp. 5–32, 2001 | DOI `10.1023/A:1010933404324` | |
| [4] | Haralick, Shanmugam, Dinstein, "Textural features for image classification," *IEEE Trans. SMC*, SMC-3(6), 610–621, 1973 | Volume, issue, pages, year | DOI `10.1109/TSMC.1973.4309314` omitted for space |
| [5] | Ojala, Pietikäinen, Mäenpää, "Multiresolution gray-scale and rotation invariant texture classification with LBP," *IEEE TPAMI*, 24(7), 971–987, 2002 | Volume, issue, pages, year | DOI `10.1109/TPAMI.2002.1017623` omitted for space |
| [6] | Otsu, "A threshold selection method from gray-level histograms," *IEEE Trans. SMC*, 9(1), 62–66, 1979 | Volume, issue, pages, year | DOI `10.1109/TSMC.1979.4310076` omitted for space |
| [7] | He, Zhang, Ren, Sun, "Deep residual learning for image recognition," *CVPR*, 2016, 770–778 | Venue, year, pages | |
| [8] | Sandler, Howard, Zhu, Zhmoginov, Chen, "MobileNetV2: Inverted residuals and linear bottlenecks," *CVPR*, 2018, 4510–4520 | Venue, year, pages | |

Dropped for space (verified but not cited): the Mendeley deposit `10.17632/6nxnjbn9w6.1` and
Zuiderveld's CLAHE chapter — the latter describes a technique the deployed pipeline does not use.

**No reference was invented, and no citation is made to a paper the repository does not rely on.**

---

## 4. Numerical consistency checks

Automated string audit of every text run in the `.pptx` against the artefact files. All checks passed
(17/17).

| Claim on poster | Poster text | Source artefact | Status |
|---|---|---|---|
| Headline test macro-F1 | 0.9350 | `evaluation/phase4_test/metrics.json` → `macro_f1` | pass |
| Test accuracy | 0.9497 | same → `accuracy` | pass |
| Test weighted F1 | 0.9492 | same → `weighted_f1` | pass |
| Test macro-F1 95 % CI | [0.9137, 0.9553] | same → `macro_f1_ci95` | pass |
| Errors / test size | 47 / 935 | same → `n_incorrect`, `n_test_images` | pass |
| Validation macro-F1, RF | 0.9539 | `evaluation/metrics/p0_comparison.json` → runs.A5 | pass |
| Validation macro-F1, ResNet50 | 0.9521 | same → runs.A3 | pass |
| Validation macro-F1, MobileNetV2 multi-task / frozen | 0.9401 / 0.9376 | same → runs.A4 / runs.A2 | pass |
| Paired difference | +0.0018 | same → `deployment_decision.head_to_head.delta` | pass |
| Difference 95 % CI | [−0.0207, +0.0233] | same → `delta_ci95` | pass |
| McNemar p | 0.253 | same → `mcnemar.p_value` | pass |
| Dataset size | 6,150 | `splits.csv` row count; HF card | pass |
| Split sizes | 4,286 / 929 / 935 | `splits.csv` | pass |
| Feature count | 83 | `docs`/model card; `website/utils/image_processor.py` | pass |
| Damaged F1 / recall / support | 0.8636 / 0.7755 / 49 | `phase4_test/metrics.json` per class | pass |
| Largest confusion | Damaged → Old, 9 of 49 | `phase4_test/predictions.csv` | pass |
| Label-cluster ambiguity | 41 clusters / 134 images | `datasets/metadata/…` duplicate-cluster report | pass |
| Robustness extremes | 0.90 control, 0.43 low resolution | `evaluation/phase4_robustness/summary.json` | pass |
| Inference latency / artefact | 27.2 ms, 15.7 MB | `inference_latency.json` (A3/A5) | pass |

Forbidden-phrase scan (all absent): "detects internal spoilage", "guarantees freshness", "food-safe",
"superior to deep learning", "outperforms all", and any statement mixing 6,850 with the working set.

### Integrity statements actually made on the poster
- 6,150 (Hugging Face) and 6,850 (original publication) are named as **different distributions** and never mixed.
- Validation scores sit in section 8 and are labelled "validation"; the test figure is a separate
  dark-green panel labelled "scored once, after model selection".
- Random Forest is stated as the deployed model; the three deep models are described as *compared*, not deployed.
- The app is described as predicting **quality**, not vegetable type.
- Controlled-degradation results are called "a degradation study — not field photography".
- The accessibility audit is called "a check, not a conformance claim" (no WCAG claim).
- Section 12 states: external appearance only, no internal spoilage / contamination / edibility / nutrition,
  Ripe and Unripe are maturity labels.

---

## 5. Figures — provenance

Every figure is generated from repository artefacts by `poster_source/make_assets.py`; none is drawn by hand,
stock photography or synthesised.

| File | Source | Notes |
|---|---|---|
| `fig_distribution.png` | `datasets/metadata/splits.csv` | counts computed, not typed |
| `samples_grid.png` | test-split images, one per class | real dataset photographs, labels as provided |
| `fig_models.png` | `p0_comparison.json` validation runs | CIs from the stored bootstrap results |
| `cm_test.png` | `phase4_test/predictions.csv` | confusion matrix rebuilt from saved predictions |
| `fig_per_class.png` | `phase4_test/metrics.json` | precision/recall/F1 per class |
| `fig_robustness.png` | `phase4_robustness/summary.json` | accuracy per controlled condition |
| `pipeline_visual.png` | `website/utils/image_processor.py` | uses the deployed functions |
| `app_screenshot.png` | captured from the running Streamlit app | no invented UI |

The methodology flowchart is **not** an image: it is drawn with native PPTX rounded rectangles and
connectors, so it stays editable.

---

## 6. Editability and PDF/PPTX parity

- **114 shapes** on one slide: 8 pictures, 3 native tables, the rest text boxes and auto-shapes.
- All text, tables, colours and the flowchart are editable in PowerPoint. Only the eight charts are
  raster images (200 dpi at final size), which is a deliberate trade-off: they are regenerable from
  code by re-running `make_assets.py`.
- The PDF is produced **from this exact PPTX** by PowerPoint's own exporter
  (`poster_source/export_pdf.ps1`, `SaveAs` format 32), so PDF and PPTX contain the same content by
  construction. The PNG preview is rendered from the same export path.
- Verified after export: PDF page count 1, MediaBox 1189 × 841 mm, file 745 KB.

---

## 7. Layout audit

Automated geometry check in `build_poster.py` runs at every build and reported:

```
shapes crossing the references band: 0
COL1 body ends at 29.75 in   (band starts 29.86)
COL2 body ends at 29.20 in
COL3 body ends at 29.13 in
```

Rendered at 3600 px wide and inspected column by column (four visual passes in total). Defects found and
fixed during QA:

1. Column 3 ended 4.6 in short of the references band → the robustness study was added as section 10
   (it also closed a genuine content gap).
2. A section header overlapped the last bullet of the preceding block → root cause was the text
   measurement: it used a 1.2 em line height instead of Segoe UI's ~1.33 em, and it ignored the bullet
   indent width. Both fixed in `_para_height`/`caption`, then the whole poster was re-fitted.
3. Column 2 overran the references band by 2.7 in after the type was enlarged → column 2 re-proportioned
   (step boxes, pipeline figure, table row heights) and copy tightened.
4. "19 / 20" wrapped inside its stat card → tightened to "19/20".
5. A vegetable class went missing from the distribution chart in an earlier iteration because the class
   name was hard-coded from the brief ("Chili Pepper") instead of read from `splits.csv`
   ("Chile Pepper"); the asset now derives names from the data.

Final state: three columns within 0.6 in of each other, nothing overlaps, nothing is clipped by the
references band, no text sits outside its own box in the rendered PDF.

---

## 8. Placeholders (must be completed before submission)

These are rendered as clearly marked bracketed placeholders — no invented identities:

| Field | Placeholder on poster |
|---|---|
| Author names (3 slots + 1 guide) | `[Author One]`, `[Author Two]`, `[Author Three]`, `[Faculty Guide]` |
| Department | `[Department of Computer Science and Engineering]` |
| Institution | `[Institution / University Name]` |
| Location | `[City, Country]` |
| Corresponding e-mail | `[email address]` |
| Conference | `[Conference / Symposium Name, Year]` |
| Track / session | `[Conference track / session details]` |
| Institutional logo | none placed (deliberately) |

No IEEE membership grade, no conference logo and no institutional branding were fabricated.

---

## 9. What this audit does **not** establish

- **No physical print proof.** The poster was inspected as PDF/PNG only. Colour appearance on paper
  (especially the dark green `#0E4A32` and the muted greens) depends on the printer, substrate and any
  RGB→CMYK conversion the print shop applies. Ask for a proof if colour fidelity matters.
- **No embedded-font check at the RIP.** The PDF requests Segoe UI; a print shop on a non-Windows system
  may substitute it. Printing the supplied PDF (rather than re-exporting the PPTX elsewhere) avoids this.
- **No conference compliance check.** With no template in the repository, size, margins and column count
  follow the brief's A0/three-column default, not a specific conference's rulebook.
- **Charts are raster.** They are 200 dpi at final size (~2865 px wide for a 14.3 in figure), which is
  ample for print but they are not vector; regenerate them with a larger `DPI` in `make_assets.py` if a
  printer requests it.
- **Readability at 1–2 m was reasoned from point sizes, not measured with people.** Body 17–20 pt on A0
  is comfortably above the common floor; the 13.5 pt footer is fine print by design.

---

# Appendix — A3 VisionX poster (`IPCV_Poster_A3_Landscape`)

A second poster was produced in the format of the supplied course template
(`IPCV_Poster_A3_Landscape.pdf`, DSU VisionX 2026 Project Expo). It reuses the same artefacts, the same
numbers and the same honesty rules; only the format differs.

| Property | Value |
|---|---|
| Size | 16.535 × 11.693 in = **420 × 297 mm** (A3 landscape) |
| PDF | 1 page, MediaBox 1190.5 × 841.9 pt = **420 × 297 mm** |
| Preview | `ipcv_poster_preview.png`, 2600 × 1839 px |
| Shapes | 104 (all text, tables and the flowchart editable; 2 figures are images) |
| Type | title 24 pt · section headers 14.5 pt · body 8–9.5 pt · captions 8–8.5 pt · micro-labels 7–7.5 pt |

## Template mapping

All ten numbered sections of the template are present and in its order:
1 Abstract · 2 Problem Statement · 3 Objectives (4) · 4 Dataset Source (dataset / images / split / labels)
· 5 Methodology / System Architecture (6-step flow) · 6 IPCV Techniques Used (6 chips) · 7 Results
(Fig. 1 input-to-mask pipeline + Table 1 test metrics) · 8 Tools & Technologies (8 chips) ·
9 Conclusion · 10 References (4, IEEE style) + QR placeholder.

The template places section 8 before 7 in its own text stream, and this poster follows that arrangement
(columns read 5 · 6 · 8, then 7 · 9 · 10).

## Authors as supplied

| Name | USN |
|---|---|
| Skanda Prasad Y | ENG24AM0293 |
| Sannala Tharun Kumar Reddy | ENG24AM0276 |
| Shivaram K | ENG24AM0287 |
| Mohammed Ahamed Khan | ENG24AM0222 |

Guided by **Prof. Ms. Soma Samantha**, Dept. of CSE (AI & ML), Dayananda Sagar University.

## Numbers on this poster

Same automated audit as section 2: accuracy 0.9497, macro-F1 0.9350, weighted F1 0.9492, 47 of 935
errors, 27.2 ms median inference, validation 0.9539 (Random Forest) vs 0.9521 (ResNet50), Δ +0.0018,
95 % CI [−0.0207, +0.0233], McNemar p = 0.253, split 4,286 / 929 / 935, 6,150 images, 83 features —
all read from the artefacts at build time, none typed by hand. Validation and test are labelled
separately; the 6,150 vs 6,850 distinction is stated under the dataset table; no food-safety or
superiority claim is made.

## A3-specific defects found and fixed during rendering

1. Title at 27 pt wrapped onto the subtitle → 24 pt, header reflowed.
2. The A0 figures became unreadable at 5 in → `make_a3_assets.py` redraws them at placement size.
3. QR placeholder sat in the wrong column (an EMU/inches mix-up in its x-position) → fixed.
4. Dataset table centred its values → detail column left-aligned.
5. Three columns were re-fitted to end at 11.10 / 11.08 / 11.07 in against an 11.11 in limit.

## Still a placeholder

`QR` box in section 10 (no repository URL was supplied) — drop in the code/demo QR before printing.
Everything else on the sheet is final apart from the same institution-level fields the template fixes
(university, department, course code and semester are taken from the template itself).
