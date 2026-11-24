"""About the project: scope, method and honest limits."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from website.utils.artefacts import comparison, deployed_run  # noqa: E402
from website.components.theme import inject_css, sidebar  # noqa: E402
from website.utils.paths import DEPLOYED_MODEL_NAME  # noqa: E402

st.set_page_config(page_title="About — VegCheck", page_icon="ℹ️", layout="wide")

inject_css()
sidebar("about")

st.markdown(
    """
<div class="vegcheck-hero vegcheck-hero--compact">
  <p class="vegcheck-eyebrow">About</p>
  <h1>About this project</h1>
  <p>Vegetable quality and freshness detection using image processing and deep
  learning — what was built, what was measured, and what it cannot do.</p>
  <div class="vegcheck-badges">
    <span class="vegcheck-badge">6,150 images</span>
    <span class="vegcheck-badge">4 models compared</span>
    <span class="vegcheck-badge">deployed web application</span>
  </div>
</div>
""",
    unsafe_allow_html=True)

st.markdown(
    """
An academic image-processing and computer-vision project.

The system classifies a photograph of a vegetable into one of five visual quality
categories: `Damaged`, `Dried`, `Old`, `Ripe`, `Unripe`.
""")

st.subheader("Method in brief")
st.markdown(
    f"""
1. **Audit.** 6,150 images were checked for corruption, exact duplicates and
   perceptual near-duplicates (dHash). Duplicate clusters were forced to stay
   within one split, so the same physical specimen cannot appear in both training
   and test data.
2. **Classical pipeline.** Foreground segmentation by LAB distance from the
   estimated background, then 83 descriptors: colour moments and hue histograms,
   GLCM Haralick texture, uniform-LBP histograms, and shape morphology.
3. **Deep learning.** MobileNetV2 (single-task and multi-task) and ResNet50 were
   trained as frozen feature extractors under an identical protocol.
4. **Selection.** All four models were scored on the same 929 holdout images with
   macro F1 as the primary metric, backed by bootstrap confidence intervals and
   McNemar tests. The deployed model is **{DEPLOYED_MODEL_NAME}**.
""")

st.subheader("Why a Random Forest is deployed")

try:
    _comp = comparison()
    _deployed = deployed_run()
    _head = _comp["deployment_decision"]["head_to_head"]
    _runner = _comp["runs"][_comp["deployment_decision"]["runner_up"]]
    _why = f"""
Phase 2's result was not the expected one: **deep learning did not outperform the
handcrafted baseline** on this dataset. The Random Forest reached
{_deployed['quality']['macro_f1']:.4f} macro F1 against {_runner['display_name']}'s
{_runner['quality']['macro_f1']:.4f} — a difference of {_head['delta']:+.4f} with a
paired 95 % confidence interval of [{_head['delta_ci95'][0]:+.4f},
{_head['delta_ci95'][1]:+.4f}] that spans zero (McNemar p = {_head['mcnemar']['p_value']:.2f})
— while running in a fraction of the time and memory.

Choosing the network because the project title mentions deep learning would have
been motivated reasoning. The deep models remain fully reported in
`reports/phase2_p0_results.md`; they simply did not win.

This is a finding about *this* dataset at *this* scale with frozen backbones. It
is not a general claim that deep learning is unnecessary for vegetable grading.
"""
except Exception:  # noqa: BLE001 - page must render even without artefacts
    _why = (
        "Phase 2's result was not the expected one: deep learning did not outperform "
        "the handcrafted baseline on this dataset. The Random Forest was selected "
        "because it matched the best network on accuracy at a fraction of the cost. "
        "Evaluation figures are unavailable in this checkout; see "
        "`reports/phase2_p0_results.md`."
    )

st.markdown(_why)

st.subheader("Known limitations")
st.markdown(
    """
- **No internal inspection.** A photograph cannot reveal spoilage, contamination,
  pesticide residue or nutrition.
- **Maturity is not safety.** `Ripe`/`Unripe` describe ripeness, not edibility.
- **One dataset.** 6,150 images of four pepper/tomato types on plain backgrounds,
  from one redistribution. Performance on phone photos in a kitchen is unknown.
- **Missing class.** `Tomato/Dried` does not exist in this redistribution.
- **Label ambiguity.** 41 clusters (134 images) carry genuinely conflicting labels,
  and `Ripe`/`Old` boundaries are inherently fuzzy.
- **One dataset, one test pass.** Validation drove every decision; the 935-image
  test split was scored once in Phase 4 and is now spent, so it cannot arbitrate
  future model comparisons.
""")

st.subheader("Technology")
st.markdown(
    """
Python 3.12 · OpenCV 5 · scikit-image · scikit-learn · TensorFlow 2.21 (CPU) ·
NumPy · pandas · Matplotlib · Streamlit.
""")

st.subheader("Where to read more")
for path, label in (
    ("reports/results.md", "Phase 1 results"),
    ("reports/limitations.md", "Limitations (including the duplicate-cluster correction)"),
    ("reports/phase2_preflight.md", "Phase 2 preflight: leakage audit and risks"),
    ("reports/phase2_p0_results.md", "Phase 2 P0 experiment results"),
    ("reports/phase3_website_report.md", "This application's report"),
    ("Admin/decisions.md", "Decision log"),
):
    st.markdown(f"- `{path}` — {label}")

st.divider()
st.caption("Predictions are visual classifications only. Do not use them for "
           "food-safety decisions.")
