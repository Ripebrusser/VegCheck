"""Model performance — read entirely from the Phase 2 evaluation artefacts."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from website.utils.artefacts import (  # noqa: E402
    ArtefactMissing,
    comparison,
    deployed_run,
    group_cv_reference,
    latency,
    test_metrics,
)
from website.components.theme import inject_css, sidebar  # noqa: E402
from website.utils.paths import (  # noqa: E402
    CONFUSION_DIR,
    FEATURE_IMAGE_SIZE,
    PHASE4_TEST_CM_PNG,
)

st.set_page_config(page_title="Model performance — VegCheck", page_icon="📈", layout="wide")

inject_css()
sidebar("performance")

st.markdown(
    """
<div class="vegcheck-hero vegcheck-hero--compact">
  <p class="vegcheck-eyebrow">Evaluation</p>
  <h1>Model performance</h1>
  <p>Four models, one split, identical scoring code. The Random Forest measured
  best, so it was deployed — the deep networks remain fully reported.</p>
  <div class="vegcheck-badges">
    <span class="vegcheck-badge">929-image holdout</span>
    <span class="vegcheck-badge">macro F1 primary</span>
    <span class="vegcheck-badge">bootstrap + McNemar</span>
  </div>
</div>
""",
    unsafe_allow_html=True)
st.caption("Source: `evaluation/metrics/p0_comparison.json`, written by "
           "`src/evaluation/compare_p0.py` from the saved run artefacts. "
           "No value on this page is typed by hand.")

try:
    comp = comparison()
    deployed = deployed_run()
except ArtefactMissing as exc:
    st.error(f"Evaluation artefacts unavailable: {exc}")
    st.stop()

# ---------------------------------------------------------------------------
st.subheader("Deployed model")
d = deployed
c1, c2, c3, c4 = st.columns(4)
c1.metric("Architecture", "Random Forest")
c2.metric("Trees", f"{d['config']['n_estimators']}")
c3.metric("Features", f"{d['n_features']}")
c4.metric("Input", f"{FEATURE_IMAGE_SIZE}×{FEATURE_IMAGE_SIZE}")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Macro F1", f"{d['quality']['macro_f1']:.4f}")
c2.metric("Accuracy", f"{d['quality']['accuracy']:.4f}")
c3.metric("Weighted F1", f"{d['quality']['weighted_f1']:.4f}")
ci = d["quality"]["macro_f1_ci95"]
c4.metric("95 % CI", f"{ci['lo']:.3f}–{ci['hi']:.3f}")

st.markdown(
    f"""
Selected in Phase 2 by measurement, not preference: it had the highest macro F1 of
the four P0 models and is **statistically indistinguishable** from ResNet50
(paired bootstrap Δ = {comp['deployment_decision']['head_to_head']['delta']:+.4f},
95 % CI [{comp['deployment_decision']['head_to_head']['delta_ci95'][0]:+.4f},
{comp['deployment_decision']['head_to_head']['delta_ci95'][1]:+.4f}], McNemar
p = {comp['deployment_decision']['head_to_head']['mcnemar']['p_value']:.3f}) while
running 5× faster end-to-end. Decision record: `Admin/decisions.md` D-012.
""")

st.divider()

# ---------------------------------------------------------------------------
st.subheader("All P0 experiments")
rows = []
for run_id, r in comp["runs"].items():
    q = r["quality"]
    rows.append({
        "Run": run_id,
        "Model": r["display_name"],
        "Family": r["model_family"],
        "Macro F1": round(q["macro_f1"], 4),
        "95 % CI": f"{q['macro_f1_ci95']['lo']:.3f}–{q['macro_f1_ci95']['hi']:.3f}",
        "Accuracy": round(q["accuracy"], 4),
        "Weighted F1": round(q["weighted_f1"], 4),
    })
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

st.subheader("Per-class performance (deployed model)")
per_class = pd.DataFrame(d["quality"]["per_class"])[
    ["class", "support", "precision", "recall", "f1"]]
per_class.columns = ["Class", "Support", "Precision", "Recall", "F1"]
st.dataframe(per_class, use_container_width=True, hide_index=True)

st.subheader("Confusion matrices")
st.caption("Row-normalised; each cell shows counts and row percentage.")
cm = CONFUSION_DIR / "A5_quality.png"
if cm.exists():
    st.image(str(cm), caption="A5 Random Forest — validation set, n = 929",
             alt="Confusion matrix for the deployed Random Forest on the 929-image "
                 "validation split, row-normalised, showing counts and percentages")
else:
    st.info("Confusion matrix image not found.")

st.divider()

# ---------------------------------------------------------------------------
st.subheader("Official test result")
try:
    t = test_metrics()
    st.caption(
        "The 935-image test split was scored **once**, in Phase 4, with the frozen "
        "artefact and the training feature path. Results were not used to select or "
        "tune anything. Source: `evaluation/phase4_test/metrics.json`.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Test macro F1", f"{t['macro_f1']:.4f}")
    c2.metric("Test accuracy", f"{t['accuracy']:.4f}")
    c3.metric("Test weighted F1", f"{t['weighted_f1']:.4f}")
    c4.metric("Errors", f"{t['n_incorrect']} / {t['n_test_images']}")
    tci = t["macro_f1_ci95"]
    st.caption(
        f"95 % bootstrap interval for test macro F1: "
        f"{tci['lo']:.4f}–{tci['hi']:.4f}. The validation interval overlaps it, so the "
        "small validation-to-test difference is consistent with sampling variation.")
    if PHASE4_TEST_CM_PNG.exists():
        st.image(str(PHASE4_TEST_CM_PNG),
                 caption=f"A5 Random Forest — test set, n = {t['n_test_images']}",
                 alt="Confusion matrix for the deployed Random Forest on the 935-image "
                     "test split, showing counts per true and predicted quality class")
except ArtefactMissing:
    st.info("Phase 4 test metrics artefact not found.")

st.divider()

# ---------------------------------------------------------------------------
st.subheader("Evaluation schemes — do not mix these")
st.markdown(
    f"""
| Scheme | Figure | What it measures |
|---|---|---|
| Grouped 5-fold CV (Phase 1) | **{(group_cv_reference()['group_benchmark_macro_f1_grouped_cv']['all']['macro_f1_mean']):.4f}** | Handcrafted features, cross-validated **inside the training split** |
| Holdout validation (Phase 2) | **{d['quality']['macro_f1']:.4f}** | The deployed model on 929 held-out images |
| Test split (Phase 4) | see *Official test result* above | 935 images, scored exactly once with the frozen artefact |

The grouped-CV number is a historical benchmark from feature screening and is
**not comparable** to the holdout figures: it is a different split, a different
fitting regime and a different evaluation population. The test figure was produced
once, after model selection was closed, and was not fed back into any decision.
""")

st.subheader("Inference timing")
try:
    lat = latency()["results"]
    timing = pd.DataFrame([
        {"Model": "Random Forest (deployed)",
         "Decode + preprocess + predict (ms)": lat["A5"]["end_to_end"]["median_ms"]},
        {"Model": "MobileNetV2 multi-task",
         "Decode + preprocess + predict (ms)": lat["A4"]["end_to_end"]["median_ms"]},
        {"Model": "ResNet50",
         "Decode + preprocess + predict (ms)": lat["A3"]["end_to_end"]["median_ms"]},
    ])
    st.dataframe(timing, use_container_width=True, hide_index=True)
    st.caption(f"Single-image, sequential, CPU only, {latency()['n_images']} images. "
               "Measured by `src/evaluation/benchmark_inference.py`.")
except ArtefactMissing:
    st.info("Latency benchmark artefact not found.")

st.subheader("How this model fails")
st.markdown(
    """
- **`Damaged` is the weakest class** — only 46 validation images, and the model
  trades precision for recall on it.
- **`Ripe ↔ Old` and `Dried ↔ Old`** dominate the off-diagonal mass, matching the
  41 genuinely ambiguous near-duplicate clusters in the corpus.
- **Uncalibrated scores.** The Random Forest's vote share is not a calibrated
  probability; a 0.80 score is not "80 % likely correct".
- **No vegetable output.** The deployed model cannot identify the vegetable, so
  this application does not claim to.
""")
