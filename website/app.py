"""VegCheck — Vegetable Quality and Freshness Detection.

Run from inside ``website/``:

    streamlit run app.py

The deployed model is the Random Forest selected in Phase 2 by measurement
(``Admin/decisions.md`` D-012), not by preference. It predicts quality only; no
vegetable prediction is shown because the deployed artefact cannot produce one.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from website.components.prediction_display import render_history, render_result
from website.components.theme import inject_css, sidebar
from website.components.preprocessing_view import render_preprocessing
from website.components.upload import render_upload_facts, render_uploader
from website.utils.artefacts import deployed_run
from website.utils.cache import get_bundle, get_model_summary
from website.utils.history import add_entry
from website.utils.image_processor import preprocess
from website.utils.model_loader import ModelUnavailable, PredictionError, predict_quality
from website.utils.paths import (
    DEPLOYED_MODEL_NAME,
    FEATURE_IMAGE_SIZE,
    QUALITY_CLASSES,
)

st.set_page_config(
    page_title="VegCheck — Vegetable Quality Detection",
    page_icon="🥦",
    layout="wide",
)

inject_css()
sidebar()


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

try:
    run = deployed_run()
    headline = (f"Deployed model · {DEPLOYED_MODEL_NAME} · "
                f"validation macro-F1 {run['quality']['macro_f1']:.4f}")
except Exception:  # noqa: BLE001 - the page must still render without artefacts
    headline = f"Deployed model · {DEPLOYED_MODEL_NAME}"

st.markdown(
    f"""
<div class="vegcheck-hero">
  <p class="vegcheck-eyebrow">Image Processing &amp; Computer Vision · 24AM3504</p>
  <h1>VegCheck</h1>
  <p>Vegetable quality classification from a single photograph, using classical
  image processing and machine learning.</p>
  <div class="vegcheck-badges">
    <span class="vegcheck-badge">{headline}</span>
    <span class="vegcheck-badge">{FEATURE_IMAGE_SIZE}×{FEATURE_IMAGE_SIZE} input</span>
    <span class="vegcheck-badge">{len(QUALITY_CLASSES)} quality classes</span>
  </div>
</div>
""",
    unsafe_allow_html=True)

with st.expander("What this system does — and does not do", expanded=False):
    st.markdown(
        f"""
**It predicts one of {len(QUALITY_CLASSES)} visual quality categories:**
{", ".join(f"`{c}`" for c in QUALITY_CLASSES)}.

The model measures colour, texture and shape descriptors of the segmented
vegetable and classifies the image against patterns learned from 4,286 training
photographs.

**What it cannot do.** It cannot see inside a vegetable, so it cannot establish
internal freshness, edibility, contamination or nutritional value. *Ripe* and
*Unripe* describe **maturity**, not safety. A prediction is a visual similarity
judgement, not a food-safety verdict.

**Vegetable identification is not shown.** The deployed model predicts quality
only. A multi-task network that also identified the vegetable was trained and
evaluated in Phase 2, but it was not the best model and is not used here.
""")

try:
    summary = get_model_summary()
    get_bundle()
    model_ok = True
except ModelUnavailable as exc:
    model_ok = False
    st.error(
        f"**The trained model could not be loaded.** {exc}\n\n"
        "The interface below still renders, but predictions are disabled. "
        "Produce the artefact with "
        "`python -m src.training.train --run-id A5`.")
    summary = None
except Exception as exc:  # noqa: BLE001
    model_ok = False
    st.error(f"**Unexpected error while loading the model.** {exc}")
    summary = None

st.divider()

# ---------------------------------------------------------------------------
# Upload → preprocess → predict
# ---------------------------------------------------------------------------


def _run_prediction(pre, filename: str):
    """Single inference path, used by both the button and a ``?predict=1`` link."""
    bundle = get_bundle()
    prediction = predict_quality(bundle, pre.feature_row)
    st.session_state["last_prediction"] = prediction
    st.session_state["last_filename"] = filename
    add_entry(
        st.session_state,
        filename=filename,
        quality=prediction.label,
        confidence=prediction.confidence,
        model=DEPLOYED_MODEL_NAME,
        latency_ms=prediction.latency_ms,
    )
    return prediction



rgb, info, upload_error = render_uploader()

if rgb is not None and info is not None:
    st.subheader("2 · Uploaded image")
    left, right = st.columns([1, 1.1])
    with left:
        st.image(rgb, caption=info["filename"], use_container_width=True,
                 alt=f"Uploaded vegetable photograph: {info['filename']}")
    with right:
        render_upload_facts(info)

    try:
        pre = preprocess(rgb)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Preprocessing failed: {exc}")
        pre = None

    if pre is not None:
        st.divider()
        render_preprocessing(pre)
        st.divider()

        st.subheader("4 · Run the model")
        if not model_ok:
            st.warning("Prediction is disabled because the model is unavailable.")
        else:
            auto = "predict" in st.query_params
            clicked = st.button("Predict quality", type="primary")
            if (clicked or auto) and st.session_state.get("last_filename") != info["filename"]:
                try:
                    _run_prediction(pre, info["filename"])
                except (ModelUnavailable, PredictionError) as exc:
                    st.error(f"Prediction failed: {exc}")
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Unexpected prediction error: {exc}")

        prediction = st.session_state.get("last_prediction")
        if prediction is not None and model_ok:
            st.divider()
            render_result(prediction, st.session_state.get("last_filename", info["filename"]))

st.divider()
render_history(st.session_state)

st.divider()
st.markdown(
    f"""
<div class="vegcheck-footer">
  <span>Academic project · prediction is a visual classification only</span>
  <span>{DEPLOYED_MODEL_NAME} · 224×224 input · no images or personal data stored</span>
</div>
""",
    unsafe_allow_html=True)
