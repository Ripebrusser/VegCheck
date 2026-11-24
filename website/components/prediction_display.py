"""Prediction result, probability chart, explanation and history table."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from website.components.confidence_chart import probability_figure
from website.utils.history import clear_entries, get_entries
from website.utils.model_loader import Prediction


def render_result(prediction: Prediction, filename: str) -> None:
    st.subheader("5 · Result")

    c1, c2, c3 = st.columns([1.2, 1, 1])
    c1.metric("Predicted quality", prediction.label)
    c2.metric("Top score", f"{prediction.confidence * 100:.1f}%")
    c3.metric("Inference", f"{prediction.latency_ms:.0f} ms")

    st.caption(
        f"Second most likely: **{prediction.second_label}** at "
        f"{prediction.second_confidence * 100:.1f}%. "
        "Scores are the Random Forest's vote share across its trees. They are "
        "**not calibrated probabilities**, so 0.80 does not mean \"correct 80 % of "
        "the time\", and a high score is not a guarantee of correctness.")

    fig = probability_figure(
        prediction.class_names, prediction.probabilities, prediction.index,
        title=f"All {len(prediction.class_names)} classes — {filename}")
    st.pyplot(fig, use_container_width=True)

    st.markdown(
        f"""
**What this means.** The model classified the uploaded image as **{prediction.label}**,
based on visual patterns — colour, texture and shape descriptors — learned from
the training images.

This is a visual classification result. It is **not** an assessment of whether the
vegetable is safe to eat, and it cannot detect internal spoilage, contamination or
nutritional value. *Ripe* and *Unripe* describe maturity; they are not
"fresh" and "rotten".
"""
    )


def render_history(state) -> None:
    st.subheader("Prediction history")
    entries = get_entries(state)

    if not entries:
        st.caption("No predictions yet in this session. Nothing is written to disk.")
        return

    frame = pd.DataFrame([
        {
            "Time": e["timestamp"],
            "File": e["filename"],
            "Quality": e["quality"],
            "Confidence": f"{e['confidence'] * 100:.1f}%",
            "Inference (ms)": f"{e['latency_ms']:.0f}",
        }
        for e in reversed(entries)
    ])
    st.dataframe(frame, use_container_width=True, hide_index=True)
    st.caption("Session only — clearing the browser session clears this list. "
               "No images or personal data are stored.")

    if st.button("Clear history", type="secondary"):
        clear_entries(state)
        st.rerun()
