"""Streamlit caching wrappers.

The model and the artefacts are loaded once per server process, not on every
widget interaction. Keeping the decorators here means the rest of the code stays
importable and testable without Streamlit.
"""

from __future__ import annotations

import streamlit as st

from website.utils.model_loader import load_bundle, model_summary


@st.cache_resource(show_spinner="Loading the trained model…")
def get_bundle() -> dict:
    """The deployed Random Forest, loaded once per process.

    One throw-away prediction warms scikit-learn's internal buffers so the first
    real request is not reported at several times the steady-state cost.
    """
    import numpy as np

    from src.features.handcrafted import ALL_FEATURES

    bundle = load_bundle()
    bundle["pipeline"].predict_proba(np.zeros((1, len(ALL_FEATURES)), dtype=np.float32))
    return bundle


@st.cache_resource
def get_model_summary() -> dict:
    return model_summary(get_bundle())
