"""Side-by-side view of every stage the deployed model actually applies."""

from __future__ import annotations

import streamlit as st

from website.utils.image_processor import Preprocessed
from website.utils.paths import FEATURE_IMAGE_SIZE


def render_preprocessing(pre: Preprocessed) -> None:
    st.subheader("3 · Preprocessing")
    st.caption(
        f"These are the only transformations the deployed model applies: resize to "
        f"{FEATURE_IMAGE_SIZE}×{FEATURE_IMAGE_SIZE}, foreground segmentation, then "
        "colour/texture/shape descriptors. No CLAHE, filtering or ImageNet "
        "normalisation — those belong to experiments that were not deployed.")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.image(pre.original_rgb, caption="Original", use_container_width=True,
                 alt="The uploaded photograph at its original resolution")
    with c2:
        st.image(pre.resized_rgb,
                 caption=f"Resized {FEATURE_IMAGE_SIZE}×{FEATURE_IMAGE_SIZE}",
                 use_container_width=True,
                 alt=f"The photograph resized to {FEATURE_IMAGE_SIZE} by "
                     f"{FEATURE_IMAGE_SIZE} pixels, the model input size")
    with c3:
        st.image(pre.overlay_rgb, caption="Segmented foreground", use_container_width=True,
                 alt="The photograph with the detected foreground vegetable "
                     "highlighted in green")

    seg = pre.segmentation
    c1, c2, c3 = st.columns(3)
    c1.metric("Segmentation", "ok" if seg["ok"] else "fallback")
    c2.metric("Foreground", f"{seg['area_fraction'] * 100:.1f}% of frame")
    c3.metric("Mask pixels", f"{seg['mask_pixels']:,}")

    if not seg["ok"]:
        st.warning(
            "Segmentation did not find a plausible foreground, so shape features are "
            "zeroed and texture is measured on the whole frame — exactly what the "
            "training pipeline does in this situation. Predictions remain valid but "
            "are less informative for such images.")

    with st.expander("Feature vector (83 descriptors)"):
        st.caption("The values actually passed to the Random Forest.")
        items = list(pre.feature_values.items())
        for start in range(0, len(items), 3):
            cols = st.columns(3)
            for col, (name, value) in zip(cols, items[start:start + 3]):
                col.metric(name, f"{value:.4g}")
