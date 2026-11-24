"""Upload widget and its validation feedback."""

from __future__ import annotations

import numpy as np
import streamlit as st

from website.utils.paths import ALLOWED_EXTENSIONS, MAX_UPLOAD_BYTES
from website.utils.samples import sample_images
from website.utils.validation import ValidationError, decode_image, describe_upload, validate_upload


def render_uploader(key: str = "uploader") -> tuple[np.ndarray | None, dict | None, str | None]:
    """Render the input controls; return ``(rgb, info, error)``.

    Exactly one of ``rgb`` and ``error`` is non-None once an image is supplied.
    """
    st.subheader("1 · Provide an image")
    st.caption(
        f"JPG, JPEG or PNG · up to {MAX_UPLOAD_BYTES / 1024 ** 2:.0f} MB · "
        "minimum 64×64 pixels")

    # ``?sample=N`` opens the app on a sample image, which makes demo links
    # shareable without anyone having to prepare a file.
    default_mode = 1 if "sample" in st.query_params else 0
    mode = st.radio(
        "Source",
        options=("Upload my own image", "Use a sample validation image"),
        index=default_mode,
        horizontal=True,
        label_visibility="collapsed",
        key=f"{key}_mode",
    )

    name: str | None = None
    data: bytes | None = None

    if mode == "Upload my own image":
        upload = st.file_uploader(
            "Choose a vegetable photograph",
            type=[e.lstrip(".") for e in ALLOWED_EXTENSIONS],
            key=key,
            label_visibility="collapsed",
        )
        if upload is None:
            st.info("No image selected. Upload a photograph to run a prediction.")
            return None, None, None
        name, data = upload.name, upload.getvalue()
    else:
        samples = sample_images()
        if not samples:
            st.warning("No sample images are available in this checkout.")
            return None, None, None
        default_index = 0
        try:
            requested = int(st.query_params.get("sample", 0))
            default_index = max(0, min(requested, len(samples) - 1))
        except (TypeError, ValueError):
            default_index = 0

        choice = st.selectbox(
            "Sample image",
            options=list(range(len(samples))),
            index=default_index,
            format_func=lambda i: samples[i]["label"],
            key=f"{key}_sample",
        )
        st.caption("Held-out validation images. Their true labels are shown; the "
                   "model does not see them.")
        name, data = samples[choice]["name"], samples[choice]["bytes"]

    try:
        validate_upload(name, data)
        rgb = decode_image(data)
    except ValidationError as exc:
        st.error(f"**Rejected:** {exc}")
        return None, None, str(exc)

    info = describe_upload(name, data, rgb)
    return rgb, info, None


def render_upload_facts(info: dict) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Width", f"{info['width']} px")
    c2.metric("Height", f"{info['height']} px")
    c3.metric("File size", f"{info['size_mb']:.2f} MB")
    c4.metric("Megapixels", f"{info['megapixels']:.2f}")
    st.caption(f"File: `{info['filename']}`")
