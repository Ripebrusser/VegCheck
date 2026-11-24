"""Dataset information — every figure read from the frozen audit artefacts."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from website.utils.artefacts import (  # noqa: E402
    DATASET_LIMITATIONS,
    ArtefactMissing,
    audit_summary,
    class_distribution,
    split_summary,
)
from website.components.theme import inject_css, sidebar  # noqa: E402
from website.utils.paths import QUALITY_CLASSES, VEGETABLE_CLASSES  # noqa: E402

st.set_page_config(page_title="Dataset — VegCheck", page_icon="📊", layout="wide")

inject_css()
sidebar("dataset")

st.markdown(
    """
<div class="vegcheck-hero vegcheck-hero--compact">
  <p class="vegcheck-eyebrow">Dataset</p>
  <h1>Dataset information</h1>
  <p>6,150 photographs of four vegetable types on plain backgrounds, audited for
  corruption and perceptual duplicates before any model saw them.</p>
  <div class="vegcheck-badges">
    <span class="vegcheck-badge">New VegNet redistribution</span>
    <span class="vegcheck-badge">near-duplicate grouped splits</span>
  </div>
</div>
""",
    unsafe_allow_html=True)
st.caption("Figures come from `datasets/metadata/split_summary.json`, "
           "`audit_summary.json` and `splits.csv`.")

try:
    splits = split_summary()
    dist = class_distribution()
except ArtefactMissing as exc:
    st.error(f"Dataset artefacts unavailable: {exc}")
    st.stop()

st.subheader("Source")
st.markdown(
    """
The **New VegNet** vegetable-image dataset, distributed on Hugging Face as a
redistribution of the original publication dataset. Images are stored at
256×256 pixels; the working copy under `datasets/processed/` is unmodified apart
from organisation into split folders.
""")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total images", f"{dist['total']:,}")
c2.metric("Vegetables", str(len(VEGETABLE_CLASSES)))
c3.metric("Quality classes", str(len(QUALITY_CLASSES)))
c4.metric("Cell imbalance", f"{dist['cell_imbalance_ratio']:.1f}×")
st.caption(
    f"Largest populated cell {dist['largest_cell']:,} images, smallest "
    f"{dist['smallest_cell']:,}. Across the five quality classes alone the ratio is "
    f"{dist['quality_imbalance_ratio']:.1f}×.")

st.subheader("Split")
try:
    split_counts = {k: int(v) for k, v in splits.get("counts", {}).items()}
except Exception:  # noqa: BLE001
    split_counts = dist["per_split"]

cols = st.columns(3)
for col, name in zip(cols, ("train", "validation", "test")):
    n = split_counts.get(name, dist["per_split"].get(name, 0))
    pct = 100.0 * n / max(dist["total"], 1)
    col.metric(name.capitalize(), f"{n:,}", f"{pct:.1f}% of corpus")

st.caption(
    "Splits are **grouped by near-duplicate cluster**, so photographs of the same "
    "physical specimen never straddle a boundary. The application's model was "
    "trained on the training split, selected on the validation split, and has "
    "**never** been evaluated on the test split.")

st.subheader("Quality class distribution")
quality_counts = pd.DataFrame(
    [{"Quality": k, "Images": v, "Share": f"{100 * v / dist['total']:.1f}%"}
     for k, v in sorted(dist["per_quality"].items(), key=lambda kv: -kv[1])]
)
st.dataframe(quality_counts, use_container_width=True, hide_index=True)

st.subheader("Vegetable × quality")
pivot = dist["pivot"].copy()
pivot.index.name = "Vegetable"
st.dataframe(pivot, use_container_width=True)
st.caption(
    f"{dist['n_cells_populated']} of {dist['n_cells_possible']} cells are populated. "
    + (", ".join(f"**{c['vegetable']}/{c['quality']}** is empty" for c in dist["absent_cells"])
       if dist["absent_cells"] else ""))

st.subheader("Dataset limitations")
for i, text in enumerate(DATASET_LIMITATIONS, 1):
    st.markdown(f"{i}. {text}")

st.divider()
st.caption(
    "Corrected duplicate statistics supersede the Phase 1 figures (42 clusters / "
    "866 images). The earlier 732-image component was a transitive-chaining "
    "artefact; genuine ambiguity is 41 clusters / 134 images, detailed in "
    "`reports/limitations.md` §L4.")
