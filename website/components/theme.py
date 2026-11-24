"""Shared chrome for every page: stylesheet, sidebar brand and navigation.

The implicit Streamlit sidebar entry for the main script is labelled with the
filename (``app``), so the built-in navigation is switched off in
``.streamlit/config.toml`` and replaced here with explicitly named links.

The links are plain anchors rather than ``st.page_link`` because ``page_link``
requires pages declared through ``st.navigation``; a bare anchor also renders
without side effects when a page is executed on its own by ``AppTest``.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

CSS_PATH = Path(__file__).resolve().parents[1] / "styles" / "style.css"

#: (key, url, label, icon)
PAGES: tuple[tuple[str, str, str, str], ...] = (
    ("predict", "/", "Predict quality", "🔍"),
    ("dataset", "/Dataset_Information", "Dataset", "📊"),
    ("performance", "/Model_Performance", "Performance", "📈"),
    ("about", "/About_Project", "About", "ℹ️"),
)

_SIDEBAR_HTML = """
<div class="vegcheck-brand">
  <div class="vegcheck-brand-mark">🥦</div>
  <div>
    <div class="vegcheck-brand-name">VegCheck</div>
    <div class="vegcheck-brand-sub">Quality detection</div>
  </div>
</div>
"""

_SIDEBAR_CARD = """
<div class="vegcheck-side-card">
  <b>Image Processing &amp; Computer Vision</b><br>
  24AM3504 · V Semester B.Tech<br>
  Dayananda Sagar University
</div>
"""


def inject_css() -> None:
    """Load the shared stylesheet. Safe to call once per page."""
    if CSS_PATH.exists():
        st.markdown(f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>",
                    unsafe_allow_html=True)


def _nav_html(active: str) -> str:
    items = []
    for key, url, label, icon in PAGES:
        cls = "active" if key == active else ""
        items.append(
            f'<a class="{cls}" href="{url}" target="_self">'
            f'<span class="vegcheck-nav-icon">{icon}</span>{label}</a>'
        )
    return '<nav class="vegcheck-nav">' + "".join(items) + "</nav>"


def sidebar(active: str = "predict") -> None:
    """Brand block, named page links, and the course card."""
    with st.sidebar:
        st.markdown(_SIDEBAR_HTML, unsafe_allow_html=True)
        st.markdown(_nav_html(active), unsafe_allow_html=True)
        st.markdown(_SIDEBAR_CARD, unsafe_allow_html=True)
