"""Class-probability visualisation.

The chart is drawn from the model's own output vector — no smoothing, rounding or
rescaling that could make a weak prediction look decisive.
"""

from __future__ import annotations

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_ACCENT = "#2e7d32"
_MUTED = "#b0bec5"


def probability_figure(class_names, probabilities, predicted_index: int, title: str = ""):
    """Horizontal bar chart with the predicted class highlighted."""
    names = list(class_names)
    probs = np.asarray(probabilities, dtype=float) * 100.0
    if len(names) != len(probs):
        raise ValueError("class names and probabilities have different lengths")

    order = np.argsort(probs)
    colours = [_ACCENT if i == predicted_index else _MUTED for i in order]

    fig, ax = plt.subplots(figsize=(6.4, 0.52 * len(names) + 1.1))
    bars = ax.barh([names[i] for i in order], [probs[i] for i in order],
                   color=colours, height=0.62)

    for bar, value in zip(bars, [probs[i] for i in order]):
        ax.text(bar.get_width() + max(probs) * 0.015, bar.get_y() + bar.get_height() / 2,
                f"{value:.1f}%", va="center", fontsize=9, color="#263238")

    ax.set_xlim(0, max(100.0, float(probs.max()) * 1.18))
    ax.set_xlabel("random-forest vote share (%)")
    ax.set_title(title or "Class scores", fontsize=10, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="both", labelsize=9)
    fig.tight_layout()
    return fig
