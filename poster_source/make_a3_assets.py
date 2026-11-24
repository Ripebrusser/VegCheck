"""Figures drawn at A3 placement size for the VisionX poster.

    python poster_source/make_a3_assets.py

The A0 figures in ``assets/`` are drawn for a 14.9 in column; placed in a 5 in A3
column their 15-20 pt labels would print at 5-7 pt. These are redrawn at the size
they are actually placed at, so a 6.5 pt label is 6.5 pt on paper.

Outputs (in ``assets/``):
    samples_grid_a3.png     3.55 × 0.78 in  - one test photograph per quality class
    pipeline_visual_a3.png  5.00 × 1.18 in  - the deployed preprocessing, four panels
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "assets"
DPI = 400

EMERALD_DK = "#0a3b2a"
EDGE = "#b9cdc3"
QUALITY_ORDER = ["Damaged", "Dried", "Old", "Ripe", "Unripe"]


def _read(path: Path) -> np.ndarray:
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


def samples_grid() -> Path:
    df = pd.read_csv(ROOT / "datasets" / "metadata" / "splits.csv")
    fig, axes = plt.subplots(1, 5, figsize=(3.55, 0.78), dpi=DPI)
    for ax, quality in zip(axes, QUALITY_ORDER):
        sub = df[(df["quality"] == quality) & (df["split"] == "test")
                 & (df["vegetable"] == "Bell Pepper")]
        if sub.empty:
            sub = df[(df["quality"] == quality) & (df["vegetable"] == "Bell Pepper")]
        row = sub.iloc[0]
        path = (ROOT / "datasets" / "processed" / "test" / row["vegetable"] / quality
                / row["file_name"])
        if not path.exists():
            path = (ROOT / "datasets" / "processed" / row["split"] / row["vegetable"]
                    / quality / row["file_name"])
        ax.imshow(cv2.cvtColor(_read(path), cv2.COLOR_BGR2RGB))
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(EDGE)
            spine.set_linewidth(1.0)
        ax.set_title(quality, fontsize=6.5, fontweight="bold", color=EMERALD_DK, pad=2.5)
    fig.tight_layout(pad=0.18)
    path = OUT / "samples_grid_a3.png"
    fig.savefig(path, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    return path


def pipeline_visual() -> Path:
    import sys
    sys.path.insert(0, str(ROOT))
    from website.utils.image_processor import preprocess

    df = pd.read_csv(ROOT / "datasets" / "metadata" / "splits.csv")
    row = df[(df["split"] == "test") & (df["quality"] == "Ripe")
             & (df["vegetable"] == "Bell Pepper")].iloc[0]
    path = (ROOT / "datasets" / "processed" / "test" / row["vegetable"] / row["quality"]
            / row["file_name"])
    bgr = _read(path)
    pre = preprocess(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    mask_rgb = (np.dstack([pre.mask] * 3) if pre.mask is not None
                else np.full_like(pre.resized_rgb, 255))

    panels = [
        ("Input\n256 × 256", pre.original_rgb),
        ("Resized\n224 × 224", pre.resized_rgb),
        ("Mask\nLAB distance + Otsu", mask_rgb),
        ("Masked\nfeature input", pre.overlay_rgb),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(5.0, 1.52), dpi=DPI)
    for ax, (label, img) in zip(axes, panels):
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(EDGE)
            spine.set_linewidth(0.9)
        ax.set_title(label, fontsize=6.8, color=EMERALD_DK, pad=3, linespacing=1.05)
    fig.tight_layout(pad=0.16)
    path = OUT / "pipeline_visual_a3.png"
    fig.savefig(path, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    return path


def main() -> None:
    for fn in (samples_grid, pipeline_visual):
        p = fn()
        from PIL import Image
        w, h = Image.open(p).size
        print(f"{p.name:24s} {w}×{h} px  ({w / DPI:.2f} × {h / DPI:.2f} in at {DPI} dpi)")


if __name__ == "__main__":
    main()
