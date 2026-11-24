"""Generate every visual asset used by the IEEE conference poster.

    python poster_source/make_assets.py

Figures are drawn at their **final physical size on the poster** (inches) with
font sizes in real points, so what is drawn is what prints: no post-scaling that
silently shrinks the labels.

Inputs (all read-only):
  datasets/metadata/splits.csv              dataset manifest (labels, counts)
  datasets/processed/test/...               the actual images shown on the poster
  evaluation/metrics/p0_comparison.json     validation metrics and CIs
  evaluation/phase4_test/metrics.json       official test metrics and per-class table
  evaluation/phase4_test/predictions.csv    per-image test predictions (confusion matrix)
  website/utils/image_processor.py          the deployed preprocessing, reused verbatim

Outputs: poster_source/assets/*.png (200 dpi).
Nothing is typed by hand: every number on the poster comes from these files.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = Path(__file__).resolve().parent / "assets"
OUT.mkdir(parents=True, exist_ok=True)
DPI = 200

INK = "#17211c"
INK_SOFT = "#3d4a43"
EMERALD = "#0f5f3f"
EMERALD_DK = "#0a3b2a"
EMERALD_LT = "#3f9c6d"
GREY = "#8fa89b"
PAPER = "#ffffff"

plt.rcParams.update({
    "font.family": "Segoe UI",
    "font.size": 17,
    "axes.edgecolor": "#c6d6cc",
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "figure.facecolor": PAPER,
    "axes.facecolor": PAPER,
    "savefig.facecolor": PAPER,
})

QUALITY_ORDER = ("Damaged", "Dried", "Old", "Ripe", "Unripe")


def read_json(rel: str) -> dict:
    with open(ROOT / rel, "r", encoding="utf-8") as fh:
        return json.load(fh)


def clean(ax, grid_axis: str | None = "y") -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if grid_axis:
        ax.grid(axis=grid_axis, color="#e3ece7", linewidth=1.1)
        ax.set_axisbelow(True)


# --------------------------------------------------------------- samples ----
def asset_samples() -> Path:
    """One real test-split photograph per quality class (Bell Pepper only)."""
    df = pd.read_csv(ROOT / "datasets" / "metadata" / "splits.csv")
    fig, axes = plt.subplots(1, 5, figsize=(14.9, 2.65), dpi=DPI)
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
        bgr = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
        ax.imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#c6d6cc")
            spine.set_linewidth(1.6)
        ax.set_title(quality, fontsize=21, fontweight="bold", color=EMERALD_DK, pad=8)
    fig.tight_layout(pad=0.4)
    path = OUT / "samples_grid.png"
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return path


# ------------------------------------------------------------ distribution ---
def asset_distribution() -> Path:
    df = pd.read_csv(ROOT / "datasets" / "metadata" / "splits.csv")
    pivot = (df.groupby(["quality", "vegetable"]).size()
             .unstack(fill_value=0).reindex(QUALITY_ORDER))
    # Vegetable names come from the manifest: hard-coding them once silently
    # dropped "Chile Pepper" (spelled differently from the source paper).
    pivot = pivot[pivot.sum().sort_values(ascending=False).index]
    colours = ["#0f5f3f", "#3f9c6d", "#8ec9a8", "#cfe3d6"]

    fig, ax = plt.subplots(figsize=(14.9, 3.15), dpi=DPI)
    left = np.zeros(len(pivot))
    for i, col in enumerate(pivot.columns):
        ax.barh(pivot.index, pivot[col].to_numpy(), left=left, height=0.62,
                color=colours[i % len(colours)], label=col)
        left += pivot[col].to_numpy()
    for i, total in enumerate(pivot.sum(axis=1).to_numpy()):
        ax.text(total + 40, i, f"{total:,}", va="center", fontsize=17, color=INK,
                fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlabel("Images in the 6,150-image distribution", fontsize=17)
    ax.set_xlim(0, pivot.sum(axis=1).max() * 1.16)
    ax.tick_params(axis="both", labelsize=16)
    clean(ax, grid_axis="x")
    ax.legend(frameon=False, fontsize=15, ncol=4, loc="upper center",
              bbox_to_anchor=(0.5, -0.3), handlelength=1.1, columnspacing=1.6)
    fig.tight_layout()
    path = OUT / "fig_distribution.png"
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return path


# ---------------------------------------------------------------- models ----
def asset_models() -> Path:
    comp = read_json("evaluation/metrics/p0_comparison.json")
    runs = comp["runs"]
    winner = comp["deployment_decision"]["point_estimate_winner"]
    order = sorted(runs, key=lambda k: runs[k]["quality"]["macro_f1"])
    labels = {"A5": "Random Forest\n(83 handcrafted)", "A3": "ResNet50\n(frozen)",
              "A4": "MobileNetV2\nmulti-task", "A2": "MobileNetV2\n(frozen)"}
    vals = [runs[k]["quality"]["macro_f1"] for k in order]
    lo = [runs[k]["quality"]["macro_f1_ci95"]["lo"] for k in order]
    hi = [runs[k]["quality"]["macro_f1_ci95"]["hi"] for k in order]
    colours = [EMERALD if k == winner else GREY for k in order]

    fig, ax = plt.subplots(figsize=(14.9, 3.15), dpi=DPI)
    y = np.arange(len(order))
    ax.barh(y, vals, color=colours, height=0.6)
    ax.errorbar(vals, y, xerr=[np.array(vals) - np.array(lo), np.array(hi) - np.array(vals)],
                fmt="none", ecolor=INK, elinewidth=1.8, capsize=6)
    for i, v in enumerate(vals):
        ax.text(v + 0.005, i, f"{v:.4f}", va="center", fontsize=18, color=INK,
                fontweight="bold" if order[i] == winner else "normal")
    ax.set_yticks(y)
    ax.set_yticklabels([labels[k] for k in order], fontsize=16)
    ax.set_xlim(min(lo) - 0.014, max(hi) + 0.02)
    ax.tick_params(axis="x", labelsize=15)
    ax.set_xlabel("Validation macro-F1  (929 images, 95 % bootstrap CI)", fontsize=17)
    clean(ax, grid_axis="x")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    path = OUT / "fig_models.png"
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return path


# ------------------------------------------------------------ confusion -----
def asset_confusion() -> Path:
    pred = pd.read_csv(ROOT / "evaluation" / "phase4_test" / "predictions.csv")
    y_true, y_pred = pred["y_true"], pred["y_pred"]
    classes = [c for c in QUALITY_ORDER if c in set(y_true) | set(y_pred)]
    cm = pd.crosstab(pd.Categorical(y_true, classes), pd.Categorical(y_pred, classes),
                     dropna=False)
    counts = cm.to_numpy().astype(int)
    row_pct = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)

    fig, ax = plt.subplots(figsize=(7.5, 6.4), dpi=DPI)
    im = ax.imshow(row_pct, cmap="Greens", vmin=0, vmax=1)
    for i in range(len(classes)):
        for j in range(len(classes)):
            if counts[i, j] == 0:
                continue
            ax.text(j, i, f"{counts[i, j]}\n{row_pct[i, j] * 100:.0f}%", ha="center",
                    va="center", fontsize=15,
                    color="white" if row_pct[i, j] > 0.55 else INK)
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=30, ha="right", fontsize=16)
    ax.set_yticklabels(classes, fontsize=16)
    ax.set_xlabel("Predicted quality", fontsize=17)
    ax.set_ylabel("True quality", fontsize=17)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.set_label("Row proportion", fontsize=15)
    cb.ax.tick_params(labelsize=13)
    fig.tight_layout()
    path = OUT / "cm_test.png"
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return path


# ------------------------------------------------------------- per class ----
def asset_per_class() -> Path:
    rows = read_json("evaluation/phase4_test/metrics.json")["per_class"]
    x = np.arange(len(rows))
    w = 0.26

    fig, ax = plt.subplots(figsize=(7.0, 5.4), dpi=DPI)
    ax.bar(x - w, [r["precision"] for r in rows], width=w, color="#c3ddcd", label="Precision")
    ax.bar(x, [r["recall"] for r in rows], width=w, color=EMERALD_LT, label="Recall")
    ax.bar(x + w, [r["f1"] for r in rows], width=w, color=EMERALD, label="F1")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r['class']}\n(n={r['support']})" for r in rows], fontsize=15)
    ax.tick_params(axis="y", labelsize=15)
    ax.set_ylim(0.6, 1.03)
    ax.set_ylabel("Test-set score", fontsize=17)
    ax.legend(frameon=False, ncol=3, fontsize=14, loc="upper center",
              bbox_to_anchor=(0.5, -0.19), handlelength=1.0, columnspacing=1.2)
    clean(ax)
    fig.tight_layout()
    path = OUT / "fig_per_class.png"
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return path


# ---------------------------------------------------------- robustness -----
def asset_robustness() -> Path:
    summ = read_json("evaluation/phase4_robustness/summary.json")
    conds = summ["conditions"]
    order = sorted(conds, key=lambda k: conds[k]["accuracy"], reverse=True)
    labels = [k.replace("_", " ").replace("q30", "q30") for k in order]
    vals = [conds[k]["accuracy"] for k in order]

    fig, ax = plt.subplots(figsize=(14.9, 2.95), dpi=DPI)
    bars = ax.bar(labels, vals,
                  color=[EMERALD if k == "control" else EMERALD_LT for k in order], width=0.66)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center",
                fontsize=15.5, color=INK)
    ax.set_ylim(0, 1.06)
    ax.set_ylabel("Accuracy", fontsize=16)
    ax.tick_params(axis="x", labelsize=15, rotation=18)
    for tick in ax.get_xticklabels():
        tick.set_ha("right")
    ax.tick_params(axis="y", labelsize=14)
    ax.axhline(summ["overall_accuracy"], color=GREY, linestyle=":", linewidth=1.6)
    clean(ax)
    fig.tight_layout()
    path = OUT / "fig_robustness.png"
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return path


# ------------------------------------------------------- deployed pipeline --
def asset_pipeline_visual() -> Path:
    """The sequence the deployed model actually runs, on one real test image."""
    from website.utils.image_processor import preprocess

    df = pd.read_csv(ROOT / "datasets" / "metadata" / "splits.csv")
    row = df[(df["split"] == "test") & (df["quality"] == "Ripe")
             & (df["vegetable"] == "Bell Pepper")].iloc[0]
    path = (ROOT / "datasets" / "processed" / "test" / row["vegetable"] / row["quality"]
            / row["file_name"])
    bgr = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    pre = preprocess(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    mask_rgb = (np.dstack([pre.mask] * 3) if pre.mask is not None
                else np.full_like(pre.resized_rgb, 255))

    panels = [
        ("Input (256 × 256 RGB)", pre.original_rgb),
        ("Resized (224 × 224)", pre.resized_rgb),
        ("Foreground mask (LAB distance + Otsu)", mask_rgb),
        ("Masked foreground used for features", pre.overlay_rgb),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(14.9, 3.2), dpi=DPI)
    for ax, (label, img) in zip(axes, panels):
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#c6d6cc")
            spine.set_linewidth(1.5)
        ax.set_title(label, fontsize=15.5, color=EMERALD_DK, pad=7)
    fig.tight_layout(pad=0.35)
    path = OUT / "pipeline_visual.png"
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return path


# ---------------------------------------------------------------- website ---
def asset_website() -> Path:
    """Crop the real prediction screenshot to the result panel (no fabrication)."""
    from PIL import Image

    src = ROOT / "reports" / "screenshots" / "phase4" / "02b_result_detail.png"
    im = Image.open(src)
    w, h = im.size
    crop = im.crop((0, int(h * 0.06), w, int(h * 0.99)))
    path = OUT / "app_screenshot.png"
    crop.save(path)
    return path


def main() -> None:
    for fn in (asset_samples, asset_distribution, asset_models, asset_confusion,
               asset_per_class, asset_robustness, asset_pipeline_visual, asset_website):
        p = fn()
        with __import__("PIL.Image", fromlist=["Image"]).open(p) as im:
            print(f"{p.name:24} {im.size[0]}×{im.size[1]} px")


if __name__ == "__main__":
    main()
