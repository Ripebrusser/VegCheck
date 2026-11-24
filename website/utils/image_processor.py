"""Preprocessing for the deployed model — the *training* pipeline, reused.

A5 was fitted on 83 handcrafted descriptors extracted at 256×256 with the
foreground mask from :func:`src.preprocessing.segment_vegetable`
(``src.features.screening._load_one``).  This module performs exactly that
sequence and nothing more.

Deliberately **not** applied here: CLAHE, bilateral/median/Gaussian filtering and
ImageNet or ``tf`` normalisation. Those belong to experiments that were not
deployed; running them would feed the model a different distribution from the one
it was fitted on.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from src.features.handcrafted import ALL_FEATURES, extract_features
from src.preprocessing.pipeline import segment_vegetable
from website.utils.paths import FEATURE_IMAGE_SIZE

#: Colour of the foreground highlight in the segmentation preview.
_MASK_COLOUR = np.array([40, 200, 120], dtype=np.float32)


@dataclass(frozen=True)
class Preprocessed:
    """Everything the UI needs, plus the exact feature vector for the model."""

    original_rgb: np.ndarray
    resized_rgb: np.ndarray
    mask: np.ndarray | None
    overlay_rgb: np.ndarray
    feature_row: np.ndarray
    feature_values: dict[str, float] = field(default_factory=dict)
    segmentation: dict = field(default_factory=dict)

    @property
    def mask_found(self) -> bool:
        return self.mask is not None


def preprocess(rgb: np.ndarray) -> Preprocessed:
    """RGB uint8 → feature row identical to the training-time extraction."""
    if rgb is None or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("preprocess() expects an RGB uint8 image with 3 channels")

    size = FEATURE_IMAGE_SIZE
    resized = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)

    # segment_vegetable expects BGR, matching the training call site.
    mask, info = segment_vegetable(cv2.cvtColor(resized, cv2.COLOR_RGB2BGR))
    if not info.get("ok", False):
        mask = None

    feats = extract_features(resized, mask)

    # Defence in depth: a drifted feature set must fail loudly, not silently
    # shift every column and produce a confident wrong answer.
    if tuple(feats.keys()) != ALL_FEATURES:
        missing = set(ALL_FEATURES) - set(feats)
        extra = set(feats) - set(ALL_FEATURES)
        raise RuntimeError(
            f"feature set drifted from training: missing={sorted(missing)} "
            f"extra={sorted(extra)}")

    row = np.array([[feats[k] for k in ALL_FEATURES]], dtype=np.float32)
    row = np.nan_to_num(row, nan=0.0, posinf=0.0, neginf=0.0)

    return Preprocessed(
        original_rgb=rgb,
        resized_rgb=resized,
        mask=mask,
        overlay_rgb=overlay(resized, mask),
        feature_row=row,
        feature_values={k: float(v) for k, v in feats.items()},
        segmentation={
            "ok": bool(info.get("ok", False)),
            "area_fraction": float(info.get("area_fraction", 0.0)),
            # count_nonzero, not sum: the mask is uint8 with values 0/255, so
            # summing it returns 255x the number of selected pixels.
            "mask_pixels": int(np.count_nonzero(mask)) if mask is not None else 0,
        },
    )


def overlay(rgb: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    """Green tint over the segmented foreground, for visual inspection only."""
    if mask is None:
        return rgb.copy()
    out = rgb.astype(np.float32).copy()
    sel = mask.astype(bool)
    out[sel] = 0.55 * out[sel] + 0.45 * _MASK_COLOUR
    return np.clip(out, 0, 255).astype(np.uint8)
