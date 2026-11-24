"""Model loading and inference for the deployed Random Forest.

Kept free of Streamlit so it can be unit-tested directly; caching lives in
:mod:`website.utils.cache`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.features.handcrafted import ALL_FEATURES
from website.utils.paths import (
    DEPLOYED_MODEL_PATH,
    DEPLOYED_RUN_ID,
    QUALITY_CLASSES,
)


class ModelUnavailable(RuntimeError):
    """The model artefact is missing or inconsistent with the documented setup."""


class PredictionError(RuntimeError):
    """The model produced something that is not a valid probability distribution."""


@dataclass(frozen=True)
class Prediction:
    label: str
    index: int
    confidence: float
    probabilities: np.ndarray
    class_names: tuple[str, ...]
    latency_ms: float
    second_label: str
    second_confidence: float


def load_bundle(path: Path | None = None) -> dict:
    """Load and verify the deployed artefact."""
    import joblib

    path = Path(path or DEPLOYED_MODEL_PATH)
    if not path.exists():
        raise ModelUnavailable(
            f"Model file not found at {path.name}. "
            "Expected the Phase 2 artefact produced by "
            "`python -m src.training.train --run-id A5`.")

    try:
        bundle = joblib.load(path)
    except Exception as exc:  # noqa: BLE001 - surfaced to the user
        raise ModelUnavailable(f"Model file exists but could not be loaded: {exc}") from exc

    verify_bundle(bundle)

    # Serving-only tuning: the artefact was fitted with n_jobs=-1, which helps
    # for batches but costs ~18 ms of joblib overhead per single-row call.
    # Prediction values are identical either way — this only changes parallelism.
    rf = bundle["pipeline"].named_steps.get("rf")
    if rf is not None and getattr(rf, "n_jobs", None) != 1:
        rf.set_params(n_jobs=1)

    return bundle


def verify_bundle(bundle: dict) -> None:
    """Refuse to serve a model whose label or feature contract has changed."""
    if not isinstance(bundle, dict):
        raise ModelUnavailable("Model artefact has an unexpected structure.")

    for key in ("pipeline", "features", "classes"):
        if key not in bundle:
            raise ModelUnavailable(f"Model artefact is missing '{key}'.")

    features = list(bundle["features"])
    if features != list(ALL_FEATURES):
        raise ModelUnavailable(
            "Model feature list does not match the current handcrafted feature set; "
            "the artefact would produce meaningless predictions.")

    classes = list(bundle["classes"])
    if classes != list(QUALITY_CLASSES):
        raise ModelUnavailable(
            f"Model class order {classes} does not match the project's frozen "
            f"order {list(QUALITY_CLASSES)}.")

    n_out = getattr(bundle["pipeline"].named_steps.get("rf"), "n_classes_", None)
    if n_out is not None and int(n_out) != len(QUALITY_CLASSES):
        raise ModelUnavailable(
            f"Model was trained on {n_out} classes, expected {len(QUALITY_CLASSES)}.")


def predict_quality(bundle: dict, feature_row: np.ndarray) -> Prediction:
    """Run the pipeline and validate its output before anyone sees it."""
    pipeline = bundle["pipeline"]
    classes = list(bundle["classes"])

    X = np.asarray(feature_row, dtype=np.float32)
    if X.ndim != 2 or X.shape[0] != 1 or X.shape[1] != len(ALL_FEATURES):
        raise PredictionError(
            f"expected a (1, {len(ALL_FEATURES)}) feature row, got {X.shape}")
    if not np.all(np.isfinite(X)):
        raise PredictionError("feature vector contains NaN or infinite values")

    t0 = time.perf_counter()
    proba = pipeline.predict_proba(X)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    proba = np.asarray(proba, dtype=np.float64)
    if proba.shape != (1, len(classes)):
        raise PredictionError(
            f"expected probabilities of shape (1, {len(classes)}), got {proba.shape}")
    if not np.all(np.isfinite(proba)):
        raise PredictionError("model returned NaN or infinite probabilities")
    if np.any(proba < -1e-9) or np.any(proba > 1 + 1e-9):
        raise PredictionError("model returned probabilities outside [0, 1]")

    total = float(proba.sum())
    if abs(total - 1.0) > 1e-3:
        raise PredictionError(f"probabilities sum to {total:.4f}, not 1")

    row = proba[0]
    order = np.argsort(row)[::-1]
    best = int(order[0])

    return Prediction(
        label=classes[best],
        index=best,
        confidence=float(row[best]),
        probabilities=row,
        class_names=tuple(classes),
        latency_ms=latency_ms,
        second_label=classes[int(order[1])],
        second_confidence=float(row[int(order[1])]),
    )


def model_summary(bundle: dict) -> dict:
    """Structural facts about the loaded model, read from the artefact."""
    rf = bundle["pipeline"].named_steps.get("rf")
    scaler = bundle["pipeline"].named_steps.get("scale")
    return {
        "run_id": DEPLOYED_RUN_ID,
        "type": "StandardScaler + RandomForestClassifier",
        "n_estimators": int(getattr(rf, "n_estimators", 0)),
        "n_features": len(bundle["features"]),
        "n_classes": len(bundle["classes"]),
        "classes": list(bundle["classes"]),
        "trained_on": int(getattr(rf, "n_features_in_", 0)),
        "scaler_mean_present": scaler is not None,
        "n_trees_fitted": len(getattr(rf, "estimators_", [])),
    }
