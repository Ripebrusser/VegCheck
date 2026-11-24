"""Centralised paths and metadata for the deployed model.

The deployed model is chosen by measurement, not by preference: see
``Admin/decisions.md`` D-012 and ``reports/phase2_p0_results.md``. The Random
Forest had the highest macro F1 among the four P0 models and is statistically
indistinguishable from ResNet50 (paired bootstrap CI spanning zero), while being
substantially faster and smaller.

Nothing here is a copy of a result: metrics are read from the evaluation
artefacts at runtime.
"""

from __future__ import annotations

import sys
from pathlib import Path

#: Project root, resolved from this file so the app works from any cwd.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402  (path must be set first)

# --- deployed model -------------------------------------------------------
DEPLOYED_RUN_ID = "A5"
DEPLOYED_MODEL_NAME = "Random Forest (83 handcrafted features)"
DEPLOYED_MODEL_PATH = config.FINAL_MODEL_DIR / "rf_handcrafted.joblib"

#: A5 predicts quality only. The application must never invent a vegetable
#: prediction from it — see D-012.
SUPPORTS_VEGETABLE = False

# --- artefacts the app reads (never rewrites) -----------------------------
COMPARISON_JSON = config.METRICS_DIR / "p0_comparison.json"
LATENCY_JSON = config.METRICS_DIR / "inference_latency.json"
SPLIT_SUMMARY_JSON = config.METADATA_DIR / "split_summary.json"
AUDIT_SUMMARY_JSON = config.METADATA_DIR / "audit_summary.json"
SPLITS_CSV = config.METADATA_DIR / "splits.csv"
CONFUSION_DIR = config.CONFUSION_DIR

#: Phase 1 grouped-CV reference, kept deliberately separate from holdout numbers.
FEATURE_SCREENING_JSON = config.EVAL_REPORT_DIR / "feature_screening.json"

#: The one-time Phase 4 test evaluation. Scored once with the frozen artefact;
#: never used for model selection (``evaluation/phase4_test/metrics.json``).
PHASE4_TEST_METRICS_JSON = config.EVALUATION_DIR / "phase4_test" / "metrics.json"
PHASE4_TEST_CM_PNG = config.EVALUATION_DIR / "phase4_test" / "confusion_matrix.png"

# --- input constraints ----------------------------------------------------
ALLOWED_EXTENSIONS = (".jpg", ".jpeg", ".png")
MAX_UPLOAD_BYTES = 15 * 1024 * 1024      # 15 MiB
MIN_SIDE = 64                            # smaller than this is not a photograph
MAX_SIDE = 10_000                        # larger than this is not a vegetable photo

#: The size at which the training features were extracted.
#:
#: This MUST be ``config.IMAGE_SIZE`` (224): ``src.features.screening._load_one``
#: resizes to ``config.IMAGE_SIZE`` before computing descriptors, so 224 is the
#: distribution the Random Forest was fitted on. ``SOURCE_IMAGE_SIZE`` (256) is the
#: resolution of the files on disk, not the model input — using it here was a Phase
#: 3 defect corrected in Phase 4 (see ``reports/phase4_integrity_audit.md``).
FEATURE_IMAGE_SIZE = config.IMAGE_SIZE

QUALITY_CLASSES = tuple(config.QUALITY_CLASSES)
VEGETABLE_CLASSES = tuple(config.VEGETABLE_CLASSES)
