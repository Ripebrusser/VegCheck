"""Read-only access to the project's evaluation and dataset artefacts.

The application displays no number that is not read from a file the experiments
wrote. If an artefact is missing the pages say so instead of falling back to a
remembered value.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from website.utils.paths import (
    AUDIT_SUMMARY_JSON,
    COMPARISON_JSON,
    FEATURE_SCREENING_JSON,
    LATENCY_JSON,
    PHASE4_TEST_METRICS_JSON,
    SPLIT_SUMMARY_JSON,
    SPLITS_CSV,
)


class ArtefactMissing(RuntimeError):
    """A required evaluation artefact is absent."""


def _read_json(path: Path) -> dict:
    if not path.exists():
        raise ArtefactMissing(f"{path.name} not found at {path.parent}")
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


@lru_cache(maxsize=1)
def comparison() -> dict:
    """P0 comparison: metrics, CIs, timings, deployment decision."""
    return _read_json(COMPARISON_JSON)


@lru_cache(maxsize=1)
def latency() -> dict:
    return _read_json(LATENCY_JSON)


@lru_cache(maxsize=1)
def group_cv_reference() -> dict:
    """Phase 1 grouped-CV benchmark, kept separate from holdout results."""
    return _read_json(FEATURE_SCREENING_JSON)


@lru_cache(maxsize=1)
def test_metrics() -> dict:
    """Official Phase 4 test evaluation, written once and never re-derived."""
    return _read_json(PHASE4_TEST_METRICS_JSON)


@lru_cache(maxsize=1)
def split_summary() -> dict:
    return _read_json(SPLIT_SUMMARY_JSON)


@lru_cache(maxsize=1)
def audit_summary() -> dict:
    return _read_json(AUDIT_SUMMARY_JSON)


@lru_cache(maxsize=1)
def class_distribution() -> dict[str, Any]:
    """Counts per vegetable and per quality, computed from the frozen split file."""
    import pandas as pd

    if not SPLITS_CSV.exists():
        raise ArtefactMissing("splits.csv not found")
    df = pd.read_csv(SPLITS_CSV)

    per_quality = df["quality"].value_counts()
    per_vegetable = df["vegetable"].value_counts()
    per_split = df["split"].value_counts()

    pivot = (df.groupby(["vegetable", "quality"]).size()
             .unstack(fill_value=0)
             .reindex(index=sorted(df["vegetable"].unique()),
                      columns=sorted(df["quality"].unique()), fill_value=0))

    cells = pivot.to_numpy().ravel()
    populated = cells[cells > 0]

    return {
        "total": int(len(df)),
        "per_quality": {k: int(v) for k, v in per_quality.items()},
        "per_vegetable": {k: int(v) for k, v in per_vegetable.items()},
        "per_split": {k: int(v) for k, v in per_split.items()},
        "pivot": pivot,
        #: Ratio across the five quality classes (2044 / 317).
        "quality_imbalance_ratio": float(per_quality.max() / per_quality.min()),
        #: Ratio across populated vegetable x quality cells (1234 / 27) — this is
        #: the 45.7x figure quoted in the project documentation.
        "cell_imbalance_ratio": float(populated.max() / populated.min()),
        "largest_cell": int(populated.max()),
        "smallest_cell": int(populated.min()),
        "n_cells_populated": int(sum(1 for v in pivot.to_numpy().ravel() if v > 0)),
        "n_cells_possible": int(pivot.shape[0] * pivot.shape[1]),
        "absent_cells": [
            {"vegetable": v, "quality": q}
            for v in pivot.index for q in pivot.columns if pivot.loc[v, q] == 0
        ],
    }


@lru_cache(maxsize=1)
def deployed_run() -> dict:
    decision = comparison()["deployment_decision"]
    run_id = decision["point_estimate_winner"]
    return comparison()["runs"][run_id]


#: Findings that are statements about the dataset, not computed metrics. Each is
#: sourced from the corrected Phase 1/2 documentation; the numbers are verified
#: there and repeated here for display only.
DATASET_LIMITATIONS: tuple[str, ...] = (
    "The Hugging Face redistribution used here contains 6,150 images, while the "
    "publication / Mendeley source reports a different corpus total. The numbers "
    "in this application describe the 6,150-image redistribution only.",
    "Tomato/Dried is absent from the available redistribution: 19 of the 20 "
    "vegetable×quality cells are populated, so no model can predict that class.",
    "Ripe and Unripe are maturity labels, not direct measures of spoilage. A "
    "vegetable labelled Ripe is mature, not necessarily safe or unsafe to eat.",
    "Image-based classification cannot establish internal freshness, food safety, "
    "pesticide contamination or nutritional value.",
    "The near-duplicate investigation identified 41 genuinely ambiguous clusters "
    "involving 134 images (2.2%). An earlier 732-image connected component was a "
    "transitive-chaining artefact of dHash union-find at threshold 6, not label "
    "noise.",
    "The original split was retained unchanged. Because near-duplicates were "
    "over-grouped, a 732-image block sits entirely in the training split, so "
    "validation and test cover a narrower slice of the corpus than intended.",
)
