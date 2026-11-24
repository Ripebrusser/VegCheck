"""A few validation images offered as one-click samples.

Read-only: the bytes are loaded from the processed dataset for demonstration. No
absolute path is ever shown to the user, and nothing is copied or cached.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

import config

#: One image per quality class, chosen deterministically (first of each class in
#: the validation split), so the samples are reproducible.
MAX_SAMPLES = 5


@lru_cache(maxsize=1)
def sample_images() -> tuple[dict, ...]:
    manifest = pd.read_csv(config.METADATA_DIR / "splits.csv")
    val = manifest[manifest["split"] == "validation"]

    picked: list[dict] = []
    for quality in config.QUALITY_CLASSES:
        subset = val[val["quality"] == quality]
        if subset.empty:
            continue
        row = subset.iloc[0]
        path = (config.PROCESSED_DIR / "validation" / row["vegetable"]
                / row["quality"] / row["file_name"])
        if not Path(path).exists():
            continue
        picked.append({
            "name": str(row["file_name"]),
            "label": f"{row['quality']} — {row['vegetable']} ({row['file_name']})",
            "quality": str(row["quality"]),
            "vegetable": str(row["vegetable"]),
            "bytes": Path(path).read_bytes(),
        })
    return tuple(picked[:MAX_SAMPLES])
