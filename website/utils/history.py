"""Session-scoped prediction history.

Deliberately in-memory only: uploaded images are never written to disk and no
personal information is collected. Clearing the session clears the history.
"""

from __future__ import annotations

import time
from typing import MutableMapping

KEY = "prediction_history"
MAX_ENTRIES = 20


def add_entry(state: MutableMapping, *, filename: str, quality: str,
              confidence: float, model: str, latency_ms: float) -> dict:
    """Append one prediction and trim to :data:`MAX_ENTRIES`."""
    entry = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "filename": filename,
        "vegetable": None,          # the deployed model does not predict this
        "quality": quality,
        "confidence": float(confidence),
        "model": model,
        "latency_ms": float(latency_ms),
    }
    entries = list(state.get(KEY, []))
    entries.append(entry)
    state[KEY] = entries[-MAX_ENTRIES:]
    return entry


def get_entries(state: MutableMapping) -> list[dict]:
    return list(state.get(KEY, []))


def clear_entries(state: MutableMapping) -> None:
    state[KEY] = []
