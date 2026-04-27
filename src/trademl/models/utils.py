"""Shared model helpers."""

from __future__ import annotations

import numpy as np


def time_decay_weights(n_rows: int, *, half_life_days: int = 378) -> np.ndarray:
    """Return newest-row-heavy sample weights."""
    if n_rows <= 1:
        return np.ones(n_rows)
    idx = np.arange(n_rows)
    half_life = max(1, int(half_life_days))
    return 0.5 ** ((n_rows - 1 - idx) / half_life)
