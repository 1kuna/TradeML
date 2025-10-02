"""Labeling framework.

Targets include horizon returns and triple-barrier classification.
"""

from .horizon.horizon import horizon_returns  # noqa: F401
from .triple_barrier.triple_barrier import triple_barrier  # noqa: F401

__all__ = [
    "horizon_returns",
    "triple_barrier",
]

