"""Docs-backed default vendor request budgets for the data node."""

from __future__ import annotations

from copy import deepcopy

from trademl.data_node.provider_contracts import default_vendor_limits as _default_vendor_limits


DEFAULT_VENDOR_LIMITS: dict[str, dict[str, int]] = _default_vendor_limits()


def default_vendor_limits() -> dict[str, dict[str, int]]:
    """Return a defensive copy of the researched default vendor limits."""
    return deepcopy(DEFAULT_VENDOR_LIMITS)
