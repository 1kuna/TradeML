"""Dashboard action dispatch helpers."""

from __future__ import annotations

from typing import Any, Callable, Mapping

from trademl.dashboard.controller import NodeSettings

DashboardAction = Callable[[NodeSettings, dict[str, Any]], dict[str, Any]]


def dispatch_dashboard_action(
    settings: NodeSettings,
    action: str,
    payload: dict[str, Any],
    actions: Mapping[str, DashboardAction],
) -> dict[str, Any]:
    """Run a dashboard action from an explicit action registry."""
    handler = actions.get(action)
    if handler is not None:
        return handler(settings, payload)
    raise ValueError(f"unsupported action: {action}")
