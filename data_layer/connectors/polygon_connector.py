"""Backward-compatible Polygon connector alias.

Polygon was renamed to Massive.com in the active connector surface, but several
legacy modules on main still import ``data_layer.connectors.polygon_connector``.
Keep that import path alive so legacy orchestration/tests continue to collect.
"""

from .massive_connector import MassiveConnector as PolygonConnector

__all__ = ["PolygonConnector"]
