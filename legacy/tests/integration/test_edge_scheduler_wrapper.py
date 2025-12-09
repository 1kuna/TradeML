"""
Contract test for edge scheduler wrapper.
"""

from datetime import date
from unittest.mock import patch, MagicMock


def test_run_edge_scheduler_passes_asof_and_returns_status(tmp_path, monkeypatch):
    from data_layer.connectors import run_edge_scheduler

    cfg = tmp_path / "edge.yml"
    cfg.write_text("tasks: []\n")

    # Stub edge_collector module so wrapper import succeeds without running real ingestion
    import types
    import sys

    class DummyCollector:
        def __init__(self, *_args, **_kwargs):
            self._vendor_symbol_cursor = {"alpaca": 0}

        def run(self, *_, **__):
            return None

    dummy_module = types.SimpleNamespace(EdgeCollector=DummyCollector)
    monkeypatch.setitem(sys.modules, "edge_collector", dummy_module)

    res = run_edge_scheduler(
        asof="2024-12-31",
        config_path=str(cfg),
        node_id="node-1",
        budgets={"alpaca": 2},
    )

    assert res["status"] == "ok"
    assert res["asof"] == "2024-12-31"
    assert res["node_id"] == "node-1"
    assert res["budgets"] == {"alpaca": 2}
    assert res["datasets"] == {"alpaca": 0}
