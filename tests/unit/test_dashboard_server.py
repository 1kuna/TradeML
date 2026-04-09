from __future__ import annotations

import json
import threading
import urllib.request
from pathlib import Path

from trademl.dashboard import server as dashboard_server
from trademl.dashboard.controller import NodeSettings


def _test_settings(tmp_path: Path) -> NodeSettings:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    return NodeSettings(
        repo_root=tmp_path,
        workspace_root=workspace,
        config_path=workspace / "node.yml",
        env_path=workspace / ".env",
        local_state=workspace / "control",
        nas_mount=tmp_path / "nas",
        nas_share="//nas/trademl",
        collection_time_et="16:30",
        maintenance_hour_local=2,
        worker_id="worker-1",
    )


def test_dashboard_server_serves_index_and_live_snapshot(tmp_path: Path, monkeypatch) -> None:
    settings = _test_settings(tmp_path)
    monkeypatch.setattr(
        dashboard_server,
        "collect_dashboard_live_snapshot",
        lambda resolved: {
            "runtime": {"running": True, "pid": 123, "uptime_seconds": 9.0},
            "collection_status": {
                "coverage_percent": 97.5,
                "canonical_completed_units": 10,
                "raw_vendor_rows": 20,
                "canonical_remaining_units": 3,
                "training_critical_percent": 98.2,
                "pending_tasks": 2,
                "failed_tasks": 0,
            },
            "training_readiness": {"phase1": {"ready": True, "blockers": []}, "freeze_cutoff": {"date": "2026-03-09"}},
            "planner_eta": {"canonical_bars": {"eta_minutes": 12}},
            "budget_summary": {"checked_at": "2026-04-09T12:00:00+00:00"},
            "vendor_throughput": {"rows": []},
        },
    )
    httpd = dashboard_server.create_dashboard_server("127.0.0.1", 0, settings=settings)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        base_url = f"http://127.0.0.1:{httpd.server_port}"
        html = urllib.request.urlopen(f"{base_url}/", timeout=5).read().decode("utf-8")
        live_payload = json.loads(urllib.request.urlopen(f"{base_url}/api/live", timeout=5).read().decode("utf-8"))
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)

    assert "new EventSource('/api/live/stream')" in html
    assert "TradeML Operator Dashboard" in html
    assert live_payload["runtime"]["pid"] == 123
    assert live_payload["training_readiness"]["phase1"]["ready"] is True


def test_dashboard_server_answers_head_and_streamlit_probe_paths(tmp_path: Path, monkeypatch) -> None:
    settings = _test_settings(tmp_path)
    monkeypatch.setattr(
        dashboard_server,
        "collect_dashboard_live_snapshot",
        lambda resolved: {
            "runtime": {"running": True},
            "collection_status": {"coverage_percent": 99.0},
            "training_readiness": {"phase1": {"ready": True}},
            "planner_eta": {},
            "budget_summary": {},
            "vendor_throughput": {"rows": []},
        },
    )
    httpd = dashboard_server.create_dashboard_server("127.0.0.1", 0, settings=settings)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        base_url = f"http://127.0.0.1:{httpd.server_port}"
        head_request = urllib.request.Request(f"{base_url}/", method="HEAD")
        health_payload = json.loads(urllib.request.urlopen(f"{base_url}/_stcore/health", timeout=5).read().decode("utf-8"))
        host_config_payload = json.loads(urllib.request.urlopen(f"{base_url}/_stcore/host-config", timeout=5).read().decode("utf-8"))
        head_response = urllib.request.urlopen(head_request, timeout=5)
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)

    assert head_response.status == 200
    assert health_payload == {"ok": True}
    assert host_config_payload["useExternalAuthToken"] is False


def test_dispatch_action_routes_restart_with_passphrase(tmp_path: Path, monkeypatch) -> None:
    settings = _test_settings(tmp_path)
    seen: dict[str, object] = {}

    def fake_restart(resolved: NodeSettings, *, passphrase: str | None = None) -> dict[str, object]:
        seen["settings"] = resolved
        seen["passphrase"] = passphrase
        return {"action": "restart"}

    monkeypatch.setattr(dashboard_server, "restart_node", fake_restart)

    result = dashboard_server.dispatch_dashboard_action(settings, "restart-node", {"passphrase": "pw"})

    assert result == {"action": "restart"}
    assert seen["settings"] is settings
    assert seen["passphrase"] == "pw"
