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
    assert "Quick Glance" in html
    assert "Best Model So Far" in html
    assert "Readiness Detail" not in html
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
    monkeypatch.setattr(
        dashboard_server,
        "collect_dashboard_health_snapshot",
        lambda resolved: {"repair_tasks": {"counts": {}, "remaining_units": 0}},
    )
    httpd = dashboard_server.create_dashboard_server("127.0.0.1", 0, settings=settings)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        base_url = f"http://127.0.0.1:{httpd.server_port}"
        head_request = urllib.request.Request(f"{base_url}/", method="HEAD")
        dashboard_health = json.loads(urllib.request.urlopen(f"{base_url}/api/health", timeout=5).read().decode("utf-8"))
        health_payload = json.loads(urllib.request.urlopen(f"{base_url}/_stcore/health", timeout=5).read().decode("utf-8"))
        host_config_payload = json.loads(urllib.request.urlopen(f"{base_url}/_stcore/host-config", timeout=5).read().decode("utf-8"))
        head_response = urllib.request.urlopen(head_request, timeout=5)
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)

    assert head_response.status == 200
    assert "repair_tasks" in dashboard_health
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


def test_dispatch_action_routes_bootstrap_ledger(tmp_path: Path, monkeypatch) -> None:
    settings = _test_settings(tmp_path)
    seen: dict[str, object] = {}

    def fake_bootstrap(resolved: NodeSettings) -> dict[str, object]:
        seen["settings"] = resolved
        return {"bootstrapped": 12}

    monkeypatch.setattr(dashboard_server, "bootstrap_canonical_ledger", fake_bootstrap)

    result = dashboard_server.dispatch_dashboard_action(settings, "bootstrap-ledger", {})

    assert result == {"bootstrapped": 12}
    assert seen["settings"] is settings


def test_dispatch_action_routes_canonical_repair_payload(tmp_path: Path, monkeypatch) -> None:
    settings = _test_settings(tmp_path)
    seen: dict[str, object] = {}

    def fake_repair(
        resolved: NodeSettings,
        *,
        trading_date: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        symbol: str | None = None,
        verify_only: bool = False,
    ) -> dict[str, object]:
        seen["settings"] = resolved
        seen["trading_date"] = trading_date
        seen["start_date"] = start_date
        seen["end_date"] = end_date
        seen["symbol"] = symbol
        seen["verify_only"] = verify_only
        return {"repairable": 4}

    monkeypatch.setattr(dashboard_server, "repair_canonical_backlog", fake_repair)

    result = dashboard_server.dispatch_dashboard_action(
        settings,
        "repair-canonical",
        {
            "trading_date": "2026-04-10",
            "start_date": "2026-03-01",
            "end_date": "2026-04-10",
            "symbol": "TECK",
            "verify_only": True,
        },
    )

    assert result == {"repairable": 4}
    assert seen == {
        "settings": settings,
        "trading_date": "2026-04-10",
        "start_date": "2026-03-01",
        "end_date": "2026-04-10",
        "symbol": "TECK",
        "verify_only": True,
    }


def test_dispatch_action_routes_training_actions(tmp_path: Path, monkeypatch) -> None:
    settings = _test_settings(tmp_path)
    monkeypatch.setattr(dashboard_server, "training_preflight_status", lambda settings, *, phase, target=None: {"action": "preflight", "phase": phase, "target": target})
    monkeypatch.setattr(dashboard_server, "start_training_run", lambda settings, *, phase, report_date=None, target=None: {"action": "start", "phase": phase, "target": target, "report_date": report_date})
    monkeypatch.setattr(dashboard_server, "stop_training_run", lambda settings, *, phase, target=None: {"action": "stop", "phase": phase, "target": target})

    preflight = dashboard_server.dispatch_dashboard_action(settings, "train-preflight", {"phase": 1, "target": "workstation-remote"})
    start = dashboard_server.dispatch_dashboard_action(settings, "train-start", {"phase": 1, "target": "workstation-remote", "report_date": "2026-04-02"})
    stop = dashboard_server.dispatch_dashboard_action(settings, "train-stop", {"phase": 1, "target": "workstation-remote"})

    assert preflight["action"] == "preflight"
    assert start["report_date"] == "2026-04-02"
    assert stop["action"] == "stop"


def test_dispatch_action_routes_experiment_actions(tmp_path: Path, monkeypatch) -> None:
    settings = _test_settings(tmp_path)
    monkeypatch.setattr(dashboard_server, "start_experiment_supervisor", lambda settings, *, spec_path, poll_seconds=None, detach=True: {"action": "supervise", "spec_path": spec_path, "detach": detach})
    monkeypatch.setattr(dashboard_server, "pause_experiments", lambda settings, *, experiment_id: {"action": "pause", "experiment_id": experiment_id})
    monkeypatch.setattr(dashboard_server, "resume_experiments", lambda settings, *, experiment_id: {"action": "resume", "experiment_id": experiment_id})
    monkeypatch.setattr(dashboard_server, "stop_experiments", lambda settings, *, experiment_id: {"action": "stop", "experiment_id": experiment_id})
    monkeypatch.setattr(dashboard_server, "evaluate_experiments", lambda settings, *, experiment_id: {"action": "evaluate", "experiment_id": experiment_id})
    monkeypatch.setattr(dashboard_server, "backtest_experiments", lambda settings, *, experiment_id: {"action": "backtest", "experiment_id": experiment_id})
    monkeypatch.setattr(dashboard_server, "propose_experiment_family", lambda settings, *, experiment_id: {"action": "propose", "experiment_id": experiment_id})

    assert dashboard_server.dispatch_dashboard_action(settings, "experiments-supervise", {"spec_path": "configs/experiments/demo.yml", "detach": True})["action"] == "supervise"
    assert dashboard_server.dispatch_dashboard_action(settings, "experiments-pause", {"experiment_id": "phase1"})["action"] == "pause"
    assert dashboard_server.dispatch_dashboard_action(settings, "experiments-resume", {"experiment_id": "phase1"})["action"] == "resume"
    assert dashboard_server.dispatch_dashboard_action(settings, "experiments-stop", {"experiment_id": "phase1"})["action"] == "stop"
    assert dashboard_server.dispatch_dashboard_action(settings, "experiments-evaluate", {"experiment_id": "phase1"})["action"] == "evaluate"
    assert dashboard_server.dispatch_dashboard_action(settings, "experiments-backtest", {"experiment_id": "phase1"})["action"] == "backtest"
    assert dashboard_server.dispatch_dashboard_action(settings, "experiments-propose-next", {"experiment_id": "phase1"})["action"] == "propose"
