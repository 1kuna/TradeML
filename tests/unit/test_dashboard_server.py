from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

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


def test_dashboard_server_serves_hud_and_game_snapshot(tmp_path: Path, monkeypatch) -> None:
    settings = _test_settings(tmp_path)
    monkeypatch.setattr(
        dashboard_server,
        "collect_dashboard_game_snapshot",
        lambda resolved: {
            "node": {
                "label": "rpi-01",
                "running": True,
                "uptime_seconds": 3600.0,
                "rows_per_min": 142.5,
                "requests_per_min": 18.2,
                "rows_total": 12_438_201,
                "expected_datapoints": 14_000_000,
                "coverage_percent": 72.4,
                "training_ready_percent": 72.4,
                "eta_minutes": 192.0,
                "pending_tasks": 5,
                "failed_tasks": 0,
                "activity_pulse": True,
            },
            "training": {
                "host": "mac-mini",
                "state": "idle",
                "latest_rank_ic": 0.041,
                "best_rank_ic": 0.048,
                "streak_go": 3,
                "total_runs": 8,
                "last_decisions": [
                    {"run_ts": "2026-04-08", "decision": "GO", "mean_rank_ic": 0.041},
                ],
                "sparkline": [0.012, 0.018, 0.025, 0.033, 0.041],
            },
            "missions": [
                {"key": "canonical_bars", "label": "Bars", "status": "in_progress", "percent": 72.4, "blocking": True, "eta_minutes": 192.0, "remaining_units": 12},
            ],
            "phase1_gate": {"ready": False, "percent": 72.4, "blockers": ["bars_incomplete"], "freeze_cutoff": "2026-03-09"},
            "cluster": {
                "workers": [{"worker_id": "rpi-01", "last_heartbeat": "2026-04-10T16:30:00+00:00"}],
                "active_workers": [{"worker_id": "rpi-01", "last_heartbeat": "2026-04-10T16:30:00+00:00"}],
            },
            "updated_at": "2026-04-10T16:30:00+00:00",
        },
    )
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
    monkeypatch.setattr(
        dashboard_server,
        "collect_dashboard_setup_snapshot",
        lambda resolved, *, cluster_snapshot=None: {
            "cluster": {
                "workers": [{"worker_id": "rpi-01", "last_heartbeat": "2026-04-10T16:30:00+00:00"}],
                "active_workers": [{"worker_id": "rpi-01", "last_heartbeat": "2026-04-10T16:30:00+00:00"}],
            },
            "systemd": {"scope": "user"},
            "nas": {"share": "//nas/trademl"},
            "provider_contracts": [],
        },
    )
    httpd = dashboard_server.create_dashboard_server("127.0.0.1", 0, settings=settings)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        base_url = f"http://127.0.0.1:{httpd.server_port}"
        html = urllib.request.urlopen(f"{base_url}/", timeout=5).read().decode("utf-8")
        operator_html = urllib.request.urlopen(f"{base_url}/operator", timeout=5).read().decode("utf-8")
        game_payload = json.loads(urllib.request.urlopen(f"{base_url}/api/game", timeout=5).read().decode("utf-8"))
        live_payload = json.loads(urllib.request.urlopen(f"{base_url}/api/live", timeout=5).read().decode("utf-8"))
        setup_payload = json.loads(urllib.request.urlopen(f"{base_url}/api/setup", timeout=5).read().decode("utf-8"))
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)

    assert "<title>TradeML HQ</title>" in html
    assert "TradeML HQ" in html
    assert "Mission Control" in html
    assert "new EventSource('/api/game/stream')" in html
    assert "The Collector" in html
    assert "The Brain" in html
    assert "<title>TradeML Operator Dashboard</title>" in operator_html
    assert 'data-section="status"' in operator_html
    assert 'data-section="budgets"' in operator_html
    assert 'data-section="setup"' in operator_html
    assert 'data-section="logs"' in operator_html
    assert "const sections = ['status', 'budgets', 'setup', 'logs'];" in operator_html
    assert game_payload["node"]["label"] == "rpi-01"
    assert game_payload["training"]["streak_go"] == 3
    assert game_payload["phase1_gate"]["freeze_cutoff"] == "2026-03-09"
    assert game_payload["missions"][0]["label"] == "Bars"
    assert setup_payload["cluster"]["active_workers"][0]["worker_id"] == "rpi-01"
    assert live_payload["runtime"]["pid"] == 123
    assert live_payload["training_readiness"]["phase1"]["ready"] is True


def test_dashboard_server_setup_reads_fresh_cluster_snapshot(tmp_path: Path, monkeypatch) -> None:
    settings = _test_settings(tmp_path)
    setup_calls: list[object] = []
    monkeypatch.setattr(
        dashboard_server,
        "collect_dashboard_game_snapshot",
        lambda _: {
            "node": {"label": "rpi-01", "running": True, "cluster_active": True, "process_running": True},
            "training": {"state": "idle", "last_decisions": [], "sparkline": [], "total_runs": 0, "streak_go": 0},
            "missions": [],
            "phase1_gate": {"ready": True, "percent": 100.0, "blockers": [], "freeze_cutoff": "2026-03-09"},
            "cluster": {
                "workers": [{"worker_id": "rpi-01", "last_heartbeat": "2026-04-10T16:30:00+00:00"}],
                "active_workers": [{"worker_id": "rpi-01", "last_heartbeat": "2026-04-10T16:30:00+00:00"}],
            },
            "updated_at": "2026-04-10T16:30:00+00:00",
        },
    )
    monkeypatch.setattr(
        dashboard_server,
        "collect_dashboard_live_snapshot",
        lambda _: {
            "runtime": {"running": True},
            "collection_status": {},
            "training_readiness": {},
            "planner_eta": {},
            "budget_summary": {},
            "vendor_throughput": {"rows": []},
        },
    )
    monkeypatch.setattr(
        dashboard_server,
        "collect_dashboard_setup_snapshot",
        lambda resolved, *, cluster_snapshot=None: setup_calls.append(cluster_snapshot) or {
            "cluster": {
                "workers": [{"worker_id": "fresh-worker", "last_heartbeat": "2026-04-10T16:35:00+00:00"}],
                "active_workers": [{"worker_id": "fresh-worker", "last_heartbeat": "2026-04-10T16:35:00+00:00"}],
            },
            "systemd": {"scope": "user"},
            "nas": {"share": "//nas/trademl"},
            "provider_contracts": [],
        },
    )

    httpd = dashboard_server.create_dashboard_server("127.0.0.1", 0, settings=settings)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        base_url = f"http://127.0.0.1:{httpd.server_port}"
        setup_payload = json.loads(urllib.request.urlopen(f"{base_url}/api/setup", timeout=5).read().decode("utf-8"))
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)

    assert setup_calls == [None]
    assert setup_payload["cluster"]["active_workers"][0]["worker_id"] == "fresh-worker"


def test_dashboard_server_caches_game_snapshot_between_quick_requests(tmp_path: Path, monkeypatch) -> None:
    settings = _test_settings(tmp_path)
    calls = {"count": 0}

    def fake_game_snapshot(resolved: NodeSettings) -> dict[str, object]:
        calls["count"] += 1
        return {
            "node": {"label": f"rpi-{calls['count']}", "running": True},
            "training": {"state": "idle", "last_decisions": [], "sparkline": [], "total_runs": 0, "streak_go": 0},
            "missions": [],
            "phase1_gate": {"ready": False, "percent": 0.0, "blockers": []},
            "updated_at": "2026-04-10T16:30:00+00:00",
        }

    monkeypatch.setattr(dashboard_server, "collect_dashboard_game_snapshot", fake_game_snapshot)

    httpd = dashboard_server.create_dashboard_server("127.0.0.1", 0, settings=settings)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        base_url = f"http://127.0.0.1:{httpd.server_port}"
        payload_a = json.loads(urllib.request.urlopen(f"{base_url}/api/game", timeout=5).read().decode("utf-8"))
        payload_b = json.loads(urllib.request.urlopen(f"{base_url}/api/game", timeout=5).read().decode("utf-8"))
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)

    assert calls["count"] == 1
    assert payload_a["node"]["label"] == "rpi-1"
    assert payload_b["node"]["label"] == "rpi-1"
    assert payload_a["meta"]["source"] == "fresh"
    assert payload_a["meta"]["stale"] is False


def test_dashboard_server_returns_503_when_no_snapshot_is_available(tmp_path: Path, monkeypatch) -> None:
    settings = _test_settings(tmp_path)

    def fail_snapshot(_: NodeSettings) -> dict[str, object]:
        raise RuntimeError("boom")

    monkeypatch.setattr(dashboard_server, "collect_dashboard_game_snapshot", fail_snapshot)
    monkeypatch.setattr(dashboard_server, "collect_dashboard_live_snapshot", fail_snapshot)

    httpd = dashboard_server.create_dashboard_server("127.0.0.1", 0, settings=settings)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        base_url = f"http://127.0.0.1:{httpd.server_port}"
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            urllib.request.urlopen(f"{base_url}/api/game", timeout=5)
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)

    payload = json.loads(excinfo.value.read().decode("utf-8"))
    assert excinfo.value.code == 503
    assert payload["error"] == "snapshot_not_ready"
    assert payload["channel"] == "game"


def test_dashboard_server_serves_disk_cached_snapshot_when_startup_build_fails(tmp_path: Path, monkeypatch) -> None:
    settings = _test_settings(tmp_path)
    settings.local_state.mkdir(parents=True, exist_ok=True)
    cached_payload = {
        "node": {"label": "disk-rpi", "running": True},
        "training": {"state": "idle", "last_decisions": [], "sparkline": [], "total_runs": 0, "streak_go": 0},
        "missions": [],
        "phase1_gate": {"ready": True, "percent": 100.0, "blockers": [], "freeze_cutoff": "2026-03-09"},
        "updated_at": "2026-04-10T16:30:00+00:00",
        "meta": {
            "built_at": "2026-04-10T16:30:00+00:00",
            "build_started_at": "2026-04-10T16:29:59+00:00",
            "build_ms": 12.5,
            "stale": False,
            "source": "fresh",
            "error": None,
            "version": 1,
        },
    }
    (settings.local_state / "dashboard_game_snapshot.json").write_text(json.dumps(cached_payload), encoding="utf-8")

    monkeypatch.setattr(dashboard_server, "collect_dashboard_game_snapshot", lambda _: (_ for _ in ()).throw(RuntimeError("fail")))
    monkeypatch.setattr(
        dashboard_server,
        "collect_dashboard_live_snapshot",
        lambda _: {"runtime": {"running": True}, "collection_status": {}, "training_readiness": {}, "planner_eta": {}, "budget_summary": {}, "vendor_throughput": {"rows": []}},
    )

    httpd = dashboard_server.create_dashboard_server("127.0.0.1", 0, settings=settings)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        base_url = f"http://127.0.0.1:{httpd.server_port}"
        payload = json.loads(urllib.request.urlopen(f"{base_url}/api/game", timeout=5).read().decode("utf-8"))
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)

    assert payload["node"]["label"] == "disk-rpi"
    assert payload["meta"]["source"] == "disk_cache"
    assert payload["meta"]["stale"] is True
    assert payload["meta"]["error"] == "fail"


def test_dashboard_server_refresh_failure_keeps_last_good_snapshot(tmp_path: Path, monkeypatch) -> None:
    settings = _test_settings(tmp_path)
    calls = {"count": 0}

    def fake_game_snapshot(_: NodeSettings) -> dict[str, object]:
        calls["count"] += 1
        if calls["count"] == 1:
            return {
                "node": {"label": "fresh-rpi", "running": True},
                "training": {"state": "idle", "last_decisions": [], "sparkline": [], "total_runs": 0, "streak_go": 0},
                "missions": [],
                "phase1_gate": {"ready": False, "percent": 10.0, "blockers": [], "freeze_cutoff": None},
                "updated_at": "2026-04-10T16:30:00+00:00",
            }
        raise RuntimeError("refresh failed")

    monkeypatch.setattr(dashboard_server, "collect_dashboard_game_snapshot", fake_game_snapshot)
    monkeypatch.setattr(
        dashboard_server,
        "collect_dashboard_live_snapshot",
        lambda _: {"runtime": {"running": True}, "collection_status": {}, "training_readiness": {}, "planner_eta": {}, "budget_summary": {}, "vendor_throughput": {"rows": []}},
    )

    httpd = dashboard_server.create_dashboard_server("127.0.0.1", 0, settings=settings)
    httpd.snapshot_manager.refresh_once("game")
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        base_url = f"http://127.0.0.1:{httpd.server_port}"
        payload = json.loads(urllib.request.urlopen(f"{base_url}/api/game", timeout=5).read().decode("utf-8"))
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)

    assert calls["count"] == 2
    assert payload["node"]["label"] == "fresh-rpi"
    assert payload["meta"]["source"] == "stale_last_good"
    assert payload["meta"]["stale"] is True
    assert payload["meta"]["error"] == "refresh failed"


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
    monkeypatch.setattr(
        dashboard_server,
        "collect_dashboard_status_snapshot",
        lambda resolved: {
            "runtime": {"running": True},
            "collection_status": {"coverage_percent": 99.0},
            "training_readiness": {"phase1": {"ready": True}},
            "training_status": {"phase1": {"status": "idle"}},
            "experiment_summary": {},
            "default_training_target": {},
            "training_targets": [],
            "health": {"repair_tasks": {"counts": {}, "remaining_units": 0}},
        },
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


def test_dashboard_server_caches_status_snapshot_between_quick_requests(tmp_path: Path, monkeypatch) -> None:
    settings = _test_settings(tmp_path)
    calls = {"count": 0}

    monkeypatch.setattr(
        dashboard_server,
        "collect_dashboard_game_snapshot",
        lambda _: {
            "node": {"label": "rpi-1", "running": True},
            "training": {"state": "idle", "last_decisions": [], "sparkline": [], "total_runs": 0, "streak_go": 0},
            "missions": [],
            "phase1_gate": {"ready": True, "percent": 100.0, "blockers": [], "freeze_cutoff": "2026-03-09"},
            "updated_at": "2026-04-10T16:30:00+00:00",
        },
    )
    monkeypatch.setattr(
        dashboard_server,
        "collect_dashboard_live_snapshot",
        lambda _: {
            "runtime": {"running": True},
            "collection_status": {},
            "training_readiness": {},
            "planner_eta": {},
            "budget_summary": {},
            "vendor_throughput": {"rows": []},
        },
    )

    def fake_status_snapshot(_: NodeSettings) -> dict[str, object]:
        calls["count"] += 1
        return {
            "runtime": {"running": True},
            "collection_status": {"coverage_percent": 99.0},
            "training_readiness": {"phase1": {"ready": True}},
            "training_status": {"phase1": {"status": "idle"}},
            "experiment_summary": {"experiment_id": f"exp-{calls['count']}"},
            "default_training_target": {"name": "workstation-remote"},
            "training_targets": [],
            "health": {"repair_tasks": {"counts": {}, "remaining_units": 0}},
        }

    monkeypatch.setattr(dashboard_server, "collect_dashboard_status_snapshot", fake_status_snapshot)

    httpd = dashboard_server.create_dashboard_server("127.0.0.1", 0, settings=settings)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        base_url = f"http://127.0.0.1:{httpd.server_port}"
        payload_a = json.loads(urllib.request.urlopen(f"{base_url}/api/status", timeout=5).read().decode("utf-8"))
        payload_b = json.loads(urllib.request.urlopen(f"{base_url}/api/status", timeout=5).read().decode("utf-8"))
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)

    assert calls["count"] == 1
    assert payload_a["experiment_summary"]["experiment_id"] == "exp-1"
    assert payload_b["experiment_summary"]["experiment_id"] == "exp-1"
    assert "snapshot_health" in payload_a


def test_dashboard_server_does_not_build_status_until_status_is_requested(tmp_path: Path, monkeypatch) -> None:
    settings = _test_settings(tmp_path)
    calls = {"status": 0}

    monkeypatch.setattr(
        dashboard_server,
        "collect_dashboard_game_snapshot",
        lambda _: {
            "node": {"label": "rpi-1", "running": True},
            "training": {"state": "idle", "last_decisions": [], "sparkline": [], "total_runs": 0, "streak_go": 0},
            "missions": [],
            "phase1_gate": {"ready": True, "percent": 100.0, "blockers": [], "freeze_cutoff": "2026-03-09"},
            "updated_at": "2026-04-10T16:30:00+00:00",
        },
    )
    monkeypatch.setattr(
        dashboard_server,
        "collect_dashboard_live_snapshot",
        lambda _: {
            "runtime": {"running": True},
            "collection_status": {},
            "training_readiness": {},
            "planner_eta": {},
            "budget_summary": {},
            "vendor_throughput": {"rows": []},
        },
    )

    def fake_status_snapshot(_: NodeSettings) -> dict[str, object]:
        calls["status"] += 1
        return {
            "runtime": {"running": True},
            "collection_status": {"coverage_percent": 99.0},
            "training_readiness": {"phase1": {"ready": True}},
            "training_status": {"phase1": {"status": "idle"}},
            "experiment_summary": {"experiment_id": "exp-status"},
            "default_training_target": {"name": "workstation-remote"},
            "training_targets": [],
            "health": {"repair_tasks": {"counts": {}, "remaining_units": 0}},
        }

    monkeypatch.setattr(dashboard_server, "collect_dashboard_status_snapshot", fake_status_snapshot)

    httpd = dashboard_server.create_dashboard_server("127.0.0.1", 0, settings=settings)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        base_url = f"http://127.0.0.1:{httpd.server_port}"
        game_payload = json.loads(urllib.request.urlopen(f"{base_url}/api/game", timeout=5).read().decode("utf-8"))
        status_payload = json.loads(urllib.request.urlopen(f"{base_url}/api/status", timeout=5).read().decode("utf-8"))
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)

    assert game_payload["node"]["label"] == "rpi-1"
    assert calls["status"] == 1
    assert status_payload["experiment_summary"]["experiment_id"] == "exp-status"


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


def test_dispatch_action_routes_research_actions(tmp_path: Path, monkeypatch) -> None:
    settings = _test_settings(tmp_path)
    monkeypatch.setattr(dashboard_server, "start_research_supervisor", lambda settings, *, program_path, poll_seconds=None, detach=True: {"action": "start", "program_path": program_path, "detach": detach})
    monkeypatch.setattr(dashboard_server, "pause_research", lambda settings, *, program_id: {"action": "pause", "program_id": program_id})
    monkeypatch.setattr(dashboard_server, "resume_research", lambda settings, *, program_id: {"action": "resume", "program_id": program_id})
    monkeypatch.setattr(dashboard_server, "stop_research", lambda settings, *, program_id: {"action": "stop", "program_id": program_id})
    monkeypatch.setattr(dashboard_server, "research_status", lambda settings, *, program_id: {"action": "status", "program_id": program_id})
    monkeypatch.setattr(dashboard_server, "research_review_packet", lambda settings, *, program_id: {"action": "review", "program_id": program_id})
    monkeypatch.setattr(
        dashboard_server,
        "steer_research",
        lambda settings, *, program_id, prefer_architecture_families=None, avoid_architecture_families=None, prefer_data_families=None, avoid_data_families=None, freeze_phase=None, force_pivot=None, exploration_breadth=None: {
            "action": "steer",
            "program_id": program_id,
            "prefer_architecture_families": prefer_architecture_families,
            "avoid_data_families": avoid_data_families,
            "freeze_phase": freeze_phase,
            "force_pivot": force_pivot,
            "exploration_breadth": exploration_breadth,
        },
    )

    assert dashboard_server.dispatch_dashboard_action(settings, "research-start", {"program_path": "configs/research/perpetual_macmini.yml", "detach": True})["action"] == "start"
    assert dashboard_server.dispatch_dashboard_action(settings, "research-pause", {"program_id": "perpetual"})["action"] == "pause"
    assert dashboard_server.dispatch_dashboard_action(settings, "research-resume", {"program_id": "perpetual"})["action"] == "resume"
    assert dashboard_server.dispatch_dashboard_action(settings, "research-stop", {"program_id": "perpetual"})["action"] == "stop"
    assert dashboard_server.dispatch_dashboard_action(settings, "research-status", {"program_id": "perpetual"})["action"] == "status"
    assert dashboard_server.dispatch_dashboard_action(settings, "research-review-packet", {"program_id": "perpetual"})["action"] == "review"
    steered = dashboard_server.dispatch_dashboard_action(
        settings,
        "research-steer",
        {
            "program_id": "perpetual",
            "prefer_architecture_families": ["tree_challenger"],
            "avoid_data_families": ["price_plus_liquidity"],
            "freeze_phase": 1,
            "force_pivot": True,
            "exploration_breadth": "high",
        },
    )
    assert steered["action"] == "steer"
    assert steered["program_id"] == "perpetual"
    assert steered["freeze_phase"] == 1
