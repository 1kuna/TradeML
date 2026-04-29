"""HTTP dashboard server with client-side live updates."""

from __future__ import annotations

import argparse
import json
import logging
import threading
import time
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from trademl.dashboard.actions import DashboardAction, dispatch_dashboard_action as _dispatch_dashboard_action
from trademl.dashboard.assets import HTML_PAGE, OPERATOR_HTML_PAGE
from trademl.dashboard.controller import (
    advance_collection_stage,
    backtest_experiments,
    bootstrap_canonical_ledger,
    collect_dashboard_game_snapshot,
    collect_dashboard_health_snapshot,
    collect_dashboard_live_snapshot,
    collect_dashboard_logs_snapshot,
    collect_dashboard_setup_snapshot,
    collect_dashboard_status_snapshot,
    force_release_lease,
    install_service,
    join_cluster,
    leave_cluster,
    pause_experiments,
    pause_research,
    persist_node_settings,
    propose_experiment_family,
    rebuild_cluster_state,
    replan_coverage,
    research_review_packet,
    research_status,
    resolve_node_settings,
    resume_experiments,
    resume_research,
    restart_node,
    rotate_cluster_passphrase,
    lane_health,
    evaluate_experiments,
    repair_canonical_backlog,
    repair_status,
    run_vendor_audit,
    start_experiment_supervisor,
    start_node,
    start_research_supervisor,
    start_training_run,
    stop_node,
    stop_experiments,
    stop_research,
    stop_training_run,
    steer_research,
    training_preflight_status,
    training_runtime_logs,
    training_runtime_status,
    uninstall_worker,
    update_worker,
    update_cluster_secrets,
    verify_recent_canonical_dates,
    reset_worker,
    NodeSettings,
)
from trademl.dashboard.snapshots import DashboardSnapshotManager

LOGGER = logging.getLogger(__name__)


class DashboardHTTPServer(ThreadingHTTPServer):
    """HTTP server that holds resolved TradeML dashboard settings."""

    daemon_threads = True

    def __init__(self, server_address: tuple[str, int], *, settings: NodeSettings) -> None:
        super().__init__(server_address, DashboardRequestHandler)
        self.settings = settings
        self.snapshot_manager = DashboardSnapshotManager(
            settings=settings,
            builders={
                "game": collect_dashboard_game_snapshot,
                "live": collect_dashboard_live_snapshot,
                "status": collect_dashboard_status_snapshot,
            },
        )

    def server_close(self) -> None:
        self.snapshot_manager.close()
        super().server_close()


class DashboardRequestHandler(BaseHTTPRequestHandler):
    """Serve the operator dashboard HTML and JSON action endpoints."""

    server: DashboardHTTPServer
    stream_interval_seconds = DashboardSnapshotManager.stream_interval_seconds

    def do_HEAD(self) -> None:  # noqa: N802
        html_paths = {"/", "/operator"}
        json_paths = {
            "/api/live",
            "/api/game",
            "/api/status",
            "/api/health",
            "/api/setup",
            "/api/logs",
            "/_stcore/health",
            "/_stcore/host-config",
        }
        if self.path in html_paths or self.path in json_paths:
            content_type = "text/html; charset=utf-8" if self.path in html_paths else "application/json; charset=utf-8"
            self._write_headers(status=HTTPStatus.OK, content_type=content_type)
            return
        self.send_error(HTTPStatus.NOT_FOUND, "not found")

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/":
            self._write_html(HTML_PAGE)
            return
        if self.path == "/operator":
            self._write_html(OPERATOR_HTML_PAGE)
            return
        if self.path == "/_stcore/health":
            self._write_json({"ok": True})
            return
        if self.path == "/_stcore/host-config":
            self._write_json({"useExternalAuthToken": False, "enableCustomParentMessages": False})
            return
        if self.path == "/api/game":
            status, payload = self.server.snapshot_manager.get_latest_or_not_ready("game")
            self._write_json(payload, status=status)
            return
        if self.path == "/api/game/stream":
            self._serve_game_stream()
            return
        if self.path == "/api/live":
            status, payload = self.server.snapshot_manager.get_latest_or_not_ready("live")
            self._write_json(payload, status=status)
            return
        if self.path == "/api/status":
            status, payload = self.server.snapshot_manager.get_latest_or_not_ready("status")
            payload = dict(payload)
            payload["snapshot_health"] = self.server.snapshot_manager.health_summary()
            self._write_json(payload, status=status)
            return
        if self.path == "/api/health":
            self._write_json(collect_dashboard_health_snapshot(self.server.settings))
            return
        if self.path == "/api/setup":
            self._write_json(collect_dashboard_setup_snapshot(self.server.settings))
            return
        if self.path == "/api/logs":
            self._write_json(collect_dashboard_logs_snapshot(self.server.settings))
            return
        if self.path == "/api/live/stream":
            self._serve_live_stream()
            return
        self.send_error(HTTPStatus.NOT_FOUND, "not found")

    def do_POST(self) -> None:  # noqa: N802
        if not self.path.startswith("/api/actions/"):
            self.send_error(HTTPStatus.NOT_FOUND, "not found")
            return
        action = self.path.removeprefix("/api/actions/")
        try:
            payload = self._read_json_body()
            result = dispatch_dashboard_action(self.server.settings, action, payload)
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("dashboard_action_failed action=%s", action)
            self._write_json({"ok": False, "error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
            return
        self._write_json({"ok": True, "result": result})

    def log_message(self, fmt: str, *args: Any) -> None:
        LOGGER.info("dashboard_http " + fmt, *args)

    def _read_json_body(self) -> dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", "0") or "0")
        if content_length <= 0:
            return {}
        raw = self.rfile.read(content_length)
        if not raw:
            return {}
        return json.loads(raw.decode("utf-8"))

    def _write_headers(self, *, status: HTTPStatus, content_type: str, extra_headers: dict[str, str] | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Pragma", "no-cache")
        if extra_headers:
            for key, value in extra_headers.items():
                self.send_header(key, value)
        self.end_headers()

    def _write_json(self, payload: dict[str, Any], *, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, default=str).encode("utf-8")
        self._write_headers(status=status, content_type="application/json; charset=utf-8", extra_headers={"Content-Length": str(len(body))})
        self.wfile.write(body)
        self.wfile.flush()

    def _write_html(self, html: str) -> None:
        body = html.encode("utf-8")
        self._write_headers(status=HTTPStatus.OK, content_type="text/html; charset=utf-8", extra_headers={"Content-Length": str(len(body))})
        self.wfile.write(body)
        self.wfile.flush()

    def _serve_live_stream(self) -> None:
        self._write_headers(
            status=HTTPStatus.OK,
            content_type="text/event-stream; charset=utf-8",
            extra_headers={"Connection": "keep-alive", "X-Accel-Buffering": "no"},
        )
        while True:
            try:
                payload = json.dumps(
                    self.server.snapshot_manager.get_latest("live")
                    or {"ok": False, "error": "snapshot_not_ready", "channel": "live"},
                    default=str,
                )
                self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
                self.wfile.flush()
                time.sleep(self.stream_interval_seconds)
            except (BrokenPipeError, ConnectionResetError):
                return

    def _serve_game_stream(self) -> None:
        self._write_headers(
            status=HTTPStatus.OK,
            content_type="text/event-stream; charset=utf-8",
            extra_headers={"Connection": "keep-alive", "X-Accel-Buffering": "no"},
        )
        while True:
            try:
                payload = json.dumps(
                    self.server.snapshot_manager.get_latest("game")
                    or {"ok": False, "error": "snapshot_not_ready", "channel": "game"},
                    default=str,
                )
                self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
                self.wfile.flush()
                time.sleep(self.stream_interval_seconds)
            except (BrokenPipeError, ConnectionResetError):
                return


def dispatch_dashboard_action(settings: NodeSettings, action: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Run a dashboard action endpoint against controller helpers."""
    return _dispatch_dashboard_action(settings, action, payload, _DASHBOARD_ACTIONS)


def _cluster_secret_update(settings: NodeSettings, payload: dict[str, Any]) -> dict[str, Any]:
    key = str(payload.get("key") or "").strip()
    if not key:
        raise ValueError("secret key is required")
    passphrase = _optional_str(payload.get("passphrase"))
    if not passphrase:
        raise ValueError("cluster passphrase is required")
    return update_cluster_secrets(settings, passphrase=passphrase, updates={key: str(payload.get("value") or "")})


def _cluster_passphrase_rotation(settings: NodeSettings, payload: dict[str, Any]) -> dict[str, Any]:
    old_passphrase = _optional_str(payload.get("old_passphrase"))
    new_passphrase = _optional_str(payload.get("new_passphrase"))
    if not old_passphrase or not new_passphrase:
        raise ValueError("old and new passphrases are required")
    return rotate_cluster_passphrase(settings, old_passphrase=old_passphrase, new_passphrase=new_passphrase)


def _force_release_lease(settings: NodeSettings, payload: dict[str, Any]) -> dict[str, Any]:
    lease_id = str(payload.get("lease_id") or "").strip()
    if not lease_id:
        raise ValueError("lease_id is required")
    return {"released": force_release_lease(settings, lease_id), "lease_id": lease_id}


def _list_payload(payload: dict[str, Any], key: str) -> list[str]:
    return [str(value).strip() for value in list(payload.get(key) or []) if str(value).strip()]


_DASHBOARD_ACTIONS: dict[str, DashboardAction] = {
    "start-node": lambda settings, payload: start_node(settings, passphrase=_optional_str(payload.get("passphrase"))),
    "stop-node": lambda settings, _payload: stop_node(settings),
    "restart-node": lambda settings, payload: restart_node(settings, passphrase=_optional_str(payload.get("passphrase"))),
    "join-cluster": lambda settings, payload: join_cluster(settings, passphrase=_optional_str(payload.get("passphrase"))),
    "rebuild-state": lambda settings, payload: rebuild_cluster_state(settings, passphrase=_optional_str(payload.get("passphrase"))),
    "leave-cluster": lambda settings, _payload: leave_cluster(settings),
    "install-service": lambda settings, payload: install_service(settings, service_path=_optional_str(payload.get("service_path"))),
    "update-worker": lambda settings, _payload: update_worker(settings),
    "reset-worker": lambda settings, payload: reset_worker(settings, passphrase=_optional_str(payload.get("passphrase"))),
    "uninstall-worker": lambda settings, _payload: uninstall_worker(settings),
    "run-vendor-audit": lambda settings, _payload: run_vendor_audit(settings),
    "replan-coverage": lambda settings, _payload: replan_coverage(settings),
    "bootstrap-ledger": lambda settings, _payload: bootstrap_canonical_ledger(settings),
    "repair-canonical": lambda settings, payload: repair_canonical_backlog(
        settings,
        trading_date=_optional_str(payload.get("trading_date")),
        start_date=_optional_str(payload.get("start_date")),
        end_date=_optional_str(payload.get("end_date")),
        symbol=_optional_str(payload.get("symbol")),
        verify_only=bool(payload.get("verify_only")),
    ),
    "verify-recent": lambda settings, payload: verify_recent_canonical_dates(
        settings,
        days=int(payload.get("days") or 7),
        dataset=str(payload.get("dataset") or "equities_eod"),
        verify_only=bool(payload.get("verify_only")),
    ),
    "repair-status": lambda settings, _payload: repair_status(settings),
    "lane-health": lambda settings, payload: lane_health(settings, dataset=str(payload.get("dataset") or "equities_eod")),
    "train-preflight": lambda settings, payload: training_preflight_status(
        settings,
        phase=int(payload.get("phase") or 1),
        target=_optional_str(payload.get("target")),
    ),
    "train-start": lambda settings, payload: start_training_run(
        settings,
        phase=int(payload.get("phase") or 1),
        report_date=_optional_str(payload.get("report_date")),
        target=_optional_str(payload.get("target")),
    ),
    "train-stop": lambda settings, payload: stop_training_run(
        settings,
        phase=int(payload.get("phase") or 1),
        target=_optional_str(payload.get("target")),
    ),
    "train-status": lambda settings, payload: training_runtime_status(
        settings,
        phase=int(payload.get("phase") or 1),
        target=_optional_str(payload.get("target")),
    ),
    "train-logs": lambda settings, payload: training_runtime_logs(
        settings,
        phase=int(payload.get("phase") or 1),
        target=_optional_str(payload.get("target")),
        tail_lines=int(payload.get("tail_lines") or 50),
    ),
    "experiments-supervise": lambda settings, payload: start_experiment_supervisor(
        settings,
        spec_path=str(payload.get("spec_path") or (settings.repo_root / "configs" / "experiments" / "phase1_remote_baseline_sweep.yml")),
        poll_seconds=int(payload["poll_seconds"]) if payload.get("poll_seconds") is not None else None,
        detach=bool(payload.get("detach", True)),
    ),
    "experiments-pause": lambda settings, payload: pause_experiments(settings, experiment_id=str(payload.get("experiment_id") or "").strip()),
    "experiments-resume": lambda settings, payload: resume_experiments(settings, experiment_id=str(payload.get("experiment_id") or "").strip()),
    "experiments-stop": lambda settings, payload: stop_experiments(settings, experiment_id=str(payload.get("experiment_id") or "").strip()),
    "experiments-evaluate": lambda settings, payload: evaluate_experiments(settings, experiment_id=str(payload.get("experiment_id") or "").strip()),
    "experiments-backtest": lambda settings, payload: backtest_experiments(settings, experiment_id=str(payload.get("experiment_id") or "").strip()),
    "experiments-propose-next": lambda settings, payload: propose_experiment_family(settings, experiment_id=str(payload.get("experiment_id") or "").strip()),
    "research-start": lambda settings, payload: start_research_supervisor(
        settings,
        program_path=str(payload.get("program_path") or (settings.repo_root / "configs" / "research" / "perpetual_macmini.yml")),
        poll_seconds=int(payload["poll_seconds"]) if payload.get("poll_seconds") is not None else None,
        detach=bool(payload.get("detach", True)),
    ),
    "research-pause": lambda settings, payload: pause_research(settings, program_id=str(payload.get("program_id") or "").strip()),
    "research-resume": lambda settings, payload: resume_research(settings, program_id=str(payload.get("program_id") or "").strip()),
    "research-stop": lambda settings, payload: stop_research(settings, program_id=str(payload.get("program_id") or "").strip()),
    "research-status": lambda settings, payload: research_status(settings, program_id=str(payload.get("program_id") or "").strip()),
    "research-review-packet": lambda settings, payload: research_review_packet(settings, program_id=str(payload.get("program_id") or "").strip()),
    "research-steer": lambda settings, payload: steer_research(
        settings,
        program_id=str(payload.get("program_id") or "").strip(),
        prefer_architecture_families=_list_payload(payload, "prefer_architecture_families"),
        avoid_architecture_families=_list_payload(payload, "avoid_architecture_families"),
        prefer_data_families=_list_payload(payload, "prefer_data_families"),
        avoid_data_families=_list_payload(payload, "avoid_data_families"),
        freeze_phase=int(payload["freeze_phase"]) if payload.get("freeze_phase") is not None else None,
        force_pivot=bool(payload.get("force_pivot")) if payload.get("force_pivot") is not None else None,
        exploration_breadth=_optional_str(payload.get("exploration_breadth")),
    ),
    "save-settings": lambda settings, payload: persist_node_settings(
        settings,
        nas_share=str(payload.get("nas_share") or settings.nas_share),
        nas_mount=str(payload.get("nas_mount") or settings.nas_mount),
        collection_time_et=str(payload.get("collection_time_et") or settings.collection_time_et),
        maintenance_hour_local=int(payload.get("maintenance_hour_local") or settings.maintenance_hour_local),
        fstab_path=_optional_str(payload.get("fstab_path")),
    ),
    "update-secret": _cluster_secret_update,
    "rotate-passphrase": _cluster_passphrase_rotation,
    "force-release-lease": _force_release_lease,
    "advance-stage": lambda settings, payload: advance_collection_stage(
        settings,
        target_stage=int(payload.get("target_stage") or 0),
        symbol_count=int(payload.get("symbol_count")) if payload.get("symbol_count") is not None else None,
        years=int(payload.get("years")) if payload.get("years") is not None else None,
        passphrase=_optional_str(payload.get("passphrase")),
    ),
}


def create_dashboard_server(host: str, port: int, *, settings: NodeSettings) -> DashboardHTTPServer:
    """Create the threaded HTTP server bound to the requested interface."""
    return DashboardHTTPServer((host, port), settings=settings)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _open_browser_later(url: str) -> None:
    time.sleep(0.4)
    with threading.Lock():
        webbrowser.open(url)


def main(argv: list[str] | None = None) -> int:
    """Run the browser-served operator dashboard."""
    parser = argparse.ArgumentParser(description="TradeML dashboard server")
    parser.add_argument("--workspace-root", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--env-file", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    settings = resolve_node_settings(
        workspace_root=args.workspace_root,
        config_path=args.config,
        env_path=args.env_file,
    )
    httpd = create_dashboard_server(args.host, args.port, settings=settings)
    url = f"http://{args.host}:{httpd.server_port}"
    LOGGER.info("dashboard_listening url=%s workspace_root=%s", url, settings.workspace_root)
    if not args.no_browser:
        threading.Thread(target=_open_browser_later, args=(url,), daemon=True).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        LOGGER.info("dashboard_shutdown requested")
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
