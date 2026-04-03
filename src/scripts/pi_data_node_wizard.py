"""Pi data-node setup wizard."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import yaml

from trademl.connectors.alpaca import AlpacaConnector
from trademl.data_node.bootstrap import DEFAULT_STAGE0_SYMBOL_COUNT, Stage0UniverseBuilder
from trademl.data_node.budgets import BudgetManager
from trademl.data_node.db import DataNodeDB
from trademl.data_node.vendor_limits import DEFAULT_VENDOR_LIMITS


def run_wizard(
    *,
    root: Path,
    config_path: Path,
    stage_symbols: list[str],
    stage_years: int = 5,
    nas_mount: str = "/mnt/trademl",
    nas_share: str = "//nas/trademl",
    collection_time_et: str = "16:30",
    maintenance_hour_local: int = 2,
    fstab_path: Path | None = None,
) -> dict:
    """Initialize Pi-local state and seed Stage 0 bootstrap tasks."""
    local_state = root / "control"
    local_state.mkdir(parents=True, exist_ok=True)
    db = DataNodeDB(local_state / "node.sqlite")
    bookmarks = root / "bookmarks.json"
    stage_file = root / "stage.yml"
    bookmarks.write_text(json.dumps({"stage": 0, "symbols_seeded": len(stage_symbols)}, indent=2), encoding="utf-8")
    stage_file.write_text(
        yaml.safe_dump(
            {
                "current": 0,
                "symbols": stage_symbols,
                "years": stage_years,
                "environment": _detect_environment(),
                "schedule": {
                    "collection_time_et": collection_time_et,
                    "maintenance_hour_local": maintenance_hour_local,
                },
                "nas": {"share": nas_share, "mount": nas_mount},
            }
        ),
        encoding="utf-8",
    )
    persisted_fstab = _persist_fstab_entry(
        path=fstab_path or Path("/etc/fstab"),
        nas_share=nas_share,
        nas_mount=nas_mount,
    )
    _write_node_config(
        config_path=config_path,
        nas_mount=nas_mount,
        nas_share=nas_share,
        local_state=local_state,
        collection_time_et=collection_time_et,
        maintenance_hour_local=maintenance_hour_local,
    )

    end_date = datetime.now(UTC).date().isoformat()
    start_date = f"{int(end_date[:4]) - stage_years}-{end_date[5:]}"
    for symbol in stage_symbols:
        try:
            db.enqueue_task("equities_eod", symbol, start_date, end_date, "BOOTSTRAP", 5)
        except Exception:
            continue

    return {
        "local_state": str(local_state),
        "task_count": len(stage_symbols),
        "config_path": str(config_path),
        "fstab_path": str(persisted_fstab),
    }


def main() -> int:
    """CLI entry point for initializing Pi node state."""
    parser = argparse.ArgumentParser(description="Initialize TradeML Pi node state.")
    parser.add_argument("--root", default="~/trademl")
    parser.add_argument("--config", default="configs/node.yml")
    parser.add_argument("--stage-years", type=int, default=5)
    parser.add_argument("--nas-mount", default=None)
    parser.add_argument("--nas-share", default="//nas/trademl")
    parser.add_argument("--collection-time-et", default="16:30")
    parser.add_argument("--maintenance-hour-local", type=int, default=2)
    parser.add_argument("--env-file", default=None)
    parser.add_argument("--fstab-path", default="/etc/fstab")
    parser.add_argument("--alpaca-api-key", default="")
    parser.add_argument("--alpaca-api-secret", default="")
    parser.add_argument("--finnhub-api-key", default="")
    parser.add_argument("--alpha-vantage-api-key", default="")
    parser.add_argument("--fred-api-key", default="")
    parser.add_argument("--fmp-api-key", default="")
    parser.add_argument("--massive-api-key", default="")
    parser.add_argument("--cluster-passphrase", default="")
    parser.add_argument("--stage-symbol", action="append", default=[])
    parser.add_argument("--start-node", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).expanduser()
    env_values = {
        "TRADEML_ENV": "local",
        "NAS_MOUNT": args.nas_mount or _prompt("NAS mount", "/mnt/trademl"),
        "NAS_SHARE": args.nas_share,
        "LOCAL_STATE": str((root / "control").expanduser()),
        "EDGE_NODE_ID": os.getenv("EDGE_NODE_ID", "rpi-01"),
        "COLLECTION_TIME_ET": args.collection_time_et,
        "MAINTENANCE_HOUR_LOCAL": str(args.maintenance_hour_local),
        "ALPACA_API_KEY": args.alpaca_api_key or _prompt("Alpaca API key", os.getenv("ALPACA_API_KEY", "")),
        "ALPACA_API_SECRET": args.alpaca_api_secret or _prompt("Alpaca API secret", os.getenv("ALPACA_API_SECRET", "")),
        "ALPACA_BASE_URL": os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets/v2"),
        "ALPACA_DATA_BASE_URL": os.getenv("ALPACA_DATA_BASE_URL", "https://data.alpaca.markets"),
        "FINNHUB_API_KEY": args.finnhub_api_key or _prompt("Finnhub API key", os.getenv("FINNHUB_API_KEY", "")),
        "ALPHA_VANTAGE_API_KEY": args.alpha_vantage_api_key or _prompt("Alpha Vantage API key", os.getenv("ALPHA_VANTAGE_API_KEY", "")),
        "FRED_API_KEY": args.fred_api_key or _prompt("FRED API key", os.getenv("FRED_API_KEY", "")),
        "FMP_API_KEY": args.fmp_api_key or _prompt("FMP API key", os.getenv("FMP_API_KEY", "")),
        "MASSIVE_API_KEY": args.massive_api_key or _prompt("Massive API key", os.getenv("MASSIVE_API_KEY", "")),
        "SEC_EDGAR_USER_AGENT": os.getenv("SEC_EDGAR_USER_AGENT", "TradeML/0.1 contact@example.com"),
    }
    nas_write_ok = _test_nas_mount(Path(env_values["NAS_MOUNT"]))
    env_path = Path(args.env_file).expanduser() if args.env_file else root / ".env"
    _write_env_file(env_path, env_values)
    stage_symbols = [symbol.strip().upper() for symbol in args.stage_symbol if symbol.strip()]
    if not stage_symbols:
        builder = Stage0UniverseBuilder(
            connector=AlpacaConnector(
                base_url=os.getenv("ALPACA_DATA_BASE_URL", env_values["ALPACA_DATA_BASE_URL"]),
                api_key=env_values["ALPACA_API_KEY"],
                secret_key=env_values["ALPACA_API_SECRET"],
                budget_manager=BudgetManager({"alpaca": DEFAULT_VENDOR_LIMITS["alpaca"]}),
            )
        )
        stage_symbols = builder.build(symbol_count=DEFAULT_STAGE0_SYMBOL_COUNT)
    result = run_wizard(
        root=root,
        config_path=Path(args.config),
        stage_symbols=stage_symbols,
        stage_years=args.stage_years,
        nas_mount=env_values["NAS_MOUNT"],
        nas_share=args.nas_share,
        collection_time_et=args.collection_time_et,
        maintenance_hour_local=args.maintenance_hour_local,
        fstab_path=Path(args.fstab_path).expanduser(),
    )
    result.update({"nas_write_ok": nas_write_ok, "env_file": str(env_path)})
    if args.start_node:
        child_env = os.environ.copy()
        if args.cluster_passphrase:
            child_env["TRADEML_CLUSTER_PASSPHRASE"] = args.cluster_passphrase
        subprocess.Popen(
            [
                os.sys.executable,
                "-m",
                "trademl.data_node",
                "--config",
                str(Path(args.config).expanduser()),
                "--root",
                str(root),
                "--env-file",
                str(env_path),
            ],  # noqa: S603
            env=child_env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        result["node_started"] = True
    print(json.dumps(result))
    return 0

def _prompt(label: str, default: str) -> str:
    if not os.isatty(0):
        return default
    value = input(f"{label} [{default}]: ").strip()
    return value or default


def _detect_environment() -> dict[str, object]:
    memory_mb = 0
    if hasattr(os, "sysconf") and "SC_PAGE_SIZE" in os.sysconf_names and "SC_PHYS_PAGES" in os.sysconf_names:
        memory_mb = int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024 * 1024))
    return {"machine": platform.machine(), "platform": platform.platform(), "memory_mb": memory_mb}


def _test_nas_mount(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path, delete=True):
            pass
        return True
    except OSError:
        return False


def _write_env_file(path: Path, values: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(f"{key}={value}" for key, value in values.items()) + "\n", encoding="utf-8")


def _persist_fstab_entry(*, path: Path, nas_share: str, nas_mount: str) -> Path:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        target_path = path
    except PermissionError:
        target_path = Path.cwd() / "fstab.tradeML"
        target_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        target_path = Path.cwd() / "fstab.tradeML"
        target_path.parent.mkdir(parents=True, exist_ok=True)
    existing = target_path.read_text(encoding="utf-8") if target_path.exists() else ""
    entry = f"{nas_share} {nas_mount} cifs credentials=/etc/nas-creds,uid=pi,gid=pi 0 0"
    lines = [line for line in existing.splitlines() if line.strip() and nas_mount not in line and nas_share not in line]
    lines.append(entry)
    try:
        target_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except PermissionError:
        target_path = Path.cwd() / "fstab.tradeML"
        target_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target_path


def _write_node_config(
    *,
    config_path: Path,
    nas_mount: str,
    nas_share: str,
    local_state: Path,
    collection_time_et: str,
    maintenance_hour_local: int,
) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config = {}
    if config_path.exists():
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    node = config.setdefault("node", {})
    node["nas_mount"] = nas_mount
    node["nas_share"] = nas_share
    node["local_state"] = str(local_state)
    node["collection_time_et"] = collection_time_et
    node["maintenance_hour_local"] = maintenance_hour_local
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
