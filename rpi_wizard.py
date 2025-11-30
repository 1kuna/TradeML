#!/usr/bin/env python3
"""
Raspberry Pi data-collection wizard.

Single-command, fully interactive setup for edge ingestion on a Pi:
- Picks a storage root (external SSD encouraged) and symlinks data/log paths to it
- Creates/activates a venv and installs ingest dependencies
- Patches .env with sane defaults (edge role, local storage, node id, budgets)
- Runs self-checks, launches the node loop, and captures logs from start
- Persists state so reruns months later can resume where ingestion stopped

Usage:
  python rpi_wizard.py            # interactive flow
  python rpi_wizard.py --resume   # prefer prior state if found
  python rpi_wizard.py --dry-run  # show planned actions, do not execute
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent
LOG_ROOT = REPO_ROOT / "logs"
STATE_FILE = LOG_ROOT / "rpi_wizard_state.json"
DEFAULT_ENV = REPO_ROOT / ".env"
ENV_TEMPLATE = REPO_ROOT / ".env.template"


# ---------- Logging ----------

def _utc_formatter() -> logging.Formatter:
    fmt = "%(asctime)sZ | %(levelname)s | %(message)s"
    formatter = logging.Formatter(fmt)
    formatter.converter = lambda *args: datetime.now(timezone.utc).timetuple()
    return formatter


def setup_logging(log_dir: Path) -> Tuple[logging.Logger, Path]:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"rpi_wizard_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.log"
    logger = logging.getLogger("rpi_wizard")
    logger.setLevel(logging.INFO)
    # File handler (persist from start)
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(_utc_formatter())
    logger.addHandler(fh)
    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(_utc_formatter())
    logger.addHandler(ch)
    logger.info(f"Logging to {log_file}")
    return logger, log_file


# ---------- State management ----------

class WizardState:
    def __init__(self, path: Path):
        self.path = path
        self.data: Dict[str, str] = {}

    def load(self) -> Dict[str, str]:
        if self.path.exists():
            try:
                self.data = json.loads(self.path.read_text())
            except Exception:
                self.data = {}
        return self.data

    def save(self, extra_copy: Optional[Path] = None) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(self.data)
        payload["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        self.path.write_text(json.dumps(payload, indent=2))
        if extra_copy:
            try:
                extra_copy.parent.mkdir(parents=True, exist_ok=True)
                extra_copy.write_text(json.dumps(payload, indent=2))
            except Exception:
                pass


# ---------- Helpers ----------

def prompt_choice(prompt: str, options: List[str], default: Optional[str] = None) -> str:
    opts = "/".join([f"[{o}]" if o == default else o for o in options])
    while True:
        ans = input(f"{prompt} ({opts}): ").strip().lower()
        if not ans and default:
            return default
        for o in options:
            if ans == o.lower():
                return o
        print("Please choose a valid option.")


def prompt_yes_no(prompt: str, default: bool = True) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    while True:
        ans = input(f"{prompt} {suffix} ").strip().lower()
        if not ans:
            return default
        if ans in {"y", "yes"}:
            return True
        if ans in {"n", "no"}:
            return False
        print("Please enter y or n.")


def detect_mounts() -> List[Path]:
    mounts: List[Path] = []
    for base in (Path("/media"), Path("/mnt"), Path("/Volumes")):
        if not base.exists():
            continue
        for child in base.iterdir():
            if child.is_dir():
                mounts.append(child)
    return mounts


def ensure_env_file(env_path: Path, logger: logging.Logger) -> None:
    if not env_path.exists():
        if ENV_TEMPLATE.exists():
            shutil.copyfile(ENV_TEMPLATE, env_path)
            logger.info(f"Created {env_path} from template")
        else:
            env_path.touch()
            logger.info(f"Created empty {env_path} (template missing)")


def _load_env(path: Path) -> Dict[str, str]:
    env: Dict[str, str] = {}
    if not path.exists():
        return env
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        env[key.strip()] = val.strip()
    return env


def upsert_env(path: Path, updates: Dict[str, str], logger: logging.Logger) -> None:
    ensure_env_file(path, logger)
    lines = path.read_text().splitlines()
    existing = {k: i for i, k in enumerate([ln.split("=", 1)[0] for ln in lines if "=" in ln and not ln.strip().startswith("#")])}
    for key, val in updates.items():
        line = f"{key}={val}"
        if key in existing:
            idx = existing[key]
            lines[idx] = line
        else:
            lines.append(line)
    path.write_text("\n".join(lines) + "\n")
    logger.info(f"Updated {path} with {', '.join(updates.keys())}")


def _safe_merge_dir(src: Path, dst: Path, logger: logging.Logger):
    dst.mkdir(parents=True, exist_ok=True)
    for child in src.iterdir():
        target = dst / child.name
        if target.exists():
            logger.warning(f"Conflict while moving {child} -> {target}; leaving source in place")
            continue
        shutil.move(str(child), str(target))
    if not any(src.iterdir()):
        src.rmdir()


def ensure_data_paths(data_root: Path, logger: logging.Logger):
    """
    Ensure all data/log paths live on the external SSD and are symlinked from the repo.
    Non-destructive: merges existing contents when possible and backs up conflicts.
    """
    if os.getenv("RPI_WIZARD_SKIP_SYMLINKS", "").lower() in {"1", "true", "yes"}:
        logger.info("Skipping data path symlinks (RPI_WIZARD_SKIP_SYMLINKS set)")
        return
    mappings = [
        "data_layer/raw",
        "data_layer/curated",
        "data_layer/reference/corp_actions",
        "data_layer/reference/delistings",
        "data_layer/reference/index_membership",
        "data_layer/reference/tick_size_regime",
        "data_layer/reference/universe",
        "data_layer/reference/universe_symbols.txt",
        "data_layer/qc",
        "data_layer/manifests",
        "logs",
    ]

    for rel in mappings:
        src = REPO_ROOT / rel
        dst = data_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)

        if src.is_symlink():
            # Already redirected; ensure it points to the chosen data_root
            try:
                current = src.resolve()
                if current != dst:
                    logger.warning(f"{src} symlink points to {current}, expected {dst}. You may want to realign manually.")
            except Exception:
                logger.warning(f"Failed to resolve symlink {src}; leaving as-is")
            continue

        if src.exists():
            if src.is_dir():
                _safe_merge_dir(src, dst, logger)
            else:
                if not dst.exists():
                    shutil.move(str(src), str(dst))
                else:
                    backup = dst.with_suffix(".backup_from_repo")
                    shutil.move(str(src), str(backup))
                    logger.warning(f"File conflict for {src}; backed up to {backup}")
                try:
                    src.unlink()
                except Exception:
                    pass
        else:
            # Ensure destination exists
            if rel.endswith(".txt"):
                dst.parent.mkdir(parents=True, exist_ok=True)
            else:
                dst.mkdir(parents=True, exist_ok=True)

        if src.exists():
            try:
                if src.is_dir():
                    shutil.rmtree(src)
                else:
                    src.unlink()
            except Exception:
                pass
        try:
            src.parent.mkdir(parents=True, exist_ok=True)
            src.symlink_to(dst)
            logger.info(f"Symlinked {src} -> {dst}")
        except FileExistsError:
            # Already linked/created in a race; ignore
            pass


def run_cmd(cmd: List[str], logger: logging.Logger, env: Optional[Dict[str, str]] = None, cwd: Optional[Path] = None, dry_run: bool = False) -> None:
    msg = f"Running: {' '.join(cmd)}"
    if dry_run:
        logger.info(f"[dry-run] {msg}")
        return
    logger.info(msg)
    subprocess.run(cmd, check=True, cwd=str(cwd) if cwd else None, env=env)


def ensure_venv(venv_path: Path, logger: logging.Logger, dry_run: bool = False) -> Path:
    if venv_path.exists() and (venv_path / "bin/python").exists():
        logger.info(f"Using existing venv at {venv_path}")
        return venv_path
    if dry_run:
        logger.info(f"[dry-run] Would create venv at {venv_path}")
        return venv_path
    venv_path.parent.mkdir(parents=True, exist_ok=True)
    run_cmd([sys.executable, "-m", "venv", str(venv_path)], logger)
    logger.info(f"Created venv at {venv_path}")
    return venv_path


def install_deps(venv_path: Path, logger: logging.Logger, mode: str, dry_run: bool = False) -> None:
    pip_bin = venv_path / "bin" / "pip"
    if not pip_bin.exists():
        raise RuntimeError(f"pip not found in venv ({pip_bin})")
    base_cmd = [str(pip_bin)]
    run_cmd(base_cmd + ["install", "--upgrade", "pip"], logger, dry_run=dry_run)
    if mode == "full":
        req = REPO_ROOT / "requirements.txt"
        run_cmd(base_cmd + ["install", "-r", str(req)], logger, dry_run=dry_run)
    else:
        pkgs = [
            "boto3",
            "pandas",
            "pyarrow",
            "python-dotenv",
            "loguru",
            "alpaca-py",
            "requests",
            "exchange-calendars",
            "finnhub-python",
            "duckdb",
        ]
        run_cmd(base_cmd + ["install"] + pkgs, logger, dry_run=dry_run)
    logger.info(f"Dependencies installed ({mode})")


def node_selfcheck(venv_path: Path, logger: logging.Logger, dry_run: bool = False) -> None:
    python_bin = venv_path / "bin" / "python"
    run_cmd([str(python_bin), "scripts/node.py", "--selfcheck"], logger, cwd=REPO_ROOT, dry_run=dry_run)


def start_node_loop(venv_path: Path, log_dir: Path, logger: logging.Logger, interval: int, env_overrides: Dict[str, str], dry_run: bool = False) -> Optional[int]:
    python_bin = venv_path / "bin" / "python"
    node_log = log_dir / "node.log"
    log_dir.mkdir(parents=True, exist_ok=True)
    cmd = [str(python_bin), "scripts/node.py", "--interval", str(interval)]
    if dry_run:
        logger.info(f"[dry-run] Would launch node loop: {' '.join(cmd)} -> {node_log}")
        return None
    env = os.environ.copy()
    env.update(env_overrides)
    # Ensure PYTHONPATH includes repo root for imports
    env["PYTHONPATH"] = str(REPO_ROOT)
    with node_log.open("ab") as fh:
        proc = subprocess.Popen(cmd, stdout=fh, stderr=fh, cwd=REPO_ROOT, env=env)
    logger.info(f"Node loop started (pid={proc.pid}); logs -> {node_log}")
    return proc.pid


def pid_is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def summarize(logger: logging.Logger, summary: Dict[str, str]) -> None:
    lines = [
        "",
        "Wizard setup complete. Quick links / commands:",
        f"- Data root: {summary.get('data_root')}",
        f"- Env file: {summary.get('env_path')}",
        f"- Venv: {summary.get('venv_path')}",
        f"- Logs: tail -f {summary.get('wizard_log')} (wizard) | tail -f {summary.get('node_log','logs/node.log')} (node)",
    ]
    if summary.get("node_pid"):
        lines.append(f"- Node loop pid: {summary['node_pid']}")
    if summary.get("mlflow_ui"):
        lines.append(f"- MLflow UI: {summary['mlflow_ui']}")
    logger.info("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Interactive Raspberry Pi data-collection wizard")
    parser.add_argument("--install-mode", choices=["ingest", "full"], default="ingest", help="Dependency profile to install")
    parser.add_argument("--interval", type=int, default=int(os.getenv("RUN_INTERVAL_SECONDS", "900")), help="Sleep seconds between node cycles")
    parser.add_argument("--dry-run", action="store_true", help="Show actions without executing them")
    parser.add_argument("--fresh", action="store_true", help="Ignore saved wizard state and prompt for everything")
    args = parser.parse_args()

    logger, log_file = setup_logging(LOG_ROOT)
    state = WizardState(STATE_FILE)
    prev = state.load()

    logger.info("Starting Raspberry Pi data-collection wizard (SSOT-aligned, local-first)")
    auto_mode = os.getenv("RPI_WIZARD_AUTO", "").lower() in {"1", "true", "yes"}
    use_prev = bool(prev and not args.fresh)
    if use_prev:
        logger.info("Found prior state; reusing saved settings")
        data_root = Path(prev.get("data_root", REPO_ROOT / "data")).expanduser()
        edge_node_id = prev.get("edge_node_id", os.uname().nodename)
        venv_path = Path(prev.get("venv_path", REPO_ROOT / "venv"))
    else:
        if auto_mode:
            data_root = Path(os.getenv("DATA_ROOT", REPO_ROOT / "data")).expanduser()
            edge_node_id = os.getenv("EDGE_NODE_ID", os.uname().nodename)
            venv_path = Path(os.getenv("RPI_WIZARD_VENV", REPO_ROOT / "venv")).expanduser()
            logger.info(f"Auto mode: DATA_ROOT={data_root}, EDGE_NODE_ID={edge_node_id}, VENV={venv_path}")
        else:
            mounts = detect_mounts()
            default_root = Path(prev.get("data_root") or (mounts[0] if mounts else (REPO_ROOT / "data")))
            print("\nSelect storage path (external SSD recommended).")
            if mounts:
                print("Detected mounts:")
                for m in mounts:
                    print(f" - {m}")
            data_root_input = input(f"Data root [{default_root}]: ").strip()
            data_root = Path(data_root_input or default_root).expanduser()
            edge_node_id = input(f"Edge node id [{os.uname().nodename}]: ").strip() or os.uname().nodename
            venv_path = Path(prev.get("venv_path") or (REPO_ROOT / "venv"))
            venv_input = input(f"Venv path [{venv_path}]: ").strip()
            if venv_input:
                venv_path = Path(venv_input).expanduser()
    data_root.mkdir(parents=True, exist_ok=True)

    # Move/symlink data paths to SSD
    ensure_data_paths(data_root, logger)

    # Persist initial state before heavy work
    state.data = {
        "data_root": str(data_root),
        "storage_mode": "local",
        "edge_node_id": edge_node_id,
        "venv_path": str(venv_path),
    }
    state.save(extra_copy=data_root / "trademl_state" / "rpi_wizard_state.json")

    env_updates = {
        "ROLE": "edge",
        "STORAGE_BACKEND": "local",
        "DATA_ROOT": str(data_root),
        "EDGE_NODE_ID": edge_node_id,
        "EDGE_SCHEDULER_MODE": "per_vendor",
        "NODE_MAX_INFLIGHT_ALPACA": os.getenv("NODE_MAX_INFLIGHT_ALPACA", "3"),
        "NODE_MAX_INFLIGHT_POLYGON": os.getenv("NODE_MAX_INFLIGHT_POLYGON", "2"),
        "NODE_MAX_INFLIGHT_FINNHUB": os.getenv("NODE_MAX_INFLIGHT_FINNHUB", "2"),
        "NODE_MAX_INFLIGHT_FRED": os.getenv("NODE_MAX_INFLIGHT_FRED", "2"),
        "REQUEST_PACING_ENABLED": "true",
        "PARQUET_COMPRESSION": "zstd",
    }
    upsert_env(DEFAULT_ENV, env_updates, logger)

    ensure_venv(venv_path, logger, dry_run=args.dry_run)
    install_deps(venv_path, logger, mode=args.install_mode, dry_run=args.dry_run)
    node_selfcheck(venv_path, logger, dry_run=args.dry_run)

    node_pid = None
    existing_pid_val = prev.get("last_node_pid") if prev else None
    if existing_pid_val:
        try:
            existing_pid_int = int(existing_pid_val)
            if pid_is_running(existing_pid_int) and not args.dry_run:
                if not prompt_yes_no(f"Existing node loop appears running (pid {existing_pid_int}). Start another anyway?", default=False):
                    node_pid = existing_pid_int
        except Exception:
            pass
    if node_pid is None:
        node_pid = start_node_loop(
            venv_path,
            log_dir=data_root / "logs",
            logger=logger,
            interval=args.interval,
            env_overrides={"EDGE_NODE_ID": edge_node_id, "DATA_ROOT": str(data_root)},
            dry_run=args.dry_run,
        )

    if node_pid:
        state.data["last_node_pid"] = str(node_pid)
        state.save(extra_copy=data_root / "trademl_state" / "rpi_wizard_state.json")

    env_after = _load_env(DEFAULT_ENV)
    mlflow_ui = env_after.get("MLFLOW_TRACKING_URI") if env_after.get("MLFLOW_TRACKING_URI", "").startswith("http") else ""
    # Final summary
    summary = {
        "data_root": str(data_root),
        "env_path": str(DEFAULT_ENV),
        "venv_path": str(venv_path),
        "wizard_log": str(log_file),
        "node_log": str(data_root / "logs" / "node.log"),
        "node_pid": str(node_pid) if node_pid else "",
        "mlflow_ui": mlflow_ui,
    }
    summarize(logger, summary)

    if args.dry_run:
        logger.info("Dry-run complete. No changes were executed.")
        return
    logger.info("Wizard finished. Ingestion will continue and resume on restart using saved state.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted; exiting.")
        sys.exit(1)
    except Exception as exc:
        print(f"[ERROR] Wizard failed: {exc}", file=sys.stderr)
        sys.exit(2)
