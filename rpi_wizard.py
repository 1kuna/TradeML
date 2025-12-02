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
import platform
import shutil
import subprocess
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent
LOG_ROOT = REPO_ROOT / "logs"
STATE_FILE = LOG_ROOT / "rpi_wizard_state.json"
DEFAULT_ENV = REPO_ROOT / ".env"
ENV_TEMPLATE = REPO_ROOT / ".env.template"
ENV_PATH = Path(os.getenv("RPI_WIZARD_ENV_PATH", DEFAULT_ENV))


# ---------- Logging ----------

def _utc_formatter() -> logging.Formatter:
    fmt = "%(asctime)sZ | %(levelname)s | %(message)s"
    formatter = logging.Formatter(fmt)
    formatter.converter = lambda *args: datetime.now(timezone.utc).timetuple()
    return formatter


def _next_backup_path(path: Path) -> Path:
    """Return a unique backup path (path.bak, path.bak1, ...) without overwriting existing files."""
    idx = 0
    while True:
        suffix = ".bak" if idx == 0 else f".bak{idx}"
        candidate = path.with_name(f"{path.name}{suffix}")
        if not candidate.exists():
            return candidate
        idx += 1


def _ensure_log_dir(log_dir: Path) -> List[str]:
    """
    Make sure log_dir is a real directory (not a broken symlink or stray file).
    Returns cleanup messages to log after the logger is ready.
    """
    messages: List[str] = []
    if log_dir.is_dir():
        return messages

    if log_dir.is_symlink():
        target = None
        try:
            target = log_dir.readlink()
        except OSError:
            pass
        messages.append(f"Resetting log path {log_dir} (broken symlink -> {target or 'missing'})")
        log_dir.unlink(missing_ok=True)
    elif log_dir.exists():
        backup = _next_backup_path(log_dir)
        log_dir.rename(backup)
        messages.append(f"Moved existing log file {log_dir} to {backup} to create log directory")
    else:
        messages.append(f"Creating log directory at {log_dir}")

    log_dir.mkdir(parents=True, exist_ok=True)
    return messages


def setup_logging(log_dir: Path) -> Tuple[logging.Logger, Path]:
    prep_messages = _ensure_log_dir(log_dir)
    log_file = log_dir / f"rpi_wizard_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.log"
    logger = logging.getLogger("rpi_wizard")
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False
    # File handler (persist from start)
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(_utc_formatter())
    logger.addHandler(fh)
    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(_utc_formatter())
    logger.addHandler(ch)
    for msg in prep_messages:
        logger.info(msg)
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
    """
    Merge updates into the env file, de-duplicating keys and preserving the first-seen
    ordering of existing entries. Ensures a single line per key and removes stale
    duplicates from prior runs/tests.
    """
    ensure_env_file(path, logger)
    lines = path.read_text().splitlines()
    preamble: List[str] = []
    ordered_keys: List[str] = []
    values: Dict[str, str] = {}
    in_preamble = True

    for ln in lines:
        stripped = ln.strip()
        if in_preamble and (not stripped or stripped.startswith("#")):
            preamble.append(ln)
            continue
        in_preamble = False
        if not stripped or stripped.startswith("#") or "=" not in ln:
            continue
        key, val = ln.split("=", 1)
        key = key.strip()
        val = val.strip()
        if key not in ordered_keys:
            ordered_keys.append(key)
        values[key] = val  # last occurrence wins

    for key, val in updates.items():
        if key not in ordered_keys:
            ordered_keys.append(key)
        values[key] = val

    deduped_count = max(0, len(lines) - len(preamble) - len(ordered_keys))
    out: List[str] = []
    if preamble:
        out.extend(preamble)
    out.extend(f"{k}={values[k]}" for k in ordered_keys)
    path.write_text("\n".join(out).rstrip() + "\n")
    msg = f"Updated {path} with {', '.join(updates.keys())}"
    if deduped_count:
        msg += f" (removed {deduped_count} duplicate lines)"
    logger.info(msg)


def _sanitize_prev_state(prev: Dict[str, str], logger: logging.Logger) -> Dict[str, str]:
    """
    Drop obviously invalid prior state (e.g., macOS paths on Linux, temp pytest paths).
    Returns a sanitized copy suitable for reuse.
    """
    if not prev:
        return {}

    sanitized: Dict[str, str] = {}

    def _path_ok(path_str: str) -> Optional[Path]:
        if not path_str:
            return None
        p = Path(path_str).expanduser()
        if not p.is_absolute():
            logger.warning(f"Ignoring non-absolute path from state: {p}")
            return None
        if sys.platform.startswith("linux"):
            lowered = str(p).lower()
            if lowered.startswith("/users/") or lowered.startswith("/private/") or "pytest-of-" in lowered:
                logger.warning(f"Ignoring incompatible path from state on linux: {p}")
                return None
        return p

    data_root = _path_ok(prev.get("data_root", ""))
    venv_path = _path_ok(prev.get("venv_path", ""))
    edge_node_id = prev.get("edge_node_id", "")

    if data_root:
        sanitized["data_root"] = str(data_root)
    if venv_path:
        sanitized["venv_path"] = str(venv_path)
    if edge_node_id:
        sanitized["edge_node_id"] = edge_node_id
    return sanitized


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
            try:
                current = src.resolve(strict=False)
                current_exists = current.exists()
            except Exception:
                current = None
                current_exists = False

            if current_exists and current == dst:
                continue

            if current_exists and current != dst:
                logger.warning(f"{src} symlink points to {current}, expected {dst}. Leaving in place.")
                continue

            logger.warning(f"{src} symlink target is missing or invalid; realigning to {dst}")
            try:
                src.unlink(missing_ok=True)
            except Exception:
                logger.warning(f"Failed to remove broken symlink {src}; skipping realignment")
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


def _test_write(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    probe = path / ".rpi_wizard_write_test"
    probe.write_text("ok")
    probe.unlink(missing_ok=True)


def prompt_storage_path(default_path: Path, logger: logging.Logger) -> Path:
    while True:
        raw = input(f"Enter storage path [{default_path}]: ").strip()
        candidate = Path(raw or default_path).expanduser()
        if not candidate.is_absolute():
            print("Please provide an absolute path like /data or /mnt/ssd/trademl")
            continue
        try:
            _test_write(candidate)
            return candidate
        except FileNotFoundError:
            print(f"Path not found: {candidate}. If this is a mount, ensure it is mounted first.")
        except PermissionError:
            print(f"Permission denied writing to {candidate}. Choose a different path or adjust permissions.")
        except OSError as e:
            print(f"Path error ({e}); please try another path.")
        except Exception as e:
            logger.warning(f"Unexpected storage path error: {e}")
            print("Unable to use that path; please try another.")


def run_cmd(cmd: List[str], logger: logging.Logger, env: Optional[Dict[str, str]] = None, cwd: Optional[Path] = None, dry_run: bool = False) -> None:
    msg = f"Running: {' '.join(cmd)}"
    if dry_run:
        logger.info(f"[dry-run] {msg}")
        return
    logger.info(msg)
    subprocess.run(cmd, check=True, cwd=str(cwd) if cwd else None, env=env)


def ensure_venv(venv_path: Path, logger: logging.Logger, dry_run: bool = False) -> Path:
    def _venv_health() -> Tuple[bool, str]:
        python_bin = venv_path / "bin" / "python"
        if not python_bin.exists():
            return False, "python executable missing"
        if not os.access(python_bin, os.X_OK):
            return False, "python executable is not executable"
        try:
            output = subprocess.check_output(
                [str(python_bin), "-c", "import platform,sys; print(f\"{sys.version_info.major}.{sys.version_info.minor}\"); print(sys.platform); print(platform.machine())"],
                text=True,
                stderr=subprocess.STDOUT,
            )
        except Exception as exc:  # noqa: BLE001 - need raw exception for logging
            return False, f"python failed to start ({exc})"

        lines = [ln.strip() for ln in output.strip().splitlines() if ln.strip()]
        if len(lines) < 3:
            return False, "python health check returned incomplete data"

        venv_version, venv_platform, venv_machine = lines[:3]
        host_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        host_platform = sys.platform
        host_machine = platform.machine()

        if venv_version != host_version:
            return False, f"python version mismatch (venv={venv_version}, host={host_version})"
        if venv_platform != host_platform:
            return False, f"platform mismatch (venv={venv_platform}, host={host_platform})"
        if venv_machine and host_machine and venv_machine != host_machine:
            return False, f"architecture mismatch (venv={venv_machine}, host={host_machine})"
        return True, ""

    healthy, reason = _venv_health()
    if healthy:
        logger.info(f"Using existing venv at {venv_path}")
        return venv_path

    action_reason = reason or "venv missing or unreadable"
    if dry_run:
        verb = "recreate" if venv_path.exists() else "create"
        logger.info(f"[dry-run] Would {verb} venv at {venv_path} ({action_reason})")
        return venv_path

    if venv_path.exists():
        logger.warning(f"Rebuilding venv at {venv_path} ({action_reason})")
        shutil.rmtree(venv_path)
    else:
        logger.info(f"Creating venv at {venv_path}")
    venv_path.parent.mkdir(parents=True, exist_ok=True)
    run_cmd([sys.executable, "-m", "venv", str(venv_path)], logger)
    logger.info(f"Created venv at {venv_path}")
    return venv_path


def install_deps(venv_path: Path, logger: logging.Logger, mode: str, dry_run: bool = False) -> None:
    pip_bin = venv_path / "bin" / "pip"
    if not pip_bin.exists():
        if dry_run:
            logger.info(f"[dry-run] Would install dependencies ({mode}) into {venv_path}")
            return
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
            "pyyaml",
            "python-dotenv",
            "loguru",
            "alpaca-py",
            "requests",
            "exchange-calendars",
            "finnhub-python",
            "duckdb",
            "scikit-learn",
        ]
        run_cmd(base_cmd + ["install"] + pkgs, logger, dry_run=dry_run)
    logger.info(f"Dependencies installed ({mode})")


def node_selfcheck(venv_path: Path, logger: logging.Logger, env_overrides: Optional[Dict[str, str]] = None, dry_run: bool = False) -> None:
    python_bin = venv_path / "bin" / "python"
    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)
    run_cmd([str(python_bin), "scripts/node.py", "--selfcheck"], logger, cwd=REPO_ROOT, env=env, dry_run=dry_run)


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


def write_systemd_unit(
    venv_path: Path,
    data_root: Path,
    interval: int,
    env_path: Path,
    logger: logging.Logger,
    enable: bool = False,
    dry_run: bool = False,
) -> Optional[Path]:
    systemctl = shutil.which("systemctl")
    if sys.platform != "linux" or not systemctl:
        logger.info("Systemd not available on this platform; skipping unit creation")
        return None

    unit_dir = Path.home() / ".config" / "systemd" / "user"
    unit_dir.mkdir(parents=True, exist_ok=True)
    unit_path = unit_dir / "trademl-node.service"
    logs_dir = data_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    content = textwrap.dedent(
        f"""
        [Unit]
        Description=TradeML Node Loop
        After=network-online.target

        [Service]
        WorkingDirectory={REPO_ROOT}
        Environment=PYTHONPATH={REPO_ROOT}
        Environment=DATA_ROOT={data_root}
        EnvironmentFile={env_path}
        ExecStart={venv_path}/bin/python {REPO_ROOT}/scripts/node.py --interval {interval}
        Restart=always
        RestartSec=10
        StandardOutput=append:{logs_dir}/node.service.log
        StandardError=append:{logs_dir}/node.service.err

        [Install]
        WantedBy=default.target
        """
    ).strip() + "\n"

    if dry_run:
        logger.info(f"[dry-run] Would write systemd unit to {unit_path}")
        return unit_path

    unit_path.write_text(content)
    logger.info(f"Wrote systemd unit: {unit_path}")

    try:
        subprocess.run([systemctl, "--user", "daemon-reload"], check=True)
        if enable:
            subprocess.run([systemctl, "--user", "enable", "--now", unit_path.name], check=True)
            logger.info("Enabled systemd unit for auto-start")
        else:
            logger.info(f"Enable with: systemctl --user enable --now {unit_path.name}")
    except Exception as e:
        logger.warning(f"Failed to enable systemd unit automatically: {e}")

    return unit_path


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
        f"- Live node log: tail -f {summary.get('node_log','logs/node.log')}",
    ]
    if summary.get("node_pid"):
        lines.append(f"- Node loop pid: {summary['node_pid']}")
    if summary.get("mlflow_ui"):
        lines.append(f"- MLflow UI: {summary['mlflow_ui']}")
    if summary.get("stop_cmd"):
        lines.append(f"- Stop node: {summary['stop_cmd']}")
    logger.info("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Interactive Raspberry Pi data-collection wizard")
    parser.add_argument("--install-mode", choices=["ingest", "full"], default="ingest", help="Dependency profile to install")
    parser.add_argument("--interval", type=int, default=int(os.getenv("RUN_INTERVAL_SECONDS", "900")), help="Sleep seconds between node cycles")
    parser.add_argument("--dry-run", action="store_true", help="Show actions without executing them")
    parser.add_argument("--fresh", action="store_true", help="Ignore saved wizard state and prompt for everything")
    parser.add_argument("--write-systemd", action="store_true", help="Write systemd user unit for node loop")
    parser.add_argument("--enable-systemd", action="store_true", help="Enable and start systemd unit after writing")
    args = parser.parse_args()

    if sys.version_info < (3, 10):
        sys.stderr.write(f"Python 3.10+ required, found {sys.version.split()[0]}. Install newer Python and re-run.\n")
        sys.exit(2)

    logger, log_file = setup_logging(LOG_ROOT)
    state = WizardState(STATE_FILE)
    prev_raw = state.load()
    prev = _sanitize_prev_state(prev_raw, logger)

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
            default_root = Path(prev.get("data_root") or (REPO_ROOT / "data"))
            if not default_root.is_absolute():
                default_root = REPO_ROOT / "data"
            print("\nSelect storage path (external SSD recommended).")
            if mounts:
                print("Detected mounts:")
                for m in mounts:
                    print(f" - {m}")
            storage_choice = prompt_choice("Use local /data or custom path?", ["local", "custom"], default="local")
            if storage_choice == "local":
                data_root = Path("/data")
                try:
                    _test_write(data_root)
                except Exception as e:
                    print(f"Cannot use /data ({e}); please choose a custom path.")
                    data_root = prompt_storage_path(REPO_ROOT / "data", logger)
            else:
                data_root = prompt_storage_path(default_root, logger)
            edge_node_id = input(f"Edge node id [{os.uname().nodename}]: ").strip() or os.uname().nodename
            venv_default = Path(prev.get("venv_path") or (REPO_ROOT / "venv"))
            if not venv_default.is_absolute():
                venv_default = REPO_ROOT / "venv"
            venv_input = input(f"Venv path [{venv_default}]: ").strip()
            venv_path = Path(venv_input).expanduser() if venv_input else venv_default
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
    env_path = ENV_PATH
    upsert_env(env_path, env_updates, logger)

    ensure_venv(venv_path, logger, dry_run=args.dry_run)
    install_deps(venv_path, logger, mode=args.install_mode, dry_run=args.dry_run)
    env_overrides = {
        "EDGE_NODE_ID": edge_node_id,
        "DATA_ROOT": str(data_root),
        "STORAGE_BACKEND": "local",
    }
    node_selfcheck(venv_path, logger, env_overrides=env_overrides, dry_run=args.dry_run)

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
            env_overrides=env_overrides,
            dry_run=args.dry_run,
        )

    if node_pid:
        state.data["last_node_pid"] = str(node_pid)
        state.save(extra_copy=data_root / "trademl_state" / "rpi_wizard_state.json")

    env_after = _load_env(env_path)
    mlflow_ui = env_after.get("MLFLOW_TRACKING_URI") if env_after.get("MLFLOW_TRACKING_URI", "").startswith("http") else ""
    # Final summary
    summary = {
        "data_root": str(data_root),
        "env_path": str(env_path),
        "venv_path": str(venv_path),
        "wizard_log": str(log_file),
        "node_log": str(data_root / "logs" / "node.log"),
        "node_pid": str(node_pid) if node_pid else "",
        "mlflow_ui": mlflow_ui,
        "stop_cmd": "./bootstrap.sh stop",
    }
    summarize(logger, summary)

    if not args.dry_run:
        wants_systemd = False
        if args.write_systemd:
            wants_systemd = True
        elif not auto_mode:
            print("\nOptional: auto-run the node on reboot using systemd (user service).")
            print("This will keep ingestion running in the background even if you close the terminal.")
            wants_systemd = prompt_yes_no("Set up systemd now?", default=False)
        if wants_systemd:
            write_systemd_unit(
                venv_path=venv_path,
                data_root=data_root,
                interval=args.interval,
                env_path=env_path,
                logger=logger,
                enable=args.enable_systemd,
                dry_run=args.dry_run,
            )

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
