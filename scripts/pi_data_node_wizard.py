#!/usr/bin/env python3
"""
Pi Data-Node Setup Wizard.

Interactive setup for the unified data_node service on Raspberry Pi:
- Detects environment (Pi vs Mac, external SSD)
- Selects storage root for data_layer
- Collects API keys for vendors
- Writes .env with proper configuration
- Initializes control DB and Stage 0
- Seeds BOOTSTRAP tasks for initial universe
- Optionally starts the data_node service

Usage:
  python scripts/pi_data_node_wizard.py             # interactive flow
  python scripts/pi_data_node_wizard.py --resume    # resume from saved state
  python scripts/pi_data_node_wizard.py --dry-run   # show actions without executing

See updated_node_spec.md §7 for wizard semantics.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Ensure repo root on path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# State file location
STATE_FILE = REPO_ROOT / "logs" / "pi_data_node_wizard_state.json"
DEFAULT_ENV_PATH = REPO_ROOT / ".env"

# Vendor API key mappings
VENDOR_KEYS = {
    "alpaca": {
        "env_vars": ["ALPACA_API_KEY", "ALPACA_SECRET_KEY"],
        "required": True,
        "description": "Primary data source for equities",
    },
    "finnhub": {
        "env_vars": ["FINNHUB_API_KEY"],
        "required": False,
        "description": "Options chains and news",
    },
    "massive": {
        "env_vars": ["MASSIVE_API_KEY"],
        "required": False,
        "description": "Cross-vendor QC verification (Polygon.io rebrand)",
    },
    "fred": {
        "env_vars": ["FRED_API_KEY"],
        "required": False,
        "description": "Macro economic data",
    },
    "alpha_vantage": {
        "env_vars": ["AV_API_KEY"],
        "required": False,
        "description": "Fundamentals and forex",
    },
    "fmp": {
        "env_vars": ["FMP_API_KEY"],
        "required": False,
        "description": "Financial modeling prep data",
    },
}

# Default inflight limits for Pi (conservative)
DEFAULT_INFLIGHT = {
    "alpaca": 2,
    "finnhub": 1,
    "massive": 1,
    "fred": 1,
    "av": 1,
    "fmp": 1,
}


def print_header(text: str) -> None:
    """Print a section header."""
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}\n")


def print_info(text: str) -> None:
    """Print an info message."""
    print(f"  ℹ️  {text}")


def print_success(text: str) -> None:
    """Print a success message."""
    print(f"  ✓  {text}")


def print_warning(text: str) -> None:
    """Print a warning message."""
    print(f"  ⚠️  {text}")


def print_error(text: str) -> None:
    """Print an error message."""
    print(f"  ✗  {text}")


def check_dependencies() -> bool:
    """Check if required dependencies are installed.

    Returns True if all dependencies are available, False otherwise.
    Prints helpful instructions if dependencies are missing.
    """
    missing = []

    try:
        import loguru  # noqa: F401
    except ImportError:
        missing.append("loguru")

    try:
        import yaml  # noqa: F401
    except ImportError:
        missing.append("pyyaml")

    try:
        import dotenv  # noqa: F401
    except ImportError:
        missing.append("python-dotenv")

    if missing:
        print_error(f"Missing dependencies: {', '.join(missing)}")
        print_info("Install with: pip install " + " ".join(missing))
        print_info("Or use the bootstrap script: ./scripts/bootstrap_data_node.sh")
        return False

    return True


def prompt_yes_no(question: str, default: bool = True) -> bool:
    """Prompt for yes/no answer."""
    suffix = "[Y/n]" if default else "[y/N]"
    while True:
        answer = input(f"  {question} {suffix}: ").strip().lower()
        if answer == "":
            return default
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no"):
            return False
        print("  Please answer 'y' or 'n'")


def prompt_string(question: str, default: Optional[str] = None) -> str:
    """Prompt for string input."""
    if default:
        suffix = f"[{default}]"
    else:
        suffix = ""

    while True:
        answer = input(f"  {question} {suffix}: ").strip()
        if answer == "" and default:
            return default
        if answer:
            return answer
        if default is None:
            print("  Please enter a value")


def prompt_path(question: str, default: Optional[Path] = None) -> Path:
    """Prompt for path input."""
    default_str = str(default) if default else None
    path_str = prompt_string(question, default_str)
    return Path(path_str).expanduser().resolve()


class WizardState:
    """Manages wizard state for resume capability."""

    def __init__(self, path: Path):
        self.path = path
        self.data: dict = {}

    def load(self) -> dict:
        """Load state from file."""
        if self.path.exists():
            try:
                self.data = json.loads(self.path.read_text())
            except Exception:
                self.data = {}
        return self.data

    def save(self) -> None:
        """Save state to file."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data["updated_at"] = datetime.now(timezone.utc).isoformat()
        self.path.write_text(json.dumps(self.data, indent=2))

    def get(self, key: str, default=None):
        """Get a value from state."""
        return self.data.get(key, default)

    def set(self, key: str, value):
        """Set a value in state."""
        self.data[key] = value


def detect_environment() -> dict:
    """Detect the current environment."""
    env = {
        "platform": platform.system(),
        "machine": platform.machine(),
        "hostname": socket.gethostname(),
        "is_raspberry_pi": False,
        "has_external_ssd": False,
        "suggested_data_root": None,
    }

    # Detect Raspberry Pi
    try:
        with open("/proc/cpuinfo") as f:
            if "Raspberry Pi" in f.read():
                env["is_raspberry_pi"] = True
    except Exception:
        pass

    # Also check model file
    try:
        with open("/proc/device-tree/model") as f:
            if "Raspberry Pi" in f.read():
                env["is_raspberry_pi"] = True
    except Exception:
        pass

    # Detect external SSD (common mount points)
    ssd_candidates = [
        Path("/mnt/ssd"),
        Path("/mnt/usb"),
        Path("/media/pi"),
        Path("/mnt/data"),
    ]

    for candidate in ssd_candidates:
        if candidate.exists() and candidate.is_dir():
            # Check if it's a mount point
            try:
                if candidate.stat().st_dev != candidate.parent.stat().st_dev:
                    env["has_external_ssd"] = True
                    env["suggested_data_root"] = candidate / "data_layer"
                    break
            except Exception:
                pass

    # Default data root
    if env["suggested_data_root"] is None:
        if env["is_raspberry_pi"]:
            env["suggested_data_root"] = Path("/mnt/ssd/data_layer")
        else:
            env["suggested_data_root"] = REPO_ROOT / "data_layer"

    return env


def collect_api_keys(state: WizardState, dry_run: bool = False) -> dict:
    """Collect API keys with improved UX.

    If existing keys are found (from .env or state), shows a summary and
    offers to edit specific keys. Otherwise falls back to vendor-by-vendor
    collection.
    """
    print_header("API Key Configuration")

    # Gather existing keys from .env and state
    existing_keys = {}

    # Check .env file
    if DEFAULT_ENV_PATH.exists():
        for line in DEFAULT_ENV_PATH.read_text().splitlines():
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if value and any(key in cfg["env_vars"] for cfg in VENDOR_KEYS.values()):
                    existing_keys[key] = value

    # Check state file
    for vendor, config in VENDOR_KEYS.items():
        for env_var in config["env_vars"]:
            state_val = state.get(f"key_{env_var}")
            if state_val and env_var not in existing_keys:
                existing_keys[env_var] = state_val

    # Also check environment variables
    for vendor, config in VENDOR_KEYS.items():
        for env_var in config["env_vars"]:
            env_val = os.environ.get(env_var)
            if env_val and env_var not in existing_keys:
                existing_keys[env_var] = env_val

    # Show summary of existing keys
    if existing_keys:
        print_success(f"Found {len(existing_keys)} API key(s) configured:")
        for key, value in existing_keys.items():
            masked = value[:4] + "..." + value[-4:] if len(value) > 12 else "***"
            print(f"    - {key}: {masked}")

        # Ask if user wants to edit any
        if not prompt_yes_no("\n  Edit any API keys?", default=False):
            return existing_keys

        # Edit loop
        keys = dict(existing_keys)
        while True:
            # Build numbered list of all possible keys
            all_env_vars = []
            for vendor, config in VENDOR_KEYS.items():
                for env_var in config["env_vars"]:
                    all_env_vars.append((env_var, vendor, config))

            print("\n  Which key to edit?")
            for i, (env_var, vendor, config) in enumerate(all_env_vars, 1):
                status = "[set]" if env_var in keys else "[   ]"
                print(f"    {i}) {status} {env_var} ({vendor})")
            print(f"    0) Done editing")

            choice = input("\n  Enter number: ").strip()
            if choice == "0" or choice == "":
                break

            try:
                idx = int(choice) - 1
                if 0 <= idx < len(all_env_vars):
                    env_var, vendor, config = all_env_vars[idx]
                    current = keys.get(env_var)
                    if current:
                        masked = current[:4] + "..." + current[-4:] if len(current) > 12 else "***"
                        print(f"  Current value: {masked}")
                    value = input(f"  Enter {env_var} (or blank to remove): ").strip()
                    if value:
                        keys[env_var] = value
                        if not dry_run:
                            state.set(f"key_{env_var}", value)
                        print_success(f"{env_var} updated")
                    elif env_var in keys:
                        del keys[env_var]
                        print_info(f"{env_var} removed")
                else:
                    print_warning("Invalid selection")
            except ValueError:
                print_warning("Please enter a number")

        return keys

    # No existing keys - fall through to vendor-by-vendor collection
    print_info("No existing API keys found. Let's configure them.")
    keys = {}

    for vendor, config in VENDOR_KEYS.items():
        req = "required" if config["required"] else "optional"
        print(f"\n  {vendor.upper()} ({req}): {config['description']}")

        for env_var in config["env_vars"]:
            if config["required"]:
                value = prompt_string(f"Enter {env_var}")
            else:
                value = input(f"  Enter {env_var} (or press Enter to skip): ").strip()

            if value:
                keys[env_var] = value
                if not dry_run:
                    state.set(f"key_{env_var}", value)

    return keys


def generate_node_id(env: dict) -> str:
    """Generate a unique node ID."""
    hostname = env["hostname"].replace(".", "-").lower()
    suffix = datetime.now().strftime("%m%d")
    return f"pi-{hostname}-{suffix}"


def write_env_file(
    data_root: Path,
    node_id: str,
    api_keys: dict,
    maintenance_hour: int,
    dry_run: bool = False,
) -> None:
    """Write or update .env file."""
    print_header("Writing Configuration")

    env_path = DEFAULT_ENV_PATH

    # Read existing .env if present
    existing_lines = []
    existing_keys = set()
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                existing_lines.append(line)
                if "=" in line and not line.strip().startswith("#"):
                    key = line.split("=")[0].strip()
                    existing_keys.add(key)

    # Build new entries
    new_entries = {
        "# Data Node Configuration (generated by pi_data_node_wizard.py)": "",
        "TRADEML_ENV": "local",
        "EDGE_NODE_ID": node_id,
        "DATA_ROOT": str(data_root.parent),  # Parent of data_layer
        "STORAGE_BACKEND": "local",
        "NODE_MAINTENANCE_HOUR": str(maintenance_hour),
    }

    # Add inflight limits
    for vendor, limit in DEFAULT_INFLIGHT.items():
        new_entries[f"NODE_MAX_INFLIGHT_{vendor.upper()}"] = str(limit)

    # Add API keys
    for key, value in api_keys.items():
        new_entries[key] = value

    if dry_run:
        print_info("Would write the following to .env:")
        for key, value in new_entries.items():
            if key.startswith("#"):
                print(f"    {key}")
            elif "KEY" in key or "SECRET" in key:
                print(f"    {key}=***")
            else:
                print(f"    {key}={value}")
        return

    # Write new .env
    with open(env_path, "w") as f:
        # Write new entries first
        for key, value in new_entries.items():
            if key.startswith("#"):
                f.write(f"\n{key}\n")
            else:
                f.write(f"{key}={value}\n")

        f.write("\n# Existing configuration\n")

        # Preserve existing entries that weren't overwritten
        for line in existing_lines:
            if "=" in line and not line.strip().startswith("#"):
                key = line.split("=")[0].strip()
                if key not in new_entries:
                    f.write(line)

    print_success(f"Configuration written to {env_path}")


def initialize_database(data_root: Path, dry_run: bool = False) -> None:
    """Initialize the control database."""
    print_header("Initializing Database")

    control_dir = data_root / "control"

    if dry_run:
        print_info(f"Would create database at {control_dir / 'node.sqlite'}")
        return

    # Set DATA_ROOT for the database module
    os.environ["DATA_ROOT"] = str(data_root.parent)

    try:
        from data_node.db import get_db
        db = get_db()
        print_success(f"Database initialized at {db.db_path}")
    except Exception as e:
        print_error(f"Database initialization failed: {e}")
        raise


def initialize_stage(data_root: Path, dry_run: bool = False) -> None:
    """Initialize Stage 0 configuration."""
    print_header("Initializing Stage Configuration")

    if dry_run:
        print_info("Would initialize Stage 0 configuration")
        return

    os.environ["DATA_ROOT"] = str(data_root.parent)

    try:
        from data_node.stages import load_stage_config, get_stage_info

        config = load_stage_config()
        info = get_stage_info()

        print_success(f"Stage {info['current_stage']} initialized: {info['name']}")
        print_info(f"  Universe size: {info['universe_size']} symbols")
        print_info(f"  EOD history: {info['equities_eod_years']} years")
        print_info(f"  Minute history: {info['equities_minute_years']} year(s)")

    except Exception as e:
        print_error(f"Stage initialization failed: {e}")
        raise


def seed_bootstrap_tasks(data_root: Path, dry_run: bool = False) -> int:
    """Seed BOOTSTRAP tasks for Stage 0."""
    print_header("Seeding Bootstrap Tasks")

    if dry_run:
        print_info("Would seed BOOTSTRAP tasks for 100 symbols")
        print_info("  - equities_eod: 5 years of daily data")
        print_info("  - equities_minute: 1 year of minute data")
        return 0

    os.environ["DATA_ROOT"] = str(data_root.parent)

    try:
        from data_node.stages import seed_bootstrap_tasks as seed_tasks
        from data_node.db import get_db

        db = get_db()
        created = seed_tasks(stage=0, previous_stage=None, db=db)

        print_success(f"Seeded {created} BOOTSTRAP tasks")

        # Show queue stats
        stats = db.get_queue_stats()
        print_info(f"  Queue status: {stats}")

        return created

    except Exception as e:
        print_error(f"Bootstrap seeding failed: {e}")
        raise


def offer_start_service(data_root: Path, dry_run: bool = False) -> None:
    """Offer to start the data_node service."""
    print_header("Start Service")

    log_path = data_root.parent / "logs" / "data_node.log"

    if dry_run:
        print_info("Would offer to start data_node service")
        return

    if not prompt_yes_no("Start the data_node service now?", default=True):
        print_info("You can start it later with: ./scripts/run_data_node.sh")
        return

    print_info("Starting data_node...")

    try:
        # Start in background
        subprocess.Popen(
            [sys.executable, "-m", "data_node"],
            cwd=str(REPO_ROOT),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        print_success("data_node service started in background")
        print_info(f"Check logs: tail -f {log_path}")

    except Exception as e:
        print_error(f"Failed to start service: {e}")
        print_info("Start manually with: ./scripts/run_data_node.sh")


def run_wizard(resume: bool = False, dry_run: bool = False) -> None:
    """Run the setup wizard."""
    print_header("Pi Data-Node Setup Wizard")

    if dry_run:
        print_warning("DRY RUN MODE - No changes will be made")

    # Check dependencies before proceeding
    if not check_dependencies():
        print_error("Please install missing dependencies and re-run the wizard.")
        sys.exit(1)

    # Load state
    state = WizardState(STATE_FILE)
    if resume:
        state.load()
        if state.data:
            print_info(f"Resuming from saved state ({state.data.get('updated_at', 'unknown')})")

    # Step 1: Environment detection
    print_header("Environment Detection")
    env = detect_environment()

    print_info(f"Platform: {env['platform']} ({env['machine']})")
    print_info(f"Hostname: {env['hostname']}")

    if env["is_raspberry_pi"]:
        print_success("Raspberry Pi detected")
    else:
        print_warning("Not a Raspberry Pi - using development settings")

    if env["has_external_ssd"]:
        print_success(f"External SSD detected at {env['suggested_data_root'].parent}")
    else:
        print_warning("No external SSD detected - using local storage")

    # Step 2: Storage root selection
    print_header("Storage Configuration")

    saved_root = state.get("data_root")
    default_root = Path(saved_root) if saved_root else env["suggested_data_root"]

    data_root = prompt_path("Data root directory", default_root)

    if not dry_run:
        data_root.mkdir(parents=True, exist_ok=True)
        state.set("data_root", str(data_root))

    print_success(f"Using data root: {data_root}")

    # Step 3: Node ID
    saved_node_id = state.get("node_id")
    default_node_id = saved_node_id or generate_node_id(env)

    node_id = prompt_string("Node ID", default_node_id)

    if not dry_run:
        state.set("node_id", node_id)

    print_success(f"Node ID: {node_id}")

    # Step 4: API keys
    api_keys = collect_api_keys(state, dry_run)

    # Check required keys
    required_missing = []
    for vendor, config in VENDOR_KEYS.items():
        if config["required"]:
            for env_var in config["env_vars"]:
                if env_var not in api_keys:
                    required_missing.append(env_var)

    if required_missing:
        print_error(f"Missing required API keys: {', '.join(required_missing)}")
        if not prompt_yes_no("Continue anyway?", default=False):
            print_info("Wizard cancelled. Re-run when you have the required keys.")
            return

    # Step 5: Maintenance time
    default_hour = int(state.get("maintenance_hour", 2))
    maintenance_hour = int(prompt_string(
        "Maintenance hour (0-23, local time)",
        str(default_hour)
    ))

    if not dry_run:
        state.set("maintenance_hour", maintenance_hour)

    # Step 6: Write configuration
    write_env_file(data_root, node_id, api_keys, maintenance_hour, dry_run)

    # Step 7: Initialize database
    initialize_database(data_root, dry_run)

    # Step 8: Initialize stage
    initialize_stage(data_root, dry_run)

    # Step 9: Seed bootstrap tasks
    seed_bootstrap_tasks(data_root, dry_run)

    # Step 10: Save state
    if not dry_run:
        state.set("completed", True)
        state.set("completed_at", datetime.now(timezone.utc).isoformat())
        state.save()
        print_success(f"State saved to {STATE_FILE}")

    # Step 11: Offer to start service
    offer_start_service(data_root, dry_run)

    # Compute log path for instructions
    log_path = data_root.parent / "logs" / "data_node.log"

    print_header("Setup Complete!")
    print_info("Next steps:")
    print_info("  1. Monitor progress: ./scripts/run_data_node.sh --status")
    print_info(f"  2. View logs: tail -f {log_path}")
    print_info("  3. Start service: ./scripts/run_data_node.sh")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Pi Data-Node Setup Wizard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from saved state",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show planned actions without executing",
    )

    args = parser.parse_args()

    try:
        run_wizard(resume=args.resume, dry_run=args.dry_run)
    except KeyboardInterrupt:
        print("\n\nWizard cancelled.")
        sys.exit(1)
    except Exception as e:
        print_error(f"Wizard failed: {e}")
        raise


if __name__ == "__main__":
    main()
