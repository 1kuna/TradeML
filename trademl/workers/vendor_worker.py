from __future__ import annotations

"""
Process-per-vendor worker entrypoint.

Env (or CLI overrides):
  - VENDOR: one of alpaca|polygon|finnhub|fred (required)
  - CONCURRENCY: integer inflight cap for this vendor (optional)
  - RPM_LIMIT: requests per minute for this vendor (optional)
  - BUDGET: integer token budget per day (0 or empty = unlimited) (optional)
  - EDGE_CONFIG: path to edge config (default configs/edge.yml)
  - TASKS: optional comma-separated task list to restrict for this vendor

Starts a single VendorRunner with its own lease, executor, and budget.
"""

import argparse
import os
import signal
import sys
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from loguru import logger

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def _normalize_vendor(v: str) -> str:
    m = (v or "").strip().lower()
    if m not in ("alpaca", "polygon", "finnhub", "fred"):
        raise ValueError(f"Unsupported vendor: {v}")
    return m


def _tasks_for_vendor_all(edge, vendor: str) -> List[str]:
    # Mirror mapping logic from VendorSupervisor
    mapping = {
        "alpaca": [
            "alpaca_bars",
            "alpaca_minute",
            "alpaca_options_bars",
            "alpaca_options_chain",
            "alpaca_corporate_actions",
        ],
        "polygon": ["polygon_bars"],
        "finnhub": ["finnhub_options"],
        "fred": ["fred_treasury"],
    }
    cfg_tasks = edge.config.get("tasks", []) if isinstance(edge.config, dict) else []
    # Keep only tasks that are in config and match this vendor
    allowed = mapping.get(vendor, [])
    return [t for t in cfg_tasks if t in allowed]


def _set_vendor_caps_env(vendor: str, conc: Optional[int]):
    if conc and conc > 0:
        os.environ[f"NODE_MAX_INFLIGHT_{vendor.upper()}"] = str(int(conc))


def _apply_vendor_rpm(edge, vendor: str, rpm: Optional[int]):
    if not rpm or rpm <= 0:
        return
    rps = max(0.001, float(rpm) / 60.0)
    conn = edge.connectors.get(vendor)
    if not conn:
        return
    try:
        conn.rate_limit_per_sec = rps
        conn._min_request_interval = 1.0 / rps  # local fallback pacing
        logger.info(f"Adjusted {vendor} rate to {rps:.3f} rps (~{rpm}/min)")
    except Exception as e:
        logger.warning(f"Failed to set rate for {vendor}: {e}")


def _build_budget(vendor: str, budget_tokens: Optional[int]):
    if budget_tokens is None:
        return None
    try:
        n = int(budget_tokens)
    except Exception:
        return None
    if n <= 0:
        # Unlimited: disable explicit budget
        return None
    try:
        from ops.ssot.budget import BudgetManager
        return BudgetManager(initial_limits={vendor: n}, s3_client=None)
    except Exception as e:
        logger.warning(f"BudgetManager unavailable; continuing without explicit budget: {e}")
        return None


def main():
    load_dotenv(dotenv_path=REPO_ROOT / ".env", override=True)

    parser = argparse.ArgumentParser(description="Per-vendor worker")
    parser.add_argument("--vendor", default=os.getenv("VENDOR"), required=False)
    parser.add_argument("--concurrency", type=int, default=int(os.getenv("CONCURRENCY", "0")))
    parser.add_argument("--rpm", type=int, default=int(os.getenv("RPM_LIMIT", "0")))
    parser.add_argument("--budget", type=int, default=os.getenv("BUDGET") or 0)
    parser.add_argument("--config", default=os.getenv("EDGE_CONFIG", "configs/edge.yml"))
    parser.add_argument("--tasks", default=os.getenv("TASKS", ""))
    args = parser.parse_args()

    if not args.vendor:
        raise SystemExit("VENDOR not specified (env or --vendor)")
    vendor = _normalize_vendor(args.vendor)

    # File logger per vendor
    log_dir = REPO_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.add(log_dir / f"{vendor}.log", rotation="10 MB", retention=5)

    from legacy.scripts.edge_collector import EdgeCollector
    edge = EdgeCollector(args.config)

    # Respect per-process vendor concurrency cap via existing env hook
    _set_vendor_caps_env(vendor, args.concurrency)

    # Optionally override RPM via connector pacing
    _apply_vendor_rpm(edge, vendor, args.rpm)

    # Task selection
    if args.tasks:
        tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    else:
        tasks = _tasks_for_vendor_all(edge, vendor)

    if not tasks:
        logger.info(f"No tasks configured for vendor {vendor}; exiting")
        return

    # Budget: use repo-configured budgets by default; allow override via --budget
    try:
        default_budget = edge._init_budget()
    except Exception:
        default_budget = None
    budget_mgr = _build_budget(vendor, args.budget)
    if budget_mgr is None:
        budget_mgr = default_budget
    else:
        # If overriding, try to persist vendor-specific cap into the shared budget
        try:
            if default_budget and vendor in default_budget.state:
                vb = default_budget.state[vendor]
                # Clamp remaining to new limit to avoid bursts beyond override
                vb.limit = int(args.budget)
                vb.remaining = min(vb.remaining, vb.limit)
                default_budget.state[vendor] = vb
                default_budget._persist()
                budget_mgr = default_budget
        except Exception as e:
            logger.warning(f"Could not persist budget override for {vendor}: {e}")

    # Wire up and run a single vendor runner
    from scripts.scheduler.per_vendor import VendorRunner

    runner = VendorRunner(edge, vendor=vendor, tasks=tasks, budget_mgr=budget_mgr)

    shutdown = {"flag": False}

    def _on_sig(signum, _frame):
        logger.warning(f"{vendor}: received signal {signum}; shutting down")
        shutdown["flag"] = True
        try:
            runner.stop(timeout=2.0)
        except Exception:
            pass

    try:
        signal.signal(signal.SIGINT, _on_sig)
        signal.signal(signal.SIGTERM, _on_sig)
    except Exception:
        pass

    runner.start()

    # Block while runner thread is alive
    try:
        while not shutdown["flag"]:
            th = runner._runner_thread
            if th and th.is_alive():
                th.join(timeout=1.0)
                continue
            break
    finally:
        try:
            runner.stop(timeout=2.0)
        except Exception:
            pass


if __name__ == "__main__":
    main()
