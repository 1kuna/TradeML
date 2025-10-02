#!/usr/bin/env python3
"""
Node: Single-process orchestrator for collector + curator.

Behavior:
- Always runs self-checks first (S3 connectivity, required env, vendor API keys).
- Then enters a resilient loop: edge collector → curator → sleep.
- Resumes safely via bookmarks (collector) and watermarks (curator).

Usage:
  python scripts/node.py              # self-check + loop (default)
  python scripts/node.py --selfcheck  # run checks and exit

Environment:
- STORAGE_BACKEND must be 's3' for Pi/MinIO; local is supported for dev.
- RUN_INTERVAL_SECONDS controls sleep between cycles (default 900s).
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
from loguru import logger

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def _require_env(keys):
    return [k for k in keys if not os.getenv(k)]


def _check_s3():
    try:
        from data_layer.storage.s3_client import get_s3_client
    except Exception as e:
        return False, f"Cannot import S3 client: {e}"

    try:
        s3 = get_s3_client()
        key = f"manifests/healthcheck-{int(time.time())}.json"
        payload = {"ok": True, "ts": datetime.utcnow().isoformat()}
        s3.put_object(key, json.dumps(payload).encode("utf-8"))
        data, _ = s3.get_object(key)
        s3.delete_object(key)
        ok = json.loads(data.decode("utf-8")).get("ok") is True
        return (True, "S3 connectivity OK") if ok else (False, "S3 connectivity unexpected payload")
    except Exception as e:
        return False, f"S3 connectivity failed: {e}"


def selfcheck(verbose=True) -> bool:
    errors = []
    warns = []

    backend = os.getenv("STORAGE_BACKEND", "s3").lower()
    if backend != "s3":
        warns.append("STORAGE_BACKEND is not 's3'. Running in local mode (dev only).")

    # S3
    s3_required = ["S3_ENDPOINT", "S3_BUCKET", "S3_REGION", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]
    missing = _require_env(s3_required)
    if missing:
        errors.append("Missing S3 env vars: " + ", ".join(missing))
    else:
        ok, msg = _check_s3()
        if not ok:
            errors.append(msg)

    # Vendor keys (start minimal with Alpaca)
    vendor_required = ["ALPACA_API_KEY", "ALPACA_SECRET_KEY"]
    missing = _require_env(vendor_required)
    if missing:
        errors.append("Missing data vendor API keys: " + ", ".join(missing))

    if verbose:
        for w in warns:
            logger.warning(w)
        if errors:
            for e in errors:
                logger.error(e)
        else:
            logger.info("All self-checks passed")
    return len(errors) == 0


def run_edge_once(config_path="configs/edge.yml"):
    from scripts.edge_collector import EdgeCollector
    c = EdgeCollector(config_path)
    c.run()


def run_curator_once(config_path="configs/curator.yml"):
    from scripts.curator import Curator
    Curator(config_path).run()


def main():
    parser = argparse.ArgumentParser(description="Node: collector+curator orchestrator")
    parser.add_argument("--selfcheck", action="store_true", help="Run checks and exit")
    parser.add_argument("--interval", type=int, default=int(os.getenv("RUN_INTERVAL_SECONDS", "900")), help="Sleep between cycles (seconds)")
    parser.add_argument("--edge-config", default="configs/edge.yml")
    parser.add_argument("--curator-config", default="configs/curator.yml")
    args = parser.parse_args()

    load_dotenv(dotenv_path=REPO_ROOT / ".env", override=False)

    # File logger
    log_dir = REPO_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.add(log_dir / "node.log", rotation="10 MB", retention=5)

    if args.selfcheck:
        ok = selfcheck()
        sys.exit(0 if ok else 2)

    # Default: self-check + loop
    if not selfcheck():
        sys.exit(2)
    logger.info("Starting node loop: collector -> curator -> sleep")

    while True:
        start = time.time()
        try:
            run_edge_once(args.edge_config)
        except Exception as e:
            logger.exception(f"Collector failed: {e}")

        try:
            run_curator_once(args.curator_config)
        except Exception as e:
            logger.exception(f"Curator failed: {e}")

        elapsed = int(time.time() - start)
        sleep_for = max(1, args.interval - elapsed)
        logger.info(f"Cycle complete in {elapsed}s; sleeping {sleep_for}s")
        try:
            time.sleep(sleep_for)
        except KeyboardInterrupt:
            logger.warning("Interrupted; exiting")
            break


if __name__ == "__main__":
    main()

