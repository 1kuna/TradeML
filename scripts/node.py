#!/usr/bin/env python3
"""
Node: Single-process orchestrator for Raspberry Pi edge + SSOT cycle.

Behavior:
- Performs self-checks (S3 connectivity, env, vendor keys) and logs actionable alerts.
- Enters a resilient loop: edge (forward) → audit → backfill → curate → audit(refresh) → sleep.
- Never exits on transient failures; logs and retries next cycle (hands-off).
- Resumes safely via bookmarks (collector) and watermarks (curator).

Usage:
  python scripts/node.py              # self-check + loop (default)
  python scripts/node.py --selfcheck  # run checks and exit

Environment:
- STORAGE_BACKEND must be 's3' for Pi/MinIO; local is supported for dev.
- RUN_INTERVAL_SECONDS controls sleep between cycles (default 900s).
- Feature toggles (default true): NODE_ENABLE_EDGE, NODE_ENABLE_AUDIT,
  NODE_ENABLE_BACKFILL, NODE_ENABLE_CURATE.
"""

import os
import sys
import time
import json
import argparse
import signal
from pathlib import Path
from datetime import datetime, date

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

    # S3 connectivity check only when using s3 backend
    if backend == "s3":
        has_s3 = all(os.getenv(k) for k in ["S3_ENDPOINT", "S3_BUCKET", "S3_REGION", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"])
        if not has_s3:
            errors.append("Missing S3 env vars (S3_ENDPOINT, S3_BUCKET, S3_REGION, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)")
        else:
            ok, msg = _check_s3()
            if not ok:
                errors.append(msg)

    # Vendor keys (start minimal with Alpaca). Missing keys are warnings; tasks
    # that require them will be skipped and retried next cycle.
    vendor_required = ["ALPACA_API_KEY", "ALPACA_SECRET_KEY"]
    missing = _require_env(vendor_required)
    if missing:
        warns.append("Missing data vendor API keys: " + ", ".join(missing))

    # Polygon market status (optional)
    try:
        pkey = os.getenv("POLYGON_API_KEY")
        if pkey:
            from data_layer.connectors.polygon_connector import PolygonConnector
            pc = PolygonConnector(api_key=pkey)
            st = pc.market_status_now()
            if st:
                logger.info(f"Polygon market status: {st.get('market','unknown')} (server: {st.get('serverTime','n/a')})")
    except Exception as e:
        warns.append(f"Polygon status check skipped: {e}")

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

    # Install top-level signal handlers so Ctrl+C exits promptly
    def _handle_sig(signum, _frame):
        logger.warning(f"Received signal {signum}; exiting...")
        raise KeyboardInterrupt

    try:
        signal.signal(signal.SIGINT, _handle_sig)
        signal.signal(signal.SIGTERM, _handle_sig)
    except Exception:
        pass

    # Default: self-check + loop (never exit on failure; log + retry)
    selfcheck()
    logger.info("Starting node loop: edge -> audit -> backfill -> curate -> audit -> reports -> sleep")

    try:
        while True:
            start = time.time()
            try:
                run_edge_once(args.edge_config)
            except Exception as e:
                logger.exception(f"Collector failed: {e}")

        # Reference (corp actions, delistings) — fetched when API key exists
        try:
            from ops.ssot.reference import update_reference
            update_reference()
        except Exception as e:
            logger.exception(f"Reference update failed: {e}")
        # Delistings snapshot (FMP stable)
        try:
            from ops.reference.delistings import update_delistings_fmp
            if os.getenv("FMP_API_KEY"):
                update_delistings_fmp()
        except Exception as e:
            logger.warning(f"Delistings update failed: {e}")

        # Corporate actions updater for current universe (limited set per cycle)
        try:
            if os.getenv("NODE_ENABLE_CA_UPDATE", "true").lower() == "true":
                from ops.reference.universe import build_universe_from_curated
                from ops.reference.corp_actions import update_corp_actions
                # Build/refresh universe weekly by ADV; then update CA for first 50
                uni_file = REPO_ROOT / "data_layer/reference/universe_symbols.txt"
                if not uni_file.exists():
                    syms = build_universe_from_curated(n_top=1000)[:50]
                else:
                    syms = [s.strip() for s in uni_file.read_text().splitlines() if s.strip()][:50]
                if syms:
                    update_corp_actions(syms)
            # Update reference: index membership (ADV-based) and tick size regime (approximate)
            try:
                from ops.reference.index_membership import update_index_membership
                from ops.reference.tick_size import update_tick_size
                update_index_membership(n_top=int(os.getenv("REF_INDEX_TOPN", "1000")))
                update_tick_size(top_n_half_penny=int(os.getenv("REF_TICKSIZE_HALFPENNY_TOPN", "250")))
            except Exception as _e:
                logger.warning(f"Reference (index/ticksize) update failed: {_e}")
            # Polygon tickers snapshot (limited pages per run)
            try:
                if os.getenv("NODE_ENABLE_POLYGON_TICKERS", "true").lower() == "true" and os.getenv("POLYGON_API_KEY"):
                    from ops.reference.polygon_universe import update_polygon_universe
                    update_polygon_universe(max_pages=int(os.getenv("POLYGON_PAGES_PER_RUN", "1")))
            except Exception as _e:
                logger.warning(f"Polygon tickers snapshot failed: {_e}")
        except Exception as e:
            logger.exception(f"Corp actions updater failed: {e}")

        # SSOT: audit -> backfill -> curate -> audit(refresh)
        try:
            from ops.ssot import audit_scan
            audit_scan(["equities_eod", "equities_minute", "options_chains"])  # extend when more tables are supported
        except Exception as e:
            logger.exception(f"Audit failed: {e}")

        try:
            from ops.ssot import backfill_run
            backfill_run(budget=None)
        except Exception as e:
            logger.exception(f"Backfill failed: {e}")

        try:
            from ops.ssot import curate_incremental
            curate_incremental()
        except Exception as e:
            logger.exception(f"Curator failed: {e}")

        # Refresh completeness after curation
        try:
            from ops.ssot import audit_scan
            audit_scan(["equities_eod", "equities_minute", "options_chains"])  # quick refresh
        except Exception as e:
            logger.exception(f"Audit refresh failed: {e}")

        # Coverage heatmap report
        try:
            from ops.monitoring.coverage import coverage_heatmap
            coverage_heatmap()
        except Exception as e:
            logger.exception(f"Coverage heatmap failed: {e}")

        # Drift snapshot (features baseline) — best-effort
        try:
            from ops.monitoring.drift_snapshot import drift_snapshot
            # Small universe for snapshot
            uni_file = log_dir.parent / "data_layer" / "reference" / "universe_symbols.txt"
            if uni_file.exists():
                universe = [s.strip() for s in uni_file.read_text().splitlines() if s.strip()][:50]
            else:
                universe = ["AAPL", "MSFT", "GOOGL"]
            drift_snapshot(date.today(), universe)
        except Exception as e:
            logger.exception(f"Drift snapshot failed: {e}")

        # Options IV build + SVI surfaces (best-effort) for today's raw chains
        try:
            if os.getenv("NODE_ENABLE_OPTIONS_IV", "true").lower() == "true":
                from datetime import date as _Date
                from ops.ssot.options import build_iv, fit_surfaces
                uni_file = REPO_ROOT / "data_layer" / "reference" / "universe_symbols.txt"
                underliers = [s.strip() for s in uni_file.read_text().splitlines() if s.strip()][:20] if uni_file.exists() else ["AAPL", "MSFT"]
                _asof = _Date.today()
                try:
                    _ = build_iv(_asof, underliers)
                    logger.info(f"Options IV build status: {_.get('status')}")
                except Exception as _e:
                    logger.warning(f"Options IV build failed: {_e}")
                try:
                    _ = fit_surfaces(_asof, underliers)
                    logger.info(f"SVI surface fit status: {_.get('status')}")
                except Exception as _e:
                    logger.warning(f"Options SVI fit failed: {_e}")
            # Polygon options sampling (contracts + aggregates) — best effort
            try:
                if os.getenv("NODE_ENABLE_POLYGON_OPTIONS", "true").lower() == "true" and os.getenv("POLYGON_API_KEY"):
                    from ops.ssot.polygon_options import sample_and_persist
                    uni_file = REPO_ROOT / "data_layer" / "reference" / "universe_symbols.txt"
                    ul = [s.strip() for s in uni_file.read_text().splitlines() if s.strip()][:2] if uni_file.exists() else ["AAPL", "MSFT"]
                    sample_and_persist(ul, _asof, max_contracts_per_ul=1)
            except Exception as _e:
                logger.warning(f"Polygon options sampling failed: {_e}")
        except Exception as e:
            logger.exception(f"Options IV/SVI step failed: {e}")

        # Optional options strategies from IV (reports only; does not affect ingestion)
        try:
            if os.getenv("NODE_ENABLE_OPTIONS_REPORTS", "true").lower() == "true":
                from ops.ssot.options_strategies import build_and_emit
                # Use small underlier set from universe file if present
                uni_file = REPO_ROOT / "data_layer/reference/universe_symbols.txt"
                syms = [s.strip() for s in uni_file.read_text().splitlines() if s.strip()][:20] if uni_file.exists() else ["AAPL", "MSFT"]
                from datetime import date as _Date
                build_and_emit(_Date.today(), syms)
        except Exception as e:
            logger.exception(f"Options report failed: {e}")

        # Evaluate prior-day options PnL (delta-hedged straddles)
        try:
            if os.getenv("NODE_ENABLE_OPTIONS_EVAL", "true").lower() == "true":
                from datetime import timedelta as _TD
                from ops.ssot.options_eval import evaluate_options_pnl
                asof_prev = date.today() - _TD(days=1)
                _ = evaluate_options_pnl(asof_prev)
        except Exception as e:
            logger.exception(f"Options PnL evaluation failed: {e}")

        # Shadow logging (best-effort): log equities_xs/intraday_xs weights if recent reports exist
        try:
            from ops.ssot.shadow import log_signals
            from datetime import date as _Date
            import json as _json
            import pandas as _pd
            _asof = _Date.today()
            rep_dir = REPO_ROOT / "ops" / "reports"
            # Equities
            eq_json = rep_dir / f"equities_{_asof.isoformat()}.json"
            if eq_json.exists():
                data = _json.loads(eq_json.read_text())
                pos = data.get("positions", [])
                if pos:
                    dfw = _pd.DataFrame(pos)[["symbol", "target_w"]]
                    log_signals(_asof, dfw, out_dir=str(rep_dir / "shadow" / "equities_xs"))
            # Intraday (reuses same emitter naming)
            intr_json = rep_dir / f"equities_{_asof.isoformat()}.json"  # if intraday pipeline used different name, adjust later
            if intr_json.exists():
                pass  # already handled above
        except Exception as e:
            logger.exception(f"Shadow logging failed: {e}")

            elapsed = int(time.time() - start)
            sleep_for = max(1, args.interval - elapsed)
            logger.info(f"Cycle complete in {elapsed}s; sleeping {sleep_for}s")
            try:
                time.sleep(sleep_for)
            except KeyboardInterrupt:
                raise
    except KeyboardInterrupt:
        logger.warning("Interrupted; exiting node")
        return


if __name__ == "__main__":
    main()
