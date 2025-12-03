#!/usr/bin/env python3
"""
Node: Single-process orchestrator for Raspberry Pi edge + SSOT cycle.

Behavior:
- Performs self-checks (S3 connectivity, env, vendor keys) and logs actionable alerts.
- Enters a resilient loop: edge (forward) -> audit -> backfill -> curate -> audit(refresh) -> sleep.
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
from uuid import uuid4

from dotenv import load_dotenv
from loguru import logger

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

if sys.version_info < (3, 11):
    raise RuntimeError("Python 3.11+ is required for node orchestration; upgrade the interpreter.")


def _require_env(keys):
    return [k for k in keys if not os.getenv(k)]


def _check_local_storage(root: Path) -> tuple[bool, str]:
    try:
        root.mkdir(parents=True, exist_ok=True)
        test_file = root / f".healthcheck-{int(time.time())}.json"
        payload = {"ok": True, "ts": datetime.utcnow().isoformat()}
        test_file.write_text(json.dumps(payload))
        data = json.loads(test_file.read_text())
        test_file.unlink(missing_ok=True)
        ok = data.get("ok") is True
        return (True, "Local storage OK") if ok else (False, "Local storage unexpected payload")
    except Exception as e:
        return False, f"Local storage failed: {e}"


def _check_s3_storage() -> tuple[bool, str]:
    try:
        from data_layer.storage.s3_client import S3Client

        client = S3Client()
        key = f"healthchecks/node-selfcheck-{uuid4().hex}.json"
        payload = json.dumps({"ok": True, "ts": datetime.utcnow().isoformat()}).encode()
        client.put_object(key, payload)
        client.delete_object(key)
        return True, "S3 storage OK"
    except Exception as e:
        return False, f"S3 storage failed: {e}"


def selfcheck(verbose=True) -> bool:
    errors = []
    warns = []

    backend = os.getenv("STORAGE_BACKEND", "local").lower()
    data_root = Path(os.getenv("DATA_ROOT", REPO_ROOT / "data"))

    if backend != "local":
        warns.append(f"STORAGE_BACKEND={backend}; SSOT default is local SSD (no object store)")
        s3_ok, s3_msg = _check_s3_storage()
        if not s3_ok:
            errors.append(s3_msg)
        elif verbose:
            logger.info(s3_msg)

    ok, msg = _check_local_storage(data_root)
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


def _env_true(name: str, default: bool = True) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return str(val).lower() in {"1", "true", "yes", "on"}


def _resolve_data_root() -> tuple[Path, list[str]]:
    """
    Choose a usable DATA_ROOT. If the configured path is unreachable (e.g., stale
    Mac path on a Pi), fall back to repo-local data with a warning. Also attempts
    to reuse the wizard's persisted state when present.
    """
    warnings: list[str] = []
    env_root = os.getenv("DATA_ROOT")
    candidates = []
    if env_root:
        env_path = Path(env_root).expanduser()
        if sys.platform.startswith("linux") and str(env_path).startswith("/Users/"):
            warnings.append(f"Ignoring macOS-style DATA_ROOT {env_path} on linux")
        else:
            candidates.append(env_path)

    # Wizard state (if reachable)
    state_path = REPO_ROOT / "logs" / "rpi_wizard_state.json"
    try:
        if state_path.exists():
            import json as _json

            state_data = _json.loads(state_path.read_text())
            state_root = state_data.get("data_root")
            if state_root:
                candidates.append(Path(state_root).expanduser())
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"Could not read wizard state ({exc}); skipping")

    candidates.append(REPO_ROOT / "data")

    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            return candidate, warnings
        except Exception as exc:  # noqa: BLE001 - want raw message
            warnings.append(f"DATA_ROOT {candidate} unusable ({exc}); trying next option")

    # Last resort: current working directory /data
    fallback = Path.cwd() / "data"
    fallback.mkdir(parents=True, exist_ok=True)
    warnings.append(f"Falling back to {fallback}")
    return fallback, warnings


def main():
    parser = argparse.ArgumentParser(description="Node: collector+curator orchestrator")
    parser.add_argument("--selfcheck", action="store_true", help="Run checks and exit")
    parser.add_argument("--interval", type=int, default=int(os.getenv("RUN_INTERVAL_SECONDS", "900")), help="Sleep between cycles (seconds)")
    parser.add_argument("--edge-config", default="configs/edge.yml")
    parser.add_argument("--curator-config", default="configs/curator.yml")
    args = parser.parse_args()

    # Load .env but keep explicit env overrides (passed by wizard) intact
    load_dotenv(dotenv_path=REPO_ROOT / ".env", override=False)

    # Storage root for logs/data (external SSD recommended)
    data_root, data_root_warnings = _resolve_data_root()
    log_dir = data_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.remove()  # avoid duplicate handlers
    logger.add(log_dir / "node.log", rotation="10 MB", retention=5, enqueue=True, backtrace=False, diagnose=False)
    for w in data_root_warnings:
        logger.warning(w)

    enable_edge = _env_true("NODE_ENABLE_EDGE", True)
    enable_audit = _env_true("NODE_ENABLE_AUDIT", True)
    enable_backfill = _env_true("NODE_ENABLE_BACKFILL", True)
    enable_curate = _env_true("NODE_ENABLE_CURATE", True)
    logger.info(
        "Toggles: edge={} audit={} backfill={} curate={}",
        "on" if enable_edge else "off",
        "on" if enable_audit else "off",
        "on" if enable_backfill else "off",
        "on" if enable_curate else "off",
    )

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
            if enable_edge:
                try:
                    run_edge_once(args.edge_config)
                except Exception as e:
                    logger.exception(f"Collector failed: {e}")
            else:
                logger.debug("Edge stage disabled (NODE_ENABLE_EDGE=false)")

            # Reference (corp actions, delistings) - fetched when API key exists
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
            if enable_audit:
                try:
                    from ops.ssot import audit_scan
                    audit_scan(["equities_eod", "equities_minute", "options_chains"])  # extend when more tables are supported
                except Exception as e:
                    logger.exception(f"Audit failed: {e}")

            if enable_backfill:
                try:
                    from ops.ssot import backfill_run
                    backfill_run(budget=None)
                except Exception as e:
                    logger.exception(f"Backfill failed: {e}")

            if enable_curate:
                try:
                    from ops.ssot import curate_incremental
                    curate_incremental()
                except Exception as e:
                    logger.exception(f"Curator failed: {e}")

            # Refresh completeness after curation
            if enable_audit:
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

            # Drift snapshot (features baseline) - best-effort
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
                _asof = date.today()
                if os.getenv("NODE_ENABLE_OPTIONS_IV", "true").lower() == "true":
                    from ops.ssot.options import build_iv, fit_surfaces
                    uni_file = REPO_ROOT / "data_layer" / "reference" / "universe_symbols.txt"
                    underliers = [s.strip() for s in uni_file.read_text().splitlines() if s.strip()][:20] if uni_file.exists() else ["AAPL", "MSFT"]
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
                # Options QA: compare Alpaca vs Finnhub chains for today
                try:
                    if os.getenv("NODE_ENABLE_OPTIONS_QA", "true").lower() == "true":
                        from ops.monitoring.options_qa import chain_consistency_report
                        uni_file = REPO_ROOT / "data_layer" / "reference" / "universe_symbols.txt"
                        qa_underliers = [s.strip() for s in uni_file.read_text().splitlines() if s.strip()][:10] if uni_file.exists() else ["AAPL", "MSFT"]
                        chain_consistency_report(_asof, qa_underliers)
                except Exception as _e:
                    logger.warning(f"Options QA failed: {_e}")
                # Polygon options sampling (contracts + aggregates) - best effort
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
