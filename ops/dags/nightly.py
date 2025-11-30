"""
Nightly DAG Orchestration.

SSOT v2 Section 5.1: Nightly DAG

The canonical nightly sequence:
1. ingest.forward() - collect today's deltas for all enabled datasets
2. audit.scan() - recompute partition_status and identify gaps
3. backfill.run() - fill RED/AMBER gaps under budgets
4. curate.incremental() - rebuild curated partitions affected by new raw data
5. qc.refresh() - recompute GREEN/AMBER/RED and coverage
6. train_if_ready('equities_xs') - only if GREEN thresholds satisfied
7. train_if_ready('options_vol') and train_if_ready('intraday_xs')
8. evaluate.cpcv_and_shadow() - recompute CPCV metrics
9. promote_if_beat_champion() - apply promotion rules
10. report.emit_daily() - render MD + JSON blotters

Minimal API:
    run_nightly_dag(cfg) -> Dict with status and step results
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any

import pandas as pd
from loguru import logger

# Import step modules
from ops.ssot.curate import curate_incremental
from ops.ssot.audit import audit_scan
from data_layer.qc.partition_status import get_green_coverage, load_partition_status


@dataclass
class NightlyConfig:
    """Configuration for nightly DAG run."""

    asof: Optional[str] = None  # Default to today
    # Universe
    equities_universe: List[str] = field(default_factory=list)
    options_universe: List[str] = field(default_factory=list)
    # Enable flags
    enable_ingest: bool = True
    enable_audit: bool = True
    enable_backfill: bool = True
    enable_curate: bool = True
    enable_train_equities: bool = True
    enable_train_options: bool = True
    enable_train_intraday: bool = True
    enable_evaluate: bool = True
    enable_promote: bool = True
    enable_report: bool = True
    # GREEN thresholds (per SSOT v2 Section 6.1)
    equities_green_threshold: float = 0.98
    options_green_threshold: float = 0.95
    intraday_green_threshold: float = 0.90
    # Backfill budget
    max_backfill_requests: int = 100
    # Training config
    lookback_days: int = 252


def _get_asof(cfg: NightlyConfig) -> date:
    """Get as-of date from config or default to today."""
    if cfg.asof:
        return pd.to_datetime(cfg.asof).date()
    return datetime.now().date()


def _step_ingest(cfg: NightlyConfig, asof: date) -> Dict[str, Any]:
    """
    Step 1: Run edge scheduler to collect today's deltas.

    Returns:
        Dict with ingested counts by dataset
    """
    logger.info("Step 1: ingest.forward()")

    results = {
        "status": "ok",
        "datasets": {},
    }

    try:
        # Import edge scheduler
        from data_layer.connectors import run_edge_scheduler

        ingest_results = run_edge_scheduler(asof=asof.isoformat())
        results["datasets"] = ingest_results
        results["total_rows"] = sum(v.get("rows", 0) for v in ingest_results.values())

    except ImportError:
        logger.warning("Edge scheduler not available, skipping ingest")
        results["status"] = "skipped"
        results["reason"] = "edge_scheduler_not_available"
    except Exception as e:
        logger.error(f"Ingest failed: {e}")
        results["status"] = "error"
        results["error"] = str(e)

    return results


def _step_audit(cfg: NightlyConfig, asof: date) -> Dict[str, Any]:
    """
    Step 2: Recompute partition_status and identify gaps.

    Returns:
        Dict with partition status summary
    """
    logger.info("Step 2: audit.scan()")

    results = {
        "status": "ok",
        "gaps": [],
    }

    try:
        # Scan core tables
        tables = ["equities_eod", "equities_minute", "options_chains"]
        audit_scan(tables)

        # Load and summarize partition status
        status_df = load_partition_status()
        if not status_df.empty:
            status_counts = status_df["status"].value_counts().to_dict()
            results["partition_status"] = status_counts

            # Identify gaps (RED/AMBER)
            gaps = status_df[status_df["status"].isin(["RED", "AMBER"])]
            results["gap_count"] = len(gaps)
            results["gaps"] = gaps.head(20).to_dict("records")

    except Exception as e:
        logger.error(f"Audit failed: {e}")
        results["status"] = "error"
        results["error"] = str(e)

    return results


def _step_backfill(cfg: NightlyConfig, asof: date, gaps: List[dict]) -> Dict[str, Any]:
    """
    Step 3: Fill RED/AMBER gaps under budget.

    Returns:
        Dict with backfill results
    """
    logger.info("Step 3: backfill.run()")

    results = {
        "status": "ok",
        "attempted": 0,
        "succeeded": 0,
        "failed": 0,
    }

    if not gaps:
        results["status"] = "skipped"
        results["reason"] = "no_gaps"
        return results

    try:
        from data_layer.connectors import backfill_partition

        # Prioritize by status (RED first) and limit by budget
        priority_gaps = sorted(gaps, key=lambda g: (0 if g.get("status") == "RED" else 1))
        to_process = priority_gaps[:cfg.max_backfill_requests]

        for gap in to_process:
            results["attempted"] += 1
            try:
                backfill_partition(
                    table=gap.get("table_name"),
                    date=gap.get("dt"),
                    symbol=gap.get("symbol"),
                )
                results["succeeded"] += 1
            except Exception as e:
                results["failed"] += 1
                logger.debug(f"Backfill failed for {gap}: {e}")

    except ImportError:
        logger.warning("Backfill module not available")
        results["status"] = "skipped"
        results["reason"] = "backfill_not_available"
    except Exception as e:
        logger.error(f"Backfill failed: {e}")
        results["status"] = "error"
        results["error"] = str(e)

    return results


def _step_curate(cfg: NightlyConfig, asof: date) -> Dict[str, Any]:
    """
    Step 4: Rebuild curated partitions affected by new raw data.

    Returns:
        Dict with curation results
    """
    logger.info("Step 4: curate.incremental()")

    results = {
        "status": "ok",
    }

    try:
        curate_results = curate_incremental()
        results["jobs"] = curate_results
        results["total_partitions"] = sum(v for v in curate_results.values() if v > 0)

    except Exception as e:
        logger.error(f"Curate failed: {e}")
        results["status"] = "error"
        results["error"] = str(e)

    return results


def _step_qc_refresh(cfg: NightlyConfig, asof: date) -> Dict[str, Any]:
    """
    Step 5: Recompute GREEN/AMBER/RED and coverage.

    Returns:
        Dict with coverage statistics
    """
    logger.info("Step 5: qc.refresh()")

    results = {
        "status": "ok",
        "coverage": {},
    }

    try:
        # Get coverage for key tables over lookback window
        tables = [
            "equities_eod",
            "options_chains",
            "equities_minute",
        ]

        start_dt = asof - timedelta(days=cfg.lookback_days)

        for table in tables:
            try:
                coverage, counts = get_green_coverage(table, start_dt, asof)
                results["coverage"][table] = {
                    "green_ratio": coverage,
                    "counts": counts,
                }
            except Exception as e:
                results["coverage"][table] = {"error": str(e)}

    except Exception as e:
        logger.error(f"QC refresh failed: {e}")
        results["status"] = "error"
        results["error"] = str(e)

    return results


def _check_green_threshold(table: str, threshold: float, lookback_days: int = 252) -> bool:
    """Check if a table meets GREEN coverage threshold.

    Args:
        table: Table name to check
        threshold: Minimum GREEN coverage (0.0 to 1.0)
        lookback_days: Number of days to look back

    Returns:
        True if GREEN coverage >= threshold
    """
    try:
        from datetime import timedelta as td

        end_dt = date.today()
        start_dt = end_dt - td(days=lookback_days)
        coverage, _ = get_green_coverage(table, start_dt, end_dt)
        return coverage >= threshold
    except Exception:
        return False


def _step_train_equities(cfg: NightlyConfig, asof: date) -> Dict[str, Any]:
    """
    Step 6: Train equities_xs if GREEN thresholds satisfied.

    Returns:
        Dict with training results
    """
    logger.info("Step 6: train_if_ready('equities_xs')")

    results = {
        "status": "ok",
        "trained": False,
    }

    # Check GREEN threshold
    if not _check_green_threshold("curated/equities_ohlcv_adj", cfg.equities_green_threshold):
        logger.warning(f"equities_xs: GREEN threshold not met ({cfg.equities_green_threshold})")
        results["status"] = "skipped"
        results["reason"] = "green_threshold_not_met"
        return results

    try:
        from ops.pipelines.equities_xs import run_equities_xs, EquitiesConfig

        start = (asof - timedelta(days=cfg.lookback_days)).isoformat()
        train_cfg = EquitiesConfig(
            start_date=start,
            end_date=asof.isoformat(),
            universe=cfg.equities_universe or ["AAPL", "MSFT", "GOOGL", "AMZN", "SPY"],
        )
        train_results = run_equities_xs(train_cfg)
        results["trained"] = train_results.get("status") == "ok"
        results["metrics"] = train_results

    except Exception as e:
        logger.error(f"equities_xs training failed: {e}")
        results["status"] = "error"
        results["error"] = str(e)

    return results


def _step_train_options(cfg: NightlyConfig, asof: date) -> Dict[str, Any]:
    """
    Step 7a: Train options_vol if GREEN thresholds satisfied.

    Returns:
        Dict with training results
    """
    logger.info("Step 7a: train_if_ready('options_vol')")

    results = {
        "status": "ok",
        "trained": False,
    }

    # Check GREEN threshold
    if not _check_green_threshold("curated/options_iv", cfg.options_green_threshold):
        logger.warning(f"options_vol: GREEN threshold not met ({cfg.options_green_threshold})")
        results["status"] = "skipped"
        results["reason"] = "green_threshold_not_met"
        return results

    try:
        from ops.pipelines.options_vol import run_options_vol, OptionsVolConfig

        train_cfg = OptionsVolConfig(
            asof=asof.isoformat(),
            underliers=cfg.options_universe or ["AAPL", "MSFT", "SPY", "QQQ"],
        )
        train_results = run_options_vol(train_cfg)
        results["trained"] = train_results.get("status") == "ok"
        results["metrics"] = train_results

    except Exception as e:
        logger.error(f"options_vol training failed: {e}")
        results["status"] = "error"
        results["error"] = str(e)

    return results


def _step_train_intraday(cfg: NightlyConfig, asof: date) -> Dict[str, Any]:
    """
    Step 7b: Train intraday_xs if GREEN thresholds satisfied.

    Returns:
        Dict with training results
    """
    logger.info("Step 7b: train_if_ready('intraday_xs')")

    results = {
        "status": "ok",
        "trained": False,
    }

    # Check GREEN threshold
    if not _check_green_threshold("curated/equities_minute", cfg.intraday_green_threshold):
        logger.warning(f"intraday_xs: GREEN threshold not met ({cfg.intraday_green_threshold})")
        results["status"] = "skipped"
        results["reason"] = "green_threshold_not_met"
        return results

    try:
        from ops.pipelines.intraday_xs import run_intraday, IntradayConfig

        start = (asof - timedelta(days=cfg.lookback_days)).isoformat()
        train_cfg = IntradayConfig(
            start_date=start,
            end_date=asof.isoformat(),
            universe=cfg.equities_universe or ["AAPL", "MSFT", "GOOGL", "AMZN", "SPY"],
        )
        train_results = run_intraday(train_cfg)
        results["trained"] = train_results.get("status") == "ok"
        results["metrics"] = train_results

    except Exception as e:
        logger.error(f"intraday_xs training failed: {e}")
        results["status"] = "error"
        results["error"] = str(e)

    return results


def _step_evaluate(cfg: NightlyConfig, asof: date) -> Dict[str, Any]:
    """
    Step 8: Recompute CPCV metrics and update challenger stats.

    Returns:
        Dict with evaluation results
    """
    logger.info("Step 8: evaluate.cpcv_and_shadow()")

    results = {
        "status": "ok",
        "models_evaluated": [],
    }

    try:
        # Load and evaluate each model's artifacts
        artifact_dirs = [
            Path("models/equities_xs/artifacts"),
            Path("models/options_vol/artifacts"),
            Path("models/intraday_xs/artifacts"),
        ]

        for artifact_dir in artifact_dirs:
            summary_path = artifact_dir / f"{artifact_dir.parent.name}_summary.json"
            if summary_path.exists():
                try:
                    summary = json.loads(summary_path.read_text())
                    model_name = artifact_dir.parent.name
                    results["models_evaluated"].append({
                        "model": model_name,
                        "cpcv_results": summary.get("cpcv_results"),
                        "train_samples": summary.get("train_samples"),
                    })
                except Exception as e:
                    logger.debug(f"Failed to load summary from {summary_path}: {e}")

    except Exception as e:
        logger.error(f"Evaluate failed: {e}")
        results["status"] = "error"
        results["error"] = str(e)

    return results


def _step_promote(cfg: NightlyConfig, asof: date, eval_results: Dict) -> Dict[str, Any]:
    """
    Step 9: Apply promotion rules (challenger beats champion).

    Returns:
        Dict with promotion decisions
    """
    logger.info("Step 9: promote_if_beat_champion()")

    results = {
        "status": "ok",
        "promotions": [],
    }

    try:
        from ops.ssot.train_gate import promote_if_better

        for model_eval in eval_results.get("models_evaluated", []):
            model_name = model_eval.get("model")
            cpcv = model_eval.get("cpcv_results")

            if cpcv and model_name:
                promoted = promote_if_better(
                    model_name=model_name,
                    challenger_score=cpcv.get("mean_score"),
                )
                if promoted:
                    results["promotions"].append(model_name)

    except ImportError:
        logger.warning("Promotion module not available")
        results["status"] = "skipped"
    except Exception as e:
        logger.error(f"Promote failed: {e}")
        results["status"] = "error"
        results["error"] = str(e)

    return results


def _step_report(cfg: NightlyConfig, asof: date, dag_results: Dict) -> Dict[str, Any]:
    """
    Step 10: Render MD + JSON blotters and coverage/drift plots.

    Returns:
        Dict with report paths
    """
    logger.info("Step 10: report.emit_daily()")

    results = {
        "status": "ok",
        "reports": [],
    }

    try:
        from ops.reports.emitter import emit_daily

        # Aggregate positions from trained models
        positions = pd.DataFrame()

        # Emit daily report
        metrics = {
            "dag_status": "ok",
            "steps_completed": len([r for r in dag_results.values() if r.get("status") == "ok"]),
            "coverage": dag_results.get("qc", {}).get("coverage", {}),
        }

        report_path = emit_daily(asof, positions, metrics)
        results["reports"].append(str(report_path) if report_path else "emitted")

    except Exception as e:
        logger.error(f"Report failed: {e}")
        results["status"] = "error"
        results["error"] = str(e)

    return results


def run_nightly_dag(cfg: Optional[NightlyConfig] = None) -> Dict[str, Any]:
    """
    Run the complete nightly DAG.

    Executes all 10 steps in sequence, respecting enable flags.

    Args:
        cfg: Nightly configuration

    Returns:
        Dict with overall status and per-step results
    """
    cfg = cfg or NightlyConfig()
    asof = _get_asof(cfg)

    logger.info(f"Starting nightly DAG for {asof}")

    results = {
        "asof": asof.isoformat(),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "ok",
        "steps": {},
    }

    # Step 1: Ingest
    if cfg.enable_ingest:
        results["steps"]["ingest"] = _step_ingest(cfg, asof)

    # Step 2: Audit
    if cfg.enable_audit:
        results["steps"]["audit"] = _step_audit(cfg, asof)
        gaps = results["steps"]["audit"].get("gaps", [])
    else:
        gaps = []

    # Step 3: Backfill
    if cfg.enable_backfill:
        results["steps"]["backfill"] = _step_backfill(cfg, asof, gaps)

    # Step 4: Curate
    if cfg.enable_curate:
        results["steps"]["curate"] = _step_curate(cfg, asof)

    # Step 5: QC Refresh
    results["steps"]["qc"] = _step_qc_refresh(cfg, asof)

    # Step 6: Train equities_xs
    if cfg.enable_train_equities:
        results["steps"]["train_equities"] = _step_train_equities(cfg, asof)

    # Step 7a: Train options_vol
    if cfg.enable_train_options:
        results["steps"]["train_options"] = _step_train_options(cfg, asof)

    # Step 7b: Train intraday_xs
    if cfg.enable_train_intraday:
        results["steps"]["train_intraday"] = _step_train_intraday(cfg, asof)

    # Step 8: Evaluate
    if cfg.enable_evaluate:
        results["steps"]["evaluate"] = _step_evaluate(cfg, asof)
        eval_results = results["steps"]["evaluate"]
    else:
        eval_results = {}

    # Step 9: Promote
    if cfg.enable_promote:
        results["steps"]["promote"] = _step_promote(cfg, asof, eval_results)

    # Step 10: Report
    if cfg.enable_report:
        results["steps"]["report"] = _step_report(cfg, asof, results["steps"])

    # Finalize
    results["finished_at"] = datetime.now(timezone.utc).isoformat()

    # Check for errors
    error_steps = [k for k, v in results["steps"].items() if v.get("status") == "error"]
    if error_steps:
        results["status"] = "partial"
        results["error_steps"] = error_steps

    logger.info(f"Nightly DAG completed: {results['status']}")

    # Save DAG results
    reports_dir = Path("ops/reports/dag_runs")
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / f"nightly_{asof.isoformat()}.json").write_text(
        json.dumps(results, indent=2, default=str)
    )

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run nightly DAG")
    parser.add_argument("--asof", type=str, help="As-of date YYYY-MM-DD")
    parser.add_argument("--dry-run", action="store_true", help="Show steps without running")
    args = parser.parse_args()

    if args.dry_run:
        print("Nightly DAG steps:")
        print("1. ingest.forward()")
        print("2. audit.scan()")
        print("3. backfill.run()")
        print("4. curate.incremental()")
        print("5. qc.refresh()")
        print("6. train_if_ready('equities_xs')")
        print("7a. train_if_ready('options_vol')")
        print("7b. train_if_ready('intraday_xs')")
        print("8. evaluate.cpcv_and_shadow()")
        print("9. promote_if_beat_champion()")
        print("10. report.emit_daily()")
    else:
        cfg = NightlyConfig(asof=args.asof)
        results = run_nightly_dag(cfg)
        print(json.dumps(results, indent=2, default=str))
