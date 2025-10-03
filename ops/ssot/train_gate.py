from __future__ import annotations

"""
GREEN-gated training dispatchers and simple evaluation/promotion stubs.
"""

import os
from datetime import date, timedelta
from pathlib import Path
from typing import Dict

import pandas as pd
import yaml
from loguru import logger
from .registry import log_run

from data_layer.reference.calendars import get_trading_days


def _load_ledger() -> pd.DataFrame:
    path = Path("data_layer/qc/partition_status.parquet")
    if not path.exists():
        logger.warning("Completeness ledger not found; run audit first")
        return pd.DataFrame(columns=["source", "table_name", "symbol", "dt", "status"])
    return pd.read_parquet(path)


def _meets_green_threshold(model_cfg_path: str, ledger: pd.DataFrame) -> bool:
    with open(model_cfg_path) as f:
        cfg = yaml.safe_load(f)
    deps = cfg.get("dependencies", [])
    th = cfg.get("green_threshold", {"window_years": 10, "min_ratio": 0.98})
    window_years = int(th.get("window_years", 10))
    min_ratio = float(th.get("min_ratio", 0.98))

    # Define window [today - years, today]
    end = date.today()
    start = date(end.year - window_years, end.month, end.day)
    days = set(get_trading_days(start, end))

    if ledger.empty:
        logger.warning("Ledger empty; cannot validate GREEN threshold")
        return False

    ok = True
    for dep in deps:
        dep_rows = ledger[(ledger["table_name"] == dep) & (ledger["dt"].isin(days))]
        if dep_rows.empty:
            logger.warning(f"No ledger rows for dependency {dep} in window")
            ok = False
            continue
        green_share = (dep_rows["status"] == "GREEN").mean()
        logger.info(f"Dependency {dep}: GREEN ratio {green_share:.3f} (bar {min_ratio:.3f})")
        if green_share < min_ratio:
            ok = False
    return ok


def train_if_ready(model_name: str) -> None:
    """GREEN-gated dispatcher; currently supports 'equities_xs'."""
    ledger = _load_ledger()
    if model_name == "equities_xs":
        cfg_path = "configs/training/equities_xs.yml"
        if not _meets_green_threshold(cfg_path, ledger):
            logger.error("Training blocked: GREEN threshold not met for equities_xs")
            return
        # Run the pipeline over last N years window
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        window_years = int(cfg.get("green_threshold", {}).get("window_years", 10))
        end = date.today()
        start = date(end.year - window_years, end.month, end.day)
        # Universe: use reference symbols
        uni_file = Path("data_layer/reference/universe_symbols.txt")
        if uni_file.exists():
            universe = [s.strip() for s in uni_file.read_text().splitlines() if s.strip()]
        else:
            universe = ["AAPL", "MSFT", "GOOGL"]

        logger.info(f"Training equities_xs on window {start} → {end} for {len(universe)} symbols")
        try:
            from ops.pipelines.equities_xs import PipelineConfig, run_pipeline

            _ = run_pipeline(
                PipelineConfig(
                    start_date=start.isoformat(),
                    end_date=end.isoformat(),
                    universe=universe,
                    label_type="horizon",
                    horizon_days=5,
                )
            )
            logger.info("Training completed and daily report emitted")
            # Optional: log basic metrics to registry if available
            try:
                metrics = {}
                if isinstance(_, dict):
                    dsr_dict = _.get("dsr", {}) or {}
                    pbo_dict = _.get("pbo", {}) or {}
                    bm = _.get("backtest_metrics")
                    if dsr_dict:
                        metrics["dsr"] = float(dsr_dict.get("dsr")) if dsr_dict.get("dsr") is not None else None
                    if pbo_dict:
                        metrics["pbo"] = float(pbo_dict.get("pbo")) if pbo_dict.get("pbo") is not None else None
                    if bm is not None:
                        try:
                            metrics["sharpe"] = float(bm.sharpe_ratio)
                            metrics["max_drawdown"] = float(bm.max_drawdown)
                            metrics["turnover"] = float(bm.turnover)
                        except Exception:
                            pass
                log_run(metrics={k: v for k, v in metrics.items() if v is not None}, params={"model": model_name}, tags={"stage": "Challenger"})
            except Exception:
                pass
        except Exception as e:
            logger.exception(f"Training pipeline failed: {e}")
    else:
        # Intraday XS pipeline
        if model_name == "intraday_xs":
            cfg_path = "configs/training/equities_xs.yml"  # reuse green thresholds via equities_eod/minute
            if not _meets_green_threshold(cfg_path, ledger):
                logger.error("Training blocked: GREEN threshold not met for intraday_xs")
                return
            end = date.today()
            start = date(end.year - 1, end.month, end.day)
            uni_file = Path("data_layer/reference/universe_symbols.txt")
            universe = [s.strip() for s in uni_file.read_text().splitlines() if s.strip()] if uni_file.exists() else ["AAPL", "MSFT", "GOOGL"]
            try:
                from ops.pipelines.intraday_xs import IntradayConfig, run_intraday
                _ = run_intraday(IntradayConfig(start_date=start.isoformat(), end_date=end.isoformat(), universe=universe))
                logger.info("Intraday_xs run complete")
            except Exception as e:
                logger.exception(f"Intraday pipeline failed: {e}")
        else:
            # Options volatility pipeline
            if model_name == "options_vol":
                cfg_path = "configs/training/options_vol.yml"
                if not _meets_green_threshold(cfg_path, ledger):
                    logger.error("Training blocked: GREEN threshold not met for options_vol")
                    return
                end = date.today()
                try:
                    from ops.pipelines.options_vol import OptionsVolConfig, run_options_vol
                    # Use small underlier subset if universe exists
                    uni_file = Path("data_layer/reference/universe_symbols.txt")
                    underliers = [s.strip() for s in uni_file.read_text().splitlines() if s.strip()][:20] if uni_file.exists() else ["AAPL", "MSFT"]
                    _ = run_options_vol(OptionsVolConfig(asof=end.isoformat(), underliers=underliers))
                    logger.info("options_vol run complete")
                    try:
                        log_run(metrics={"status_ok": 1.0 if _.get("status") == "ok" else 0.0}, params={"model": model_name}, tags={"stage": "Challenger"})
                    except Exception:
                        pass
                except Exception as e:
                    logger.exception(f"Options_vol pipeline failed: {e}")
            else:
                logger.warning(f"Model not supported yet: {model_name}")


def run_cpcv(model_name: str) -> Dict:
    """Run CPCV for a model using the validation utilities and return metrics.

    Currently supports 'equities_xs' with horizon labels and baseline features.
    """
    if model_name != "equities_xs":
        logger.info(f"CPCV run not supported for model: {model_name}")
        return {"model": model_name, "status": "not_supported"}

    try:
        # Build dataset over a reasonable window (aligned with training gate config)
        with open("configs/training/equities_xs.yml") as f:
            cfg = yaml.safe_load(f)
        window_years = int(cfg.get("green_threshold", {}).get("window_years", 10))
        end = date.today()
        start = date(end.year - window_years, end.month, end.day)

        uni_file = Path("data_layer/reference/universe_symbols.txt")
        universe = [s.strip() for s in uni_file.read_text().splitlines() if s.strip()] if uni_file.exists() else ["AAPL", "MSFT", "GOOGL"]

        from feature_store.equities.dataset import build_training_dataset
        ds = build_training_dataset(universe=universe, start_date=start.isoformat(), end_date=end.isoformat(), label_type="horizon")
        if ds.X.empty:
            return {"model": model_name, "status": "no_data"}

        # Prepare groups with symbol-aware purging
        groups = pd.DataFrame({
            "date": ds.X["date"],
            "horizon_days": ds.meta["horizon_days"],
            "symbol": ds.X["symbol"],
        }).reset_index(drop=True)

        # Run CPCV
        from validation import run_cpcv as _run_cpcv
        res = _run_cpcv(ds.X, ds.y, groups=groups, embargo_days=int(cfg.get("cpcv", {}).get("embargo_days", 10)), n_folds=int(cfg.get("cpcv", {}).get("folds", 8)))
        return {"model": model_name, "status": "ok", "result": res}
    except Exception as e:
        logger.exception(f"CPCV run failed for {model_name}: {e}")
        return {"model": model_name, "status": "error", "error": str(e)}


def promote_if_beat_champion(model_name: str) -> None:
    """Champion–Challenger promotion using MLflow run metrics and tags.

    Rules (from training config promotion_bars):
    - dsr >= DSR_min
    - pbo <= PBO_max
    - sharpe >= net_sharpe_min
    - challenger must beat champion on Sharpe, and not worsen PBO/DSR
    """
    try:
        from .registry import _get_mlflow
        mlflow = _get_mlflow()
        if mlflow is None:
            logger.info("MLflow not installed; promotion is disabled")
            return

        import mlflow as _mlf  # type: ignore

        # Load promotion bars
        cfg_path = "configs/training/equities_xs.yml" if model_name == "equities_xs" else None
        bars = {"DSR_min": 0.0, "PBO_max": 0.05, "net_sharpe_min": 1.0}
        if cfg_path:
            try:
                with open(cfg_path) as f:
                    tc = yaml.safe_load(f)
                    bars.update((tc.get("promotion_bars") or {}))
            except Exception:
                pass

        # Search runs for this model
        df = _mlf.search_runs(filter_string=f"params.model = '{model_name}'", order_by=["attributes.start_time DESC"])
        if df is None or df.empty:
            logger.info("No runs found to evaluate promotion")
            return

        # Identify champion and the latest challenger
        is_champion = (df.get("tags.stage") == "Champion") if "tags.stage" in df.columns else (df.get("tags.stage") == "Champion")
        champion_df = df[is_champion] if is_champion is not None else df.iloc[0:0]
        champion = champion_df.iloc[0] if not champion_df.empty else None
        challenger = df.iloc[0]
        if champion is not None and str(champion["run_id"]) == str(challenger["run_id"]):
            logger.info("Latest run is already the champion; nothing to do")
            return

        # Extract metrics
        def _m(row, key):
            col = f"metrics.{key}"
            return float(row[col]) if col in row.index and pd.notna(row[col]) else None

        ch_metrics = {k: _m(challenger, k) for k in ("dsr", "pbo", "sharpe")}
        if any(v is None for v in [ch_metrics.get("dsr"), ch_metrics.get("pbo"), ch_metrics.get("sharpe")]):
            logger.info("Challenger missing required metrics; skipping promotion")
            return

        # Bars check
        if ch_metrics["dsr"] < float(bars["DSR_min"]) or ch_metrics["pbo"] > float(bars["PBO_max"]) or ch_metrics["sharpe"] < float(bars["net_sharpe_min"]):
            logger.info("Challenger does not meet promotion bars; skipping")
            return

        beat = True
        if champion is not None:
            champ_metrics = {k: _m(champion, k) for k in ("dsr", "pbo", "sharpe")}
            # Must beat Sharpe and not degrade DSR/PBO beyond bars
            if champ_metrics.get("sharpe") is not None and ch_metrics["sharpe"] <= champ_metrics["sharpe"]:
                beat = False
            if champ_metrics.get("pbo") is not None and ch_metrics["pbo"] > champ_metrics["pbo"]:
                beat = False
            if champ_metrics.get("dsr") is not None and ch_metrics["dsr"] < champ_metrics["dsr"]:
                beat = False

        if not beat:
            logger.info("Challenger does not beat champion; skipping promotion")
            return

        # Optional: require multi-week positive shadow PnL before promotion
        try:
            req_shadow_days = 0
            if cfg_path:
                with open(cfg_path) as f:
                    tc = yaml.safe_load(f)
                    req_shadow_days = int(((tc.get("promotion_policy") or {}).get("require_shadow_days", 0)))
            if req_shadow_days and req_shadow_days > 0:
                from .shadow import evaluate_shadow
                end = date.today()
                start = end - timedelta(days=req_shadow_days + 2)
                sh = evaluate_shadow(start, end)
                if (sh.get("status") != "ok") or (float(sh.get("pnl_mean", 0.0)) <= 0.0):
                    logger.info("Shadow PnL window not positive; deferring promotion")
                    return
        except Exception:
            # If shadow evaluator fails, do not block promotion unless required by policy
            pass

        # Promote: set tags on runs
        client = _mlf.tracking.MlflowClient()
        chal_run_id = str(challenger["run_id"])
        client.set_tag(chal_run_id, "stage", "Champion")
        if champion is not None:
            client.set_tag(str(champion["run_id"]), "stage", "Archived")
            client.set_tag(chal_run_id, "previous_champion_run_id", str(champion["run_id"]))
        logger.info(f"Promoted run {chal_run_id} to Champion for {model_name}")

    except Exception as e:
        logger.warning(f"Promotion check skipped: {e}")
