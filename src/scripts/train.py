"""Training entry point."""

from __future__ import annotations

import argparse
import json
import pickle
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import yaml

from trademl.features.equities import build_features
from trademl.features.preprocessing import rank_normalize
from trademl.labels.returns import build_labels
from trademl.models.lgbm import LightGBMModel, tune_lightgbm_via_walk_forward
from trademl.models.ridge import RidgeModel, tune_ridge_via_walk_forward
from trademl.reports.emitter import emit_report
from trademl.validation.diagnostics import ic_by_year, placebo_test, portfolio_cost_stress_test, sign_flip_canary
from trademl.validation.cpcv import combinatorially_purged_cv
from trademl.validation.dsr import deflated_sharpe_ratio
from trademl.validation.pbo import probability_of_backtest_overfitting
from trademl.validation.walk_forward import expanding_walk_forward


def _resolve_effective_end_date(qc: pd.DataFrame, report_date: str | None) -> pd.Timestamp:
    latest_date = pd.Timestamp(qc["date"].max())
    if report_date is None:
        return latest_date
    requested = pd.Timestamp(report_date)
    return min(latest_date, requested)


def _partition_date(path: Path) -> pd.Timestamp:
    return pd.Timestamp(path.parent.name.split("=", 1)[1])


def _planner_window_coverage_ratio(*, db_path: Path, window_start: pd.Timestamp, window_end: pd.Timestamp) -> float | None:
    if not db_path.exists():
        return None
    try:
        with sqlite3.connect(db_path, timeout=5.0) as connection:
            row = connection.execute(
                """
                SELECT SUM(planner_task_progress.expected_units) AS expected_units,
                       SUM(planner_task_progress.completed_units) AS completed_units
                FROM planner_tasks
                LEFT JOIN planner_task_progress
                  ON planner_tasks.task_key = planner_task_progress.task_key
                WHERE planner_tasks.task_family = 'canonical_bars'
                  AND planner_tasks.start_date >= ?
                  AND planner_tasks.end_date <= ?
                """,
                (window_start.strftime("%Y-%m-%d"), window_end.strftime("%Y-%m-%d")),
            ).fetchone()
    except sqlite3.OperationalError:
        return None
    if row is None:
        return None
    expected_units = int(row[0] or 0)
    completed_units = int(row[1] or 0)
    if expected_units <= 0:
        return None
    return min(1.0, completed_units / expected_units)


def _load_curated_panel(curated_files: list[Path]) -> tuple[pd.DataFrame, list[str]]:
    loaded_frames: list[pd.DataFrame] = []
    skipped_partitions: list[str] = []
    for path in curated_files:
        try:
            frame = pd.read_parquet(path)
        except Exception:
            skipped_partitions.append(path.parent.name.partition("=")[2])
            continue
        if frame.empty:
            skipped_partitions.append(path.parent.name.partition("=")[2])
            continue
        loaded_frames.append(frame)
    if not loaded_frames:
        raise FileNotFoundError("no readable curated parquet partitions found for the requested report window")
    return pd.concat(loaded_frames, ignore_index=True), skipped_partitions


def run_training(
    *,
    data_root: Path,
    config_path: Path,
    output_root: Path,
    report_date: str | None = None,
    model_suite: str = "full",
) -> dict:
    """Run the end-to-end training workflow and persist a report."""
    if model_suite not in {"full", "ridge_only"}:
        raise ValueError(f"unsupported model suite: {model_suite}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    qc_path = data_root / "data" / "qc" / "partition_status.parquet"
    curated_root = data_root / "data" / "curated" / "equities_ohlcv_adj"
    if not qc_path.exists():
        raise FileNotFoundError(f"missing partition status parquet: {qc_path}")

    qc = pd.read_parquet(qc_path)
    qc["date"] = pd.to_datetime(qc["date"])
    effective_end_date = _resolve_effective_end_date(qc, report_date)
    qc = qc.loc[qc["date"] <= effective_end_date].copy()
    if qc.empty:
        raise RuntimeError(f"no QC rows available on or before report_date={report_date!r}")
    window_start = effective_end_date - pd.DateOffset(years=int(config["data"].get("window_years", 5)))
    qc_window = qc.loc[
        (qc["dataset"] == "equities_eod")
        & (qc["source"] == "alpaca")
        & (qc["date"].between(window_start, effective_end_date))
    ].copy()
    qc_coverage = float((qc_window["status"] == "GREEN").mean()) if not qc_window.empty else 0.0
    planner_coverage = _planner_window_coverage_ratio(
        db_path=data_root / "control" / "node.sqlite",
        window_start=window_start,
        window_end=effective_end_date,
    )
    coverage = max(qc_coverage, planner_coverage or 0.0)
    missing_dates = sorted(qc_window.loc[qc_window["status"] != "GREEN", "date"].dt.strftime("%Y-%m-%d").unique().tolist())
    if coverage < float(config["data"]["green_threshold"]):
        raise RuntimeError(f"green coverage below threshold: {coverage:.3f}; missing dates={missing_dates[:25]}")

    curated_files = sorted(path for path in curated_root.glob("date=*/data.parquet") if _partition_date(path) <= effective_end_date)
    if not curated_files:
        raise FileNotFoundError(f"no curated parquet files found under {curated_root} on or before {effective_end_date.strftime('%Y-%m-%d')}")
    panel, skipped_curated_partitions = _load_curated_panel(curated_files)

    features = build_features(panel, config["features"])
    labels = build_labels(panel, horizon=5)
    merged = features.merge(labels, on=["date", "symbol"])
    merged = merged.dropna(subset=["label_5d"])
    feature_cols = [
        column
        for column in merged.columns
        if column not in {"date", "symbol", "raw_forward_return_5d", "label_5d", "earnings_within_5d"}
    ]
    normalized = rank_normalize(merged, feature_cols, missing_threshold=float(config["preprocessing"]["missing_threshold"]))

    validation_config = {
        "initial_train_years": int(config["validation"]["initial_train_years"]),
        "step_months": int(str(config["validation"]["step"]).split("_")[0]),
        "purge_days": 5,
    }
    run_ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    ridge_alpha = tune_ridge_via_walk_forward(
        normalized,
        feature_cols,
        "label_5d",
        expanding_walk_forward,
        validation_config,
    )
    ridge_folds = expanding_walk_forward(normalized, feature_cols, "label_5d", lambda: RidgeModel(alpha=ridge_alpha), validation_config)
    lgbm_params: dict | None = None
    lgbm_folds = []
    if model_suite == "full":
        lgbm_params = tune_lightgbm_via_walk_forward(
            normalized,
            feature_cols,
            "label_5d",
            expanding_walk_forward,
            validation_config,
            n_trials=5,
        )
        lgbm_folds = expanding_walk_forward(
            normalized,
            feature_cols,
            "label_5d",
            lambda: LightGBMModel(n_trials=0, best_params=lgbm_params),
            validation_config,
        )

    ridge_predictions = pd.concat([fold.predictions for fold in ridge_folds], ignore_index=True)
    cpcv_results = combinatorially_purged_cv(
        normalized.dropna(subset=["label_5d"]),
        feature_cols,
        "label_5d",
        lambda: RidgeModel(alpha=ridge_alpha),
        n_folds=int(config["validation"].get("cpcv_folds", 8)),
        embargo_days=int(config["validation"].get("embargo_days", 10)),
    )
    diagnostics = {
        "ic_by_year": ic_by_year(ridge_predictions["prediction"], ridge_predictions["label_5d"], ridge_predictions["date"]),
        "placebo": placebo_test(
            normalized.dropna(subset=["label_5d"]),
            feature_cols,
            "label_5d",
            lambda: RidgeModel(alpha=ridge_alpha),
            validation_runner=expanding_walk_forward,
            validation_config=validation_config,
        ),
        "cost_stress": portfolio_cost_stress_test(
            prices=panel[["date", "symbol", "close"]],
            prediction_frame=ridge_predictions,
            multiplier=float(config["portfolio"]["cost_stress_multiplier"]),
            cost_spread_bps=float(config["portfolio"].get("cost_spread_bps", 5.0)),
        ),
        "sign_flip_canary": sign_flip_canary(ridge_predictions, label_col="label_5d"),
        "cpcv": {
            "folds": len(cpcv_results),
            "mean_oos_score": float(pd.Series([result.out_of_sample_score for result in cpcv_results]).mean()) if cpcv_results else 0.0,
            "mean_retention": float(pd.Series([result.retention for result in cpcv_results]).mean()) if cpcv_results else 0.0,
        },
        "pbo": probability_of_backtest_overfitting(cpcv_results),
        "dsr": deflated_sharpe_ratio(
            pd.Series([fold.decile_spread for fold in ridge_folds], dtype=float).dropna().to_numpy(),
            num_trials=max(1, len(cpcv_results)),
        ),
    }

    assessment = _phase_one_assessment(
        ridge_mean_ic=float(pd.Series([fold.rank_ic for fold in ridge_folds]).mean()) if ridge_folds else 0.0,
        ic_by_year_result=diagnostics["ic_by_year"],
        cost_stress=diagnostics["cost_stress"],
        placebo=diagnostics["placebo"],
    )
    report = {
        "coverage": coverage,
        "qc_coverage": qc_coverage,
        "planner_window_coverage": planner_coverage,
        "window_start": window_start.strftime("%Y-%m-%d"),
        "window_end": effective_end_date.strftime("%Y-%m-%d"),
        "missing_dates": missing_dates,
        "skipped_curated_partitions": skipped_curated_partitions,
        "ridge": {
            "alpha": ridge_alpha,
            "folds": [
                {
                    "rank_ic": fold.rank_ic,
                    "decile_spread": fold.decile_spread,
                    "hit_rate": fold.hit_rate,
                    "bucket_returns": fold.bucket_returns,
                }
                for fold in ridge_folds
            ],
            "mean_rank_ic": float(pd.Series([fold.rank_ic for fold in ridge_folds]).mean()) if ridge_folds else 0.0,
            "decile_chart_data": {f"fold_{idx + 1}": fold.bucket_returns for idx, fold in enumerate(ridge_folds)},
        },
        "lightgbm": (
            {
                "folds": [
                    {
                        "rank_ic": fold.rank_ic,
                        "decile_spread": fold.decile_spread,
                        "hit_rate": fold.hit_rate,
                        "bucket_returns": fold.bucket_returns,
                    }
                    for fold in lgbm_folds
                ],
                "mean_rank_ic": float(pd.Series([fold.rank_ic for fold in lgbm_folds]).mean()) if lgbm_folds else 0.0,
                "decile_chart_data": {f"fold_{idx + 1}": fold.bucket_returns for idx, fold in enumerate(lgbm_folds)},
            }
            if model_suite == "full"
            else {"skipped": True, "reason": "ridge_only_phase1_baseline"}
        ),
        "diagnostics": diagnostics,
        "assessment": assessment,
        "model_suite": model_suite,
        "artifacts": _persist_run_artifacts(
            output_root=output_root,
            run_ts=run_ts,
            config=config,
            feature_cols=feature_cols,
            normalized=normalized,
            ridge_model=RidgeModel(alpha=ridge_alpha).fit(normalized[feature_cols].fillna(0.0), normalized["label_5d"]),
            lgbm_model=(
                LightGBMModel(n_trials=0, best_params=lgbm_params).fit(normalized[feature_cols].fillna(0.0), normalized["label_5d"])
                if model_suite == "full" and lgbm_params is not None
                else None
            ),
            report_preview={
                "ridge_mean_rank_ic": float(pd.Series([fold.rank_ic for fold in ridge_folds]).mean()) if ridge_folds else 0.0,
                "lightgbm_mean_rank_ic": float(pd.Series([fold.rank_ic for fold in lgbm_folds]).mean()) if lgbm_folds else 0.0,
                "coverage": coverage,
                "ridge_alpha": ridge_alpha,
                "model_suite": model_suite,
            },
        ),
    }
    emit_report(report=report, output_root=output_root, report_date=report_date)
    _write_training_log(output_root=output_root, run_ts=run_ts, report=report)
    return report


def main() -> int:
    """CLI entry point for the training workflow."""
    parser = argparse.ArgumentParser(description="Run the TradeML training workflow.")
    parser.add_argument("--data-root", default=".")
    parser.add_argument("--config", default="configs/equities_xs.yml")
    parser.add_argument("--output-root", default=".")
    parser.add_argument("--report-date", default=None)
    parser.add_argument("--model-suite", default="full", choices=["full", "ridge_only"])
    args = parser.parse_args()

    report = run_training(
        data_root=Path(args.data_root),
        config_path=Path(args.config),
        output_root=Path(args.output_root),
        report_date=args.report_date,
        model_suite=args.model_suite,
    )
    print(json.dumps(report, default=str))
    return 0

def _persist_run_artifacts(
    *,
    output_root: Path,
    run_ts: str,
    config: dict,
    feature_cols: list[str],
    normalized: pd.DataFrame,
    ridge_model: RidgeModel,
    lgbm_model: LightGBMModel | None,
    report_preview: dict,
) -> dict[str, str]:
    artifacts: dict[str, str] = {}
    models = {"ridge": ridge_model}
    if lgbm_model is not None:
        models["lightgbm"] = lgbm_model
    for model_name, model in models.items():
        run_dir = output_root / "models" / model_name / f"run_{run_ts}"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "feature_list.json").write_text(json.dumps(feature_cols, indent=2), encoding="utf-8")
        (run_dir / "config.json").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")
        (run_dir / "metrics.json").write_text(json.dumps(report_preview, indent=2, default=str), encoding="utf-8")
        (run_dir / "dataset_window.json").write_text(
            json.dumps(
                {
                    "min_date": str(pd.to_datetime(normalized["date"]).min().date()),
                    "max_date": str(pd.to_datetime(normalized["date"]).max().date()),
                    "rows": len(normalized),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        with (run_dir / "model.pkl").open("wb") as handle:
            pickle.dump(model, handle)
        artifacts[f"{model_name}_dir"] = str(run_dir)
    return artifacts


def _write_training_log(*, output_root: Path, run_ts: str, report: dict) -> Path:
    log_dir = output_root / "logs" / "training"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"run_{run_ts}.log"
    lines = [
        f"run_ts={run_ts}",
        f"coverage={report['coverage']:.4f}",
        f"window={report['window_start']}..{report['window_end']}",
        f"ridge_mean_rank_ic={report['ridge']['mean_rank_ic']:.4f}",
        f"lightgbm_mean_rank_ic={report.get('lightgbm', {}).get('mean_rank_ic', 0.0):.4f}",
        f"model_suite={report.get('model_suite', 'full')}",
        f"assessment={report['assessment']['decision']}",
        f"artifact_paths={json.dumps(report['artifacts'])}",
    ]
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return log_path


def _phase_one_assessment(*, ridge_mean_ic: float, ic_by_year_result: dict[int, float], cost_stress: dict, placebo: list[float]) -> dict[str, str]:
    years_positive = all(value > 0 for value in ic_by_year_result.values()) if ic_by_year_result else False
    placebo_ok = max((abs(value) for value in placebo), default=0.0) <= 0.10
    cost_ok = float(cost_stress.get("net_return", 0.0)) > 0
    ic_ok = ridge_mean_ic > 0.02
    decision = "GO" if ic_ok and years_positive and placebo_ok and cost_ok else "NO_GO"
    return {
        "decision": decision,
        "reason": f"ic_ok={ic_ok}; years_positive={years_positive}; placebo_ok={placebo_ok}; cost_ok={cost_ok}",
    }


if __name__ == "__main__":
    raise SystemExit(main())
