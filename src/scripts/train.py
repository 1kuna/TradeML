"""Training entry point."""

from __future__ import annotations

import argparse
import json
import pickle
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from trademl.data_node.training_control import read_pinned_phase_freeze
from trademl.features.equities import build_features
from trademl.features.preprocessing import rank_normalize
from trademl.labels.returns import build_labels
from trademl.modeling import DEFAULT_LABEL_VERSION, load_modeling_dataset, modeling_artifact_metadata
from trademl.models.lgbm import LightGBMModel, tune_lightgbm_via_walk_forward
from trademl.models.ridge import RidgeModel, tune_ridge_via_walk_forward
from trademl.portfolio.build import build_portfolio
from trademl.reports.emitter import emit_report
from trademl.research_architecture import build_candidate_autopsy
from trademl.validation.diagnostics import (
    feature_dependence_summary,
    fold_window_summary,
    ic_by_quarter,
    ic_by_year,
    model_comparison_summary,
    placebo_test,
    portfolio_cost_stress_test,
    sign_flip_canary,
)
from trademl.validation.metrics import bucket_metrics, rank_ic
from trademl.validation.cpcv import combinatorially_purged_cv
from trademl.validation.dsr import deflated_sharpe_ratio
from trademl.validation.negative_controls import compute_negative_control_diagnostics
from trademl.validation.pbo import probability_of_backtest_overfitting
from trademl.validation.walk_forward import expanding_walk_forward


MODEL_SUITES = {"full", "ridge_only", "advanced", "ensemble"}


def _load_catboost_components():
    """Load CatBoost components only when the advanced lane is selected."""

    from trademl.models.catboost import CatBoostModel, tune_catboost_via_walk_forward

    return CatBoostModel, tune_catboost_via_walk_forward


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


def _pinned_freeze_coverage(*, data_root: Path, effective_end_date: pd.Timestamp) -> float | None:
    pinned = read_pinned_phase_freeze(data_root=data_root, phase=1)
    if pinned is None:
        return None
    if str(pinned.get("date") or "") != effective_end_date.strftime("%Y-%m-%d"):
        return None
    explicit = pinned.get("effective_window_coverage_ratio")
    if explicit is None:
        explicit = pinned.get("window_coverage_ratio")
    if explicit is not None:
        return float(explicit)
    if bool(pinned.get("pinned")):
        return 1.0
    return float(pinned.get("coverage_ratio", 0.0) or 0.0)


def run_training(
    *,
    data_root: Path,
    config_path: Path,
    output_root: Path,
    report_date: str | None = None,
    model_suite: str = "full",
) -> dict:
    """Run the end-to-end training workflow and persist a report."""
    if model_suite not in MODEL_SUITES:
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
    pinned_freeze_coverage = _pinned_freeze_coverage(data_root=data_root, effective_end_date=effective_end_date)
    coverage = max(qc_coverage, planner_coverage or 0.0, pinned_freeze_coverage or 0.0)
    missing_dates = sorted(qc_window.loc[qc_window["status"] != "GREEN", "date"].dt.strftime("%Y-%m-%d").unique().tolist())
    if coverage < float(config["data"]["green_threshold"]):
        raise RuntimeError(f"green coverage below threshold: {coverage:.3f}; missing dates={missing_dates[:25]}")

    curated_files = sorted(path for path in curated_root.glob("date=*/data.parquet") if _partition_date(path) <= effective_end_date)
    if not curated_files:
        raise FileNotFoundError(f"no curated parquet files found under {curated_root} on or before {effective_end_date.strftime('%Y-%m-%d')}")
    panel, skipped_curated_partitions = _load_curated_panel(curated_files)

    modeling_config = dict(config.get("modeling") or {})
    label_horizon = int(modeling_config.get("label_horizon") or modeling_config.get("primary_label_horizon") or 5)
    label_col = f"label_{label_horizon}d"
    raw_label_col = f"raw_forward_return_{label_horizon}d"
    modeling_metadata: dict[str, Any] = {
        "enabled": bool(modeling_config.get("feature_store", {}).get("enabled", False)),
        "feature_set": modeling_config.get("feature_set", "legacy_in_memory"),
        "feature_version": modeling_config.get("feature_version") or "price_liquidity_v1",
        "label_version": modeling_config.get("label_version", DEFAULT_LABEL_VERSION),
        "label_horizon": label_horizon,
        "label_definition": modeling_config.get("label_definition", "universe_relative_forward_return"),
    }
    if modeling_metadata["enabled"]:
        merged, artifact_metadata = load_modeling_dataset(
            data_root=data_root,
            feature_version=str(modeling_metadata["feature_version"]),
            label_version=str(modeling_metadata["label_version"]),
            label_horizon=label_horizon,
            report_date=effective_end_date.strftime("%Y-%m-%d"),
        )
        modeling_metadata.update(artifact_metadata)
        panel = panel.loc[pd.to_datetime(panel["date"]) <= effective_end_date].copy()
    else:
        features = build_features(panel, config["features"])
        labels = build_labels(panel, horizon=label_horizon)
        merged = features.merge(labels, on=["date", "symbol"])
        modeling_metadata.update(modeling_artifact_metadata(data_root=data_root))
    merged = merged.dropna(subset=[label_col])
    feature_cols = [
        column
        for column in merged.columns
        if column
        not in {
            "date",
            "symbol",
            raw_label_col,
            label_col,
            "earnings_within_5d",
            "feature_available_at",
            "feature_set",
            "feature_version",
            "label_definition",
            "label_version",
            "data_revision_x",
            "data_revision_y",
            "data_revision",
            "universe_member",
        }
        and not str(column).startswith("target_date_")
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
        label_col,
        expanding_walk_forward,
        validation_config,
    )
    ridge_folds = expanding_walk_forward(normalized, feature_cols, label_col, lambda: RidgeModel(alpha=ridge_alpha), validation_config)
    lgbm_params: dict | None = None
    lgbm_folds = []
    if model_suite in {"full", "advanced", "ensemble"}:
        lgbm_params = tune_lightgbm_via_walk_forward(
            normalized,
            feature_cols,
            label_col,
            expanding_walk_forward,
            validation_config,
            n_trials=5,
        )
        lgbm_folds = expanding_walk_forward(
            normalized,
            feature_cols,
            label_col,
            lambda: LightGBMModel(n_trials=0, best_params=lgbm_params),
            validation_config,
        )
    catboost_params: dict | None = None
    catboost_folds = []
    if model_suite == "advanced":
        CatBoostModel, tune_catboost_via_walk_forward = _load_catboost_components()
        catboost_params = tune_catboost_via_walk_forward(
            normalized,
            feature_cols,
            label_col,
            expanding_walk_forward,
            validation_config,
            n_trials=5,
        )
        catboost_folds = expanding_walk_forward(
            normalized,
            feature_cols,
            label_col,
            lambda: CatBoostModel(n_trials=0, best_params=catboost_params),
            validation_config,
        )

    ridge_predictions = pd.concat([fold.predictions for fold in ridge_folds], ignore_index=True)
    lgbm_predictions = pd.concat([fold.predictions for fold in lgbm_folds], ignore_index=True) if lgbm_folds else pd.DataFrame()
    catboost_predictions = (
        pd.concat([fold.predictions for fold in catboost_folds], ignore_index=True) if catboost_folds else pd.DataFrame()
    )
    ensemble_predictions = (
        _ensemble_predictions({"ridge": ridge_predictions, "lightgbm": lgbm_predictions}, label_col=label_col) if model_suite == "ensemble" else pd.DataFrame()
    )
    primary_predictions = _primary_predictions_for_suite(
        model_suite=model_suite,
        ridge_predictions=ridge_predictions,
        lgbm_predictions=lgbm_predictions,
        catboost_predictions=catboost_predictions,
        ensemble_predictions=ensemble_predictions,
    )
    primary_report = _prediction_report(primary_predictions, label_col=label_col)
    cpcv_results = combinatorially_purged_cv(
        normalized.dropna(subset=[label_col]),
        feature_cols,
        label_col,
        lambda: RidgeModel(alpha=ridge_alpha),
        n_folds=int(config["validation"].get("cpcv_folds", 8)),
        embargo_days=int(config["validation"].get("embargo_days", 10)),
    )
    diagnostics = {
        "ic_by_year": ic_by_year(primary_predictions["prediction"], primary_predictions[label_col], primary_predictions["date"]),
        "ic_by_quarter": ic_by_quarter(primary_predictions["prediction"], primary_predictions[label_col], primary_predictions["date"]),
        "placebo": placebo_test(
            normalized.dropna(subset=[label_col]),
            feature_cols,
            label_col,
            lambda: RidgeModel(alpha=ridge_alpha),
            validation_runner=expanding_walk_forward,
            validation_config=validation_config,
        ),
        "cost_stress": portfolio_cost_stress_test(
            prices=panel[["date", "symbol", "close"]],
            prediction_frame=primary_predictions,
            multiplier=float(config["portfolio"]["cost_stress_multiplier"]),
            cost_spread_bps=float(config["portfolio"].get("cost_spread_bps", 5.0)),
        ),
        "sign_flip_canary": sign_flip_canary(primary_predictions, label_col=label_col),
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
        "fold_windows": fold_window_summary(_primary_folds_for_suite(model_suite=model_suite, ridge_folds=ridge_folds, lgbm_folds=lgbm_folds, catboost_folds=catboost_folds)),
    }
    diagnostics["feature_ablation"] = feature_dependence_summary(
        normalized.dropna(subset=[label_col]),
        feature_cols,
        label_col,
        primary_score=float(primary_report["mean_rank_ic"]),
    )
    diagnostics["negative_controls"] = compute_negative_control_diagnostics(
        predictions=primary_predictions,
        label_col=label_col,
        feature_frame=normalized.dropna(subset=[label_col]),
        feature_cols=feature_cols,
    )
    diagnostics["negative_controls"].update(
        {
            "max_single_feature_score_drop": diagnostics["feature_ablation"].get("max_single_feature_score_drop"),
            "min_feature_ablation_score_ratio": diagnostics["feature_ablation"].get("min_feature_ablation_score_ratio"),
        }
    )

    assessment = _phase_one_assessment(
        ridge_mean_ic=float(primary_report["mean_rank_ic"]),
        ic_by_year_result=diagnostics["ic_by_year"],
        cost_stress=diagnostics["cost_stress"],
        placebo=diagnostics["placebo"],
    )
    report = {
        "coverage": coverage,
        "qc_coverage": qc_coverage,
        "planner_window_coverage": planner_coverage,
        "pinned_freeze_coverage": pinned_freeze_coverage,
        "window_start": window_start.strftime("%Y-%m-%d"),
        "window_end": effective_end_date.strftime("%Y-%m-%d"),
        "missing_dates": missing_dates,
        "skipped_curated_partitions": skipped_curated_partitions,
        "ridge": {
            "alpha": ridge_alpha,
            **_fold_report(ridge_folds),
        },
        "lightgbm": (
            _fold_report(lgbm_folds)
            if model_suite in {"full", "advanced"}
            else {"skipped": True, "reason": "ridge_only_phase1_baseline"}
        ),
        "catboost": (
            {
                "params": catboost_params or {},
                **_fold_report(catboost_folds),
            }
            if model_suite == "advanced"
            else {"skipped": True, "reason": "advanced_lane_not_selected"}
        ),
        "ensemble": (
            primary_report
            if model_suite == "ensemble"
            else {"skipped": True, "reason": "ensemble_lane_not_selected"}
        ),
        "diagnostics": diagnostics,
        "assessment": assessment,
        "model_suite": model_suite,
        "modeling": modeling_metadata,
        "artifacts": _persist_run_artifacts(
            output_root=output_root,
            run_ts=run_ts,
            config=config,
            panel=panel,
            feature_cols=feature_cols,
            normalized=normalized,
            ridge_predictions=ridge_predictions,
            lgbm_predictions=lgbm_predictions,
            catboost_predictions=catboost_predictions,
            ensemble_predictions=ensemble_predictions,
            model_suite=model_suite,
            label_col=label_col,
            ridge_model=RidgeModel(alpha=ridge_alpha).fit(normalized[feature_cols].fillna(0.0), normalized[label_col]),
            lgbm_model=(
                LightGBMModel(n_trials=0, best_params=lgbm_params).fit(normalized[feature_cols].fillna(0.0), normalized[label_col])
                if model_suite in {"full", "advanced", "ensemble"} and lgbm_params is not None
                else None
            ),
            catboost_model=(
                _load_catboost_components()[0](n_trials=0, best_params=catboost_params).fit(
                    normalized[feature_cols].fillna(0.0),
                    normalized[label_col],
                )
                if model_suite == "advanced" and catboost_params is not None
                else None
            ),
            report_preview={
                "ridge_mean_rank_ic": _mean_rank_ic(ridge_folds),
                "lightgbm_mean_rank_ic": _mean_rank_ic(lgbm_folds),
                "catboost_mean_rank_ic": _mean_rank_ic(catboost_folds),
                "ensemble_mean_rank_ic": primary_report["mean_rank_ic"] if model_suite == "ensemble" else None,
                "coverage": coverage,
                "ridge_alpha": ridge_alpha,
                "model_suite": model_suite,
                "label_horizon": label_horizon,
                "label_col": label_col,
                "feature_version": modeling_metadata.get("feature_version"),
            },
        ),
    }
    diagnostics["model_comparison"] = model_comparison_summary(report)
    report["candidate_autopsy"] = build_candidate_autopsy(
        manifest={
            "run_id": "training_report",
            "model_suite": model_suite,
            "matrix_values": {"architecture_family": _architecture_family_for_model_suite(model_suite)},
        },
        report=report,
        gate_failures=[] if assessment["decision"] == "GO" else ["assessment.decision != GO"],
        gate={
            "max_abs_placebo_ic": 0.10,
            "min_cost_stress_net_return": 0.0,
            "strong_rejected_min_rank_ic": 0.05,
        },
    )
    emit_report(report=report, output_root=output_root, report_date=report_date)
    _write_training_log(output_root=output_root, run_ts=run_ts, report=report)
    return report


def _mean_rank_ic(folds: list[Any]) -> float:
    return float(pd.Series([fold.rank_ic for fold in folds]).mean()) if folds else 0.0


def _fold_report(folds: list[Any]) -> dict[str, object]:
    return {
        "folds": [
            {
                "rank_ic": fold.rank_ic,
                "decile_spread": fold.decile_spread,
                "hit_rate": fold.hit_rate,
                "bucket_returns": fold.bucket_returns,
            }
            for fold in folds
        ],
        "mean_rank_ic": _mean_rank_ic(folds),
        "decile_chart_data": {f"fold_{idx + 1}": fold.bucket_returns for idx, fold in enumerate(folds)},
    }


def _prediction_report(predictions: pd.DataFrame, *, label_col: str = "label_5d") -> dict[str, object]:
    """Return fold-report-shaped metrics for deterministic prediction frames."""
    if predictions.empty:
        return {"folds": [], "mean_rank_ic": 0.0, "decile_chart_data": {}}
    frame = predictions.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    folds: list[dict[str, object]] = []
    decile_chart_data: dict[str, dict[str, float]] = {}
    for idx, (date_value, group) in enumerate(frame.groupby("date", sort=True), start=1):
        decile_spread, hit_rate, bucket_returns = bucket_metrics(group, label_col=label_col)
        fold = {
            "rank_ic": rank_ic(group["prediction"], group[label_col]),
            "decile_spread": decile_spread,
            "hit_rate": hit_rate,
            "bucket_returns": bucket_returns,
        }
        folds.append(fold)
        decile_chart_data[f"fold_{idx}_{pd.Timestamp(date_value).date().isoformat()}"] = bucket_returns
    return {
        "folds": folds,
        "mean_rank_ic": float(pd.Series([fold["rank_ic"] for fold in folds], dtype=float).mean()) if folds else 0.0,
        "decile_chart_data": decile_chart_data,
    }


def _ensemble_predictions(prediction_frames: dict[str, pd.DataFrame], *, label_col: str = "label_5d") -> pd.DataFrame:
    """Build a deterministic per-date rank-averaged ensemble prediction frame."""
    merged: pd.DataFrame | None = None
    rank_columns: list[str] = []
    for name, predictions in prediction_frames.items():
        if predictions.empty:
            continue
        frame = predictions[["date", "symbol", "prediction", label_col]].copy()
        frame["date"] = pd.to_datetime(frame["date"])
        rank_col = f"{name}_rank_prediction"
        frame[rank_col] = frame.groupby("date")["prediction"].rank(method="average", pct=True)
        frame = frame[["date", "symbol", label_col, rank_col]]
        if merged is None:
            merged = frame
        else:
            merged = merged.merge(frame[["date", "symbol", rank_col]], on=["date", "symbol"], how="inner")
        rank_columns.append(rank_col)
    if merged is None or not rank_columns:
        return pd.DataFrame(columns=["date", "symbol", "prediction", label_col])
    merged["prediction"] = merged[rank_columns].mean(axis=1)
    return merged[["date", "symbol", "prediction", label_col]].sort_values(["date", "symbol"]).reset_index(drop=True)


def _primary_predictions_for_suite(
    *,
    model_suite: str,
    ridge_predictions: pd.DataFrame,
    lgbm_predictions: pd.DataFrame,
    catboost_predictions: pd.DataFrame,
    ensemble_predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Return the prediction frame that the selected suite optimizes."""
    if model_suite == "advanced" and not catboost_predictions.empty:
        return catboost_predictions
    if model_suite == "full" and not lgbm_predictions.empty:
        return lgbm_predictions
    if model_suite == "ensemble" and not ensemble_predictions.empty:
        return ensemble_predictions
    return ridge_predictions


def _primary_folds_for_suite(*, model_suite: str, ridge_folds: list[Any], lgbm_folds: list[Any], catboost_folds: list[Any]) -> list[Any]:
    """Return the fold set used by the suite's primary score."""
    if model_suite == "advanced" and catboost_folds:
        return catboost_folds
    if model_suite == "full" and lgbm_folds:
        return lgbm_folds
    return ridge_folds


def _architecture_family_for_model_suite(model_suite: str) -> str:
    """Return the registry family for a model suite."""
    if model_suite == "advanced":
        return "advanced_challenger"
    if model_suite == "ensemble":
        return "ensemble_meta"
    if model_suite == "full":
        return "tree_challenger"
    return "linear_baseline"


def main() -> int:
    """CLI entry point for the training workflow."""
    parser = argparse.ArgumentParser(description="Run the TradeML training workflow.")
    parser.add_argument("--data-root", default=".")
    parser.add_argument("--config", default="configs/equities_xs.yml")
    parser.add_argument("--output-root", default=".")
    parser.add_argument("--report-date", default=None)
    parser.add_argument("--model-suite", default="full", choices=sorted(MODEL_SUITES))
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
    panel: pd.DataFrame,
    feature_cols: list[str],
    normalized: pd.DataFrame,
    ridge_predictions: pd.DataFrame,
    lgbm_predictions: pd.DataFrame,
    catboost_predictions: pd.DataFrame,
    ensemble_predictions: pd.DataFrame,
    model_suite: str,
    label_col: str,
    ridge_model: RidgeModel,
    lgbm_model: LightGBMModel | None,
    catboost_model: Any | None,
    report_preview: dict,
) -> dict[str, str]:
    artifacts: dict[str, str] = {}
    artifacts.update(
        _persist_backtest_inputs(
            output_root=output_root,
            panel=panel,
            ridge_predictions=ridge_predictions,
            lgbm_predictions=lgbm_predictions,
            catboost_predictions=catboost_predictions,
            ensemble_predictions=ensemble_predictions,
            model_suite=model_suite,
            label_col=label_col,
            portfolio_config=dict(config.get("portfolio") or {}),
        )
    )
    models = {"ridge": ridge_model}
    if lgbm_model is not None:
        models["lightgbm"] = lgbm_model
    if catboost_model is not None:
        models["catboost"] = catboost_model
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


def _persist_backtest_inputs(
    *,
    output_root: Path,
    panel: pd.DataFrame,
    ridge_predictions: pd.DataFrame,
    lgbm_predictions: pd.DataFrame,
    catboost_predictions: pd.DataFrame,
    ensemble_predictions: pd.DataFrame,
    model_suite: str,
    label_col: str,
    portfolio_config: dict[str, Any],
) -> dict[str, str]:
    inputs_root = output_root / "artifacts" / "backtest_inputs"
    inputs_root.mkdir(parents=True, exist_ok=True)
    prices = (
        panel[["date", "symbol", "close"]]
        .copy()
        .drop_duplicates(subset=["date", "symbol"], keep="last")
        .sort_values(["date", "symbol"])
        .reset_index(drop=True)
    )
    prices_path = inputs_root / "prices.parquet"
    prices.to_parquet(prices_path, index=False)

    artifacts: dict[str, str] = {"prices_path": str(prices_path)}
    ridge_payload = _write_prediction_artifacts(inputs_root=inputs_root, prefix="ridge", predictions=ridge_predictions, label_col=label_col, portfolio_config=portfolio_config)
    artifacts.update(ridge_payload)
    primary_prefix = "ridge"
    if not lgbm_predictions.empty:
        lgbm_payload = _write_prediction_artifacts(inputs_root=inputs_root, prefix="lightgbm", predictions=lgbm_predictions, label_col=label_col, portfolio_config=portfolio_config)
        artifacts.update(lgbm_payload)
        if model_suite == "full":
            primary_prefix = "lightgbm"
    if not catboost_predictions.empty:
        catboost_payload = _write_prediction_artifacts(inputs_root=inputs_root, prefix="catboost", predictions=catboost_predictions, label_col=label_col, portfolio_config=portfolio_config)
        artifacts.update(catboost_payload)
        if model_suite == "advanced":
            primary_prefix = "catboost"
    if not ensemble_predictions.empty:
        ensemble_payload = _write_prediction_artifacts(inputs_root=inputs_root, prefix="ensemble", predictions=ensemble_predictions, label_col=label_col, portfolio_config=portfolio_config)
        artifacts.update(ensemble_payload)
        if model_suite == "ensemble":
            primary_prefix = "ensemble"
    artifacts["primary_predictions_path"] = artifacts[f"{primary_prefix}_predictions_path"]
    artifacts["primary_targets_path"] = artifacts[f"{primary_prefix}_targets_path"]
    return artifacts


def _write_prediction_artifacts(*, inputs_root: Path, prefix: str, predictions: pd.DataFrame, label_col: str, portfolio_config: dict[str, Any]) -> dict[str, str]:
    if predictions.empty:
        return {}
    prediction_frame = predictions.copy()
    if label_col in prediction_frame.columns and "label" not in prediction_frame.columns:
        prediction_frame["label"] = prediction_frame[label_col]
    prediction_frame = prediction_frame.sort_values(["date", "symbol"]).reset_index(drop=True)
    prediction_path = inputs_root / f"{prefix}_predictions.parquet"
    prediction_frame.to_parquet(prediction_path, index=False)

    targets_input = prediction_frame.rename(columns={"prediction": "score"})
    targets = build_portfolio(targets_input, portfolio_config)
    target_path = inputs_root / f"{prefix}_targets.parquet"
    targets.to_parquet(target_path, index=False)
    return {
        f"{prefix}_predictions_path": str(prediction_path),
        f"{prefix}_targets_path": str(target_path),
    }


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
        f"catboost_mean_rank_ic={report.get('catboost', {}).get('mean_rank_ic', 0.0):.4f}",
        f"ensemble_mean_rank_ic={report.get('ensemble', {}).get('mean_rank_ic', 0.0):.4f}",
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
