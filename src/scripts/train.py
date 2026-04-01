"""Training entry point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from trademl.features.equities import build_features
from trademl.features.preprocessing import rank_normalize
from trademl.labels.returns import build_labels
from trademl.models.lgbm import LightGBMModel
from trademl.models.ridge import RidgeModel
from trademl.reports.emitter import emit_report
from trademl.validation.diagnostics import cost_stress_test, ic_by_year, placebo_test
from trademl.validation.walk_forward import expanding_walk_forward


def run_training(*, data_root: Path, config_path: Path, output_root: Path, report_date: str | None = None) -> dict:
    """Run the end-to-end training workflow and persist a report."""
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    qc_path = data_root / "data" / "qc" / "partition_status.parquet"
    curated_root = data_root / "data" / "curated" / "equities_ohlcv_adj"
    if not qc_path.exists():
        raise FileNotFoundError(f"missing partition status parquet: {qc_path}")

    qc = pd.read_parquet(qc_path)
    coverage = float((qc["status"] == "GREEN").mean()) if not qc.empty else 0.0
    if coverage < float(config["data"]["green_threshold"]):
        raise RuntimeError(f"green coverage below threshold: {coverage:.3f}")

    curated_files = sorted(curated_root.glob("date=*/data.parquet"))
    if not curated_files:
        raise FileNotFoundError(f"no curated parquet files found under {curated_root}")
    panel = pd.concat((pd.read_parquet(path) for path in curated_files), ignore_index=True)

    features = build_features(panel, config["features"])
    labels = build_labels(panel, horizon=5)
    merged = features.merge(labels, on=["date", "symbol"]).dropna()
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
    ridge_folds = expanding_walk_forward(normalized, feature_cols, "label_5d", lambda: RidgeModel(alpha=1.0), validation_config)
    lgbm_folds = expanding_walk_forward(
        normalized,
        feature_cols,
        "label_5d",
        lambda: LightGBMModel(n_trials=5),
        validation_config,
    )

    ridge_predictions = pd.concat([fold.predictions for fold in ridge_folds], ignore_index=True)
    ridge_predictions["gross_return"] = ridge_predictions["label_5d"]
    ridge_predictions["cost"] = 0.0005
    diagnostics = {
        "ic_by_year": ic_by_year(ridge_predictions["prediction"], ridge_predictions["label_5d"], ridge_predictions["date"]),
        "placebo": placebo_test(normalized[feature_cols + ["label_5d"]].dropna(), feature_cols, "label_5d", lambda: RidgeModel(alpha=1.0)),
        "cost_stress": cost_stress_test(ridge_predictions[["gross_return", "cost"]], multiplier=float(config["portfolio"]["cost_stress_multiplier"])),
    }

    report = {
        "coverage": coverage,
        "ridge": {
            "folds": [
                {"rank_ic": fold.rank_ic, "decile_spread": fold.decile_spread, "hit_rate": fold.hit_rate}
                for fold in ridge_folds
            ],
            "mean_rank_ic": float(pd.Series([fold.rank_ic for fold in ridge_folds]).mean()) if ridge_folds else 0.0,
        },
        "lightgbm": {
            "folds": [
                {"rank_ic": fold.rank_ic, "decile_spread": fold.decile_spread, "hit_rate": fold.hit_rate}
                for fold in lgbm_folds
            ],
            "mean_rank_ic": float(pd.Series([fold.rank_ic for fold in lgbm_folds]).mean()) if lgbm_folds else 0.0,
        },
        "diagnostics": diagnostics,
    }
    emit_report(report=report, output_root=output_root, report_date=report_date)
    return report


def main() -> int:
    """CLI entry point for the training workflow."""
    parser = argparse.ArgumentParser(description="Run the TradeML training workflow.")
    parser.add_argument("--data-root", default=".")
    parser.add_argument("--config", default="configs/equities_xs.yml")
    parser.add_argument("--output-root", default=".")
    parser.add_argument("--report-date", default=None)
    args = parser.parse_args()

    report = run_training(
        data_root=Path(args.data_root),
        config_path=Path(args.config),
        output_root=Path(args.output_root),
        report_date=args.report_date,
    )
    print(json.dumps(report, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
