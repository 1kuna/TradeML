"""
Equities cross-sectional pipeline (long-term name).

End-to-end steps:
1) Load curated prices for universe and date range
2) Build features and labels
3) Run CPCV and fit LightGBM baseline (fallbacks to ridge/logit if unavailable)
4) Collect OOS predictions as scores
5) Build portfolio target weights from scores
6) Convert to target quantities using price and run backtest with costs
7) Report metrics incl. Sharpe, DSR; compute PBO over a small config grid
8) Emit daily positions JSON/MD for the last as-of date and persist artifacts
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Literal, Optional

import numpy as np
import pandas as pd
import yaml
from loguru import logger

try:
    import joblib
except Exception:  # pragma: no cover
    joblib = None  # type: ignore

from backtest.engine.backtester import MinimalBacktester
from data_layer.curated.loaders import load_price_panel
from feature_store.equities.dataset import build_training_dataset
from models.equities_xs import (
    LGBMParams,
    export_lgbm_onnx,
    fit_lgbm,
    predict_lgbm,
    save_lgbm_pickle,
    train_logistic_regression,
    train_ridge_regression,
)
from portfolio.build import build as build_portfolio
from ops.reports.emitter import emit_daily
from validation import CPCV, DSRCalculator, PBOCalculator
from validation.calibration import binary_calibration


def _to_date(d: object) -> date:
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, date):
        return d
    if isinstance(d, str):
        return datetime.strptime(d, "%Y-%m-%d").date()
    raise TypeError(f"Unsupported date type: {type(d)}")


def _load_training_cfg(path: str = "configs/training/equities_xs.yml") -> dict:
    try:
        with open(path) as f:
            return yaml.safe_load(f)
    except Exception:
        return {}


def _months_between(d0: date, d1: date) -> float:
    return (d1 - d0).days / 30.4375


def _time_decay_weights(dates: pd.Series, end: date, half_life_months: float) -> np.ndarray:
    if half_life_months <= 0:
        return np.ones(len(dates), dtype=float)
    ages = dates.apply(lambda d: _months_between(_to_date(d), end)).astype(float)
    w = 0.5 ** (ages / half_life_months)
    # Avoid zeros
    w = np.clip(w, 1e-6, None)
    return w.values


def _parse_regime_name(name: str) -> tuple[Optional[date], Optional[date]]:
    name = name.strip()
    if name.lower().startswith("pre-"):
        yr = int(name.split("-", 1)[-1])
        return None, date(yr, 1, 1)
    if name.endswith("+"):
        yr = int(name[:-1])
        return date(yr, 1, 1), None
    # range YYYY-YYYY
    try:
        a, b = name.split("-")
        return date(int(a), 1, 1), date(int(b), 12, 31)
    except Exception:
        return None, None


def _regime_allow_mask(dates: pd.Series, regime_cfg: list[dict]) -> np.ndarray:
    allowed_windows = []
    for r in regime_cfg or []:
        use_for = str(r.get("use_for", "")).lower()
        if use_for not in ("fit_and_validate", "fit_only"):
            continue
        start, end = _parse_regime_name(str(r.get("name", "")).strip())
        allowed_windows.append((start, end))
    if not allowed_windows:
        return np.ones(len(dates), dtype=bool)
    ds = dates.apply(_to_date)
    mask = np.zeros(len(ds), dtype=bool)
    for start, end in allowed_windows:
        if start is None and end is not None:
            mask |= ds <= end
        elif start is not None and end is None:
            mask |= ds >= start
        elif start is not None and end is not None:
            mask |= (ds >= start) & (ds <= end)
    return mask


def _json_default(obj):
    if isinstance(obj, (np.generic,)):
        return obj.item()
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _create_artifact_dir(end_date: date) -> Path:
    base = Path("models") / "equities_xs" / "artifacts"
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = base / f"{end_date.isoformat()}_{ts}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _persist_artifacts(
    artifact_dir: Path,
    cfg: PipelineConfig,
    training_cfg: dict,
    feature_cols: List[str],
    oof_scores: pd.DataFrame,
    fold_metrics: List[Dict],
    final_metrics: Dict[str, float],
    model_backend: str,
    onnx_exported: bool,
    model_path: Optional[Path] = None,
    onnx_path: Optional[Path] = None,
) -> Dict[str, str]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "feature_list.json").write_text(json.dumps(feature_cols, indent=2))
    summary = {
        "pipeline_config": asdict(cfg),
        "training_cfg": training_cfg,
        "fold_metrics": fold_metrics,
        "final_metrics": final_metrics,
        "model_backend": model_backend,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "onnx_exported": onnx_exported,
        "model_path": str(model_path) if model_path else None,
        "onnx_path": str(onnx_path) if onnx_path else None,
    }
    (artifact_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2, default=_json_default)
    )
    try:
        oof_scores.to_parquet(artifact_dir / "oof_scores.parquet", index=False)
    except Exception as exc:
        logger.warning(f"Failed to persist OOF scores to parquet: {exc}")
        oof_scores.to_csv(artifact_dir / "oof_scores.csv", index=False)

    # Maintain a 'latest' pointer for quick loading
    latest_pointer = artifact_dir.parent / "latest"
    try:
        if latest_pointer.exists() or latest_pointer.is_symlink():
            latest_pointer.unlink()
        latest_pointer.symlink_to(artifact_dir, target_is_directory=True)
    except Exception:
        # Symlinks may not be available (e.g., Windows without admin); fallback to text pointer
        (artifact_dir.parent / "latest.txt").write_text(str(artifact_dir))

    return {
        "feature_list": str(artifact_dir / "feature_list.json"),
        "training_summary": str(artifact_dir / "training_summary.json"),
        "oof_scores": str(artifact_dir / "oof_scores.parquet"),
    }


@dataclass
class PipelineConfig:
    start_date: str
    end_date: str
    universe: List[str]
    label_type: Literal["horizon", "triple_barrier"] = "horizon"
    horizon_days: int = 5
    tp_sigma: float = 2.0
    sl_sigma: float = 1.0
    max_h: int = 10
    vol_window: int = 20
    n_folds: int = 8
    embargo_days: int = 10
    initial_capital: float = 1_000_000.0
    spread_bps: float = 5.0
    # Portfolio
    gross_cap: float = 1.0
    max_name: float = 0.05
    kelly_fraction: float = 1.0


def run_pipeline(cfg: PipelineConfig) -> Dict:
    logger.info("Building training dataset...")
    ds = build_training_dataset(
        universe=cfg.universe,
        start_date=cfg.start_date,
        end_date=cfg.end_date,
        label_type=cfg.label_type,
        horizon_days=cfg.horizon_days,
        tp_sigma=cfg.tp_sigma,
        sl_sigma=cfg.sl_sigma,
        max_h=cfg.max_h,
        vol_window=cfg.vol_window,
    )
    if ds.X.empty:
        raise RuntimeError("Empty dataset produced — ensure curated data is available for the universe/date range.")

    # Prepare CPCV label metadata (date + horizon + symbol for symbol-aware purging)
    groups = pd.DataFrame({
        "date": ds.X["date"],
        "horizon_days": ds.meta["horizon_days"],
        "symbol": ds.X["symbol"],
    }).reset_index(drop=True)

    # Choose baseline
    clf_task = (cfg.label_type == "triple_barrier")

    # Sort by date to ensure time ordering and align groups
    sort_idx = ds.X["date"].argsort()
    X_sorted = ds.X.iloc[sort_idx].reset_index(drop=True)
    y_sorted = ds.y.iloc[sort_idx].reset_index(drop=True)
    groups_sorted = groups.iloc[sort_idx].reset_index(drop=True)

    # CPCV with purging and embargo per SSOT/Blueprint
    cv = CPCV(n_folds=cfg.n_folds, embargo_days=cfg.embargo_days)
    splits = cv.split(X_sorted, groups_sorted)

    # Collect OOS predictions
    oos_rows: List[Dict] = []
    fold_metrics: List[Dict[str, float]] = []
    # Load training config for regime/time-decay
    tcfg = _load_training_cfg()
    hl_months = float(((tcfg.get("time_decay") or {}).get("half_life_months", 0)))
    regime_cfg = (tcfg.get("regime_masks") or [])

    end_date = _to_date(cfg.end_date)
    all_weights = (
        _time_decay_weights(X_sorted["date"], end=end_date, half_life_months=hl_months)
        if hl_months > 0
        else np.ones(len(X_sorted))
    )
    allow_mask = _regime_allow_mask(X_sorted["date"], regime_cfg)
    feature_cols = [c for c in X_sorted.columns if c not in ("date", "symbol")]
    X_features = X_sorted[feature_cols]

    for fold_id, (tr, te) in enumerate(splits):
        meta_te = X_sorted.iloc[te][["date", "symbol"]].reset_index(drop=True)
        mask_idx = np.where(allow_mask[tr])[0]
        if mask_idx.size == 0:
            logger.warning(f"Fold {fold_id} has no samples after regime filtering; skipping.")
            continue

        X_tr_fold = X_features.iloc[tr].iloc[mask_idx]
        y_tr_fold = y_sorted.iloc[tr].iloc[mask_idx]
        sample_w = all_weights[tr][mask_idx] if all_weights is not None else None
        X_te_fold = X_features.iloc[te]

        backend = "lightgbm"
        try:
            model, met = fit_lgbm(X_tr_fold, y_tr_fold, sample_weight=sample_w)
            score = predict_lgbm(model, X_te_fold)
        except ImportError as exc:
            backend = "sklearn"
            logger.warning(f"LightGBM unavailable in fold {fold_id}; falling back to baseline: {exc}")
            if clf_task:
                model, met = train_logistic_regression(X_tr_fold, y_tr_fold, sample_weight=sample_w)
                score = model.predict_proba(X_te_fold)[:, 1]
            else:
                model, met = train_ridge_regression(X_tr_fold, y_tr_fold, sample_weight=sample_w)
                score = model.predict(X_te_fold)
        except Exception as exc:
            backend = "sklearn"
            logger.exception(f"Fold {fold_id} LightGBM failed; reverting to baseline models: {exc}")
            if clf_task:
                model, met = train_logistic_regression(X_tr_fold, y_tr_fold, sample_weight=sample_w)
                score = model.predict_proba(X_te_fold)[:, 1]
            else:
                model, met = train_ridge_regression(X_tr_fold, y_tr_fold, sample_weight=sample_w)
                score = model.predict(X_te_fold)

        met = met or {}
        met["backend"] = backend
        met["fold"] = fold_id
        fold_metrics.append(
            {
                k: float(v)
                if isinstance(v, (int, float, np.floating))
                else v
                for k, v in met.items()
            }
        )
        fold_scores = pd.DataFrame({"date": meta_te["date"], "symbol": meta_te["symbol"], "score": score})
        fold_scores["fold"] = fold_id
        oos_rows.append(fold_scores)

    if not oos_rows:
        raise RuntimeError("No OOF scores produced; check regime masks and CPCV splits.")
    oos_scores = pd.concat(oos_rows, ignore_index=True)

    artifact_dir = _create_artifact_dir(end_date)
    fit_idx = np.where(allow_mask)[0]
    if fit_idx.size == 0:
        logger.warning("Regime masks removed all samples; training on full dataset.")
        fit_idx = np.arange(len(X_sorted))
    X_fit = X_features.iloc[fit_idx]
    y_fit = y_sorted.iloc[fit_idx]
    weight_fit = all_weights[fit_idx] if all_weights is not None else None

    model_backend = "lightgbm"
    final_metrics: Dict[str, float] = {}
    final_model_path: Optional[Path] = None
    onnx_path: Optional[Path] = None
    onnx_exported = False

    try:
        final_model, final_metrics = fit_lgbm(X_fit, y_fit, sample_weight=weight_fit)
        final_model_path = artifact_dir / "model.pkl"
        save_lgbm_pickle(final_model, final_model_path)
        try:
            onnx_path = artifact_dir / "model.onnx"
            export_lgbm_onnx(final_model, feature_cols, onnx_path)
            onnx_exported = True
        except Exception as exc:  # pragma: no cover - optional dependency
            onnx_exported = False
            onnx_path = None
            logger.warning(f"ONNX export skipped: {exc}")
    except ImportError as exc:
        model_backend = "sklearn"
        logger.warning(f"LightGBM unavailable for final model; falling back to ridge/logit: {exc}")
        if clf_task:
            final_model, final_metrics = train_logistic_regression(X_fit, y_fit, sample_weight=weight_fit)
        else:
            final_model, final_metrics = train_ridge_regression(X_fit, y_fit, sample_weight=weight_fit)
        if joblib is None:
            raise ImportError("joblib is required to persist fallback models.") from exc
        final_model_path = artifact_dir / "model.pkl"
        joblib.dump(final_model, final_model_path)
    except Exception as exc:
        model_backend = "error"
        final_metrics = {"status": "failed", "error": str(exc)}
        logger.exception(f"Final model training failed: {exc}")

    artifact_files = _persist_artifacts(
        artifact_dir=artifact_dir,
        cfg=cfg,
        training_cfg=tcfg,
        feature_cols=feature_cols,
        oof_scores=oos_scores,
        fold_metrics=fold_metrics,
        final_metrics=final_metrics,
        model_backend=model_backend,
        onnx_exported=onnx_exported,
        model_path=final_model_path,
        onnx_path=onnx_path,
    )

    # Portfolio construction
    risk_cfg = {
        "gross_cap": cfg.gross_cap,
        "max_name": cfg.max_name,
        "kelly_fraction": cfg.kelly_fraction,
    }
    port = build_portfolio(oos_scores, risk_cfg)
    tw = port["target_weights"]

    # Convert weights to target quantities using close prices; include all dates with zero weight when no signal
    price_panel = load_price_panel(cfg.universe, cfg.start_date, cfg.end_date)
    px = price_panel[["date", "symbol", "close"]]
    signals = px.merge(tw, on=["date", "symbol"], how="left")
    signals["target_w"] = signals["target_w"].fillna(0.0)
    signals["target_quantity"] = (signals["target_w"] * cfg.initial_capital) / signals["close"]

    # Backtest
    bt = MinimalBacktester(initial_capital=cfg.initial_capital, spread_bps=cfg.spread_bps)
    equity_curve = bt.run(signals[["date", "symbol", "target_quantity"]], px)
    metrics_bt = bt.calculate_performance()

    # DSR on daily returns
    ret = equity_curve["equity"].pct_change().dropna().values
    dsr_calc = DSRCalculator(annual_factor=252.0)
    dsr_res = dsr_calc.calculate_dsr(ret, n_trials=max(1, cfg.n_folds))

    # PBO with a tiny config grid (example): vary ridge alpha or logit C
    # Construct IS/OOS metric matrices over splits
    from sklearn.model_selection import ParameterGrid
    if clf_task:
        param_grid = list(ParameterGrid({"C": [0.1, 1.0, 10.0]}))
        # Use accuracy as metric for simplicity
        from sklearn.metrics import accuracy_score
        n_configs = len(param_grid)
        n_trials = len(splits)
        is_perf = np.zeros((n_configs, n_trials))
        oos_perf = np.zeros((n_configs, n_trials))
        for ci, params in enumerate(param_grid):
            for ti, (tr, te) in enumerate(splits):
                X_tr = X_sorted.drop(columns=["date", "symbol"]).iloc[tr]
                y_tr = y_sorted.iloc[tr]
                X_te = X_sorted.drop(columns=["date", "symbol"]).iloc[te]
                y_te = y_sorted.iloc[te]
                # Split train further for IS via a simple holdout: last 10% of tr as val
                cut = int(0.9 * len(tr))
                tr_idx = tr[:cut]
                val_idx = tr[cut:]
                Xt, yt = X_sorted.drop(columns=["date", "symbol"]).iloc[tr_idx], y_sorted.iloc[tr_idx]
                Xv, yv = X_sorted.drop(columns=["date", "symbol"]).iloc[val_idx], y_sorted.iloc[val_idx]
                from sklearn.linear_model import LogisticRegression
                from sklearn.pipeline import Pipeline
                from sklearn.preprocessing import StandardScaler
                model = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=200, class_weight="balanced", random_state=42, **params))])
                model.fit(Xt, yt)
                yv_pred = model.predict(Xv)
                yte_pred = model.predict(X_te)
                is_perf[ci, ti] = accuracy_score(yv, yv_pred)
                oos_perf[ci, ti] = accuracy_score(y_te, yte_pred)
    else:
        param_grid = list(ParameterGrid({"alpha": [0.1, 1.0, 10.0]}))
        from sklearn.metrics import r2_score
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        n_configs = len(param_grid)
        n_trials = len(splits)
        is_perf = np.zeros((n_configs, n_trials))
        oos_perf = np.zeros((n_configs, n_trials))
        for ci, params in enumerate(param_grid):
            for ti, (tr, te) in enumerate(splits):
                X_tr = X_sorted.drop(columns=["date", "symbol"]).iloc[tr]
                y_tr = y_sorted.iloc[tr]
                X_te = X_sorted.drop(columns=["date", "symbol"]).iloc[te]
                y_te = y_sorted.iloc[te]
                cut = int(0.9 * len(tr))
                tr_idx = tr[:cut]
                val_idx = tr[cut:]
                Xt, yt = X_sorted.drop(columns=["date", "symbol"]).iloc[tr_idx], y_sorted.iloc[tr_idx]
                Xv, yv = X_sorted.drop(columns=["date", "symbol"]).iloc[val_idx], y_sorted.iloc[val_idx]
                model = Pipeline([("scaler", StandardScaler()), ("reg", Ridge(random_state=42, **params))])
                model.fit(Xt, yt)
                yv_pred = model.predict(Xv)
                yte_pred = model.predict(X_te)
                is_perf[ci, ti] = r2_score(yv, yv_pred) if len(np.unique(yv)) > 1 else 0.0
                oos_perf[ci, ti] = r2_score(y_te, yte_pred) if len(np.unique(y_te)) > 1 else 0.0

    pbo_calc = PBOCalculator(n_trials=oos_perf.shape[1])
    pbo_res = pbo_calc.calculate_pbo(is_perf, oos_perf)

    # Optional calibration metrics for classifiers
    calib = None
    if clf_task:
        # Merge OOS scores with true labels
        merged = oos_scores.merge(ds.y.rename("y_true").to_frame().reset_index(drop=True), left_index=True, right_index=True)
        # Align by date+symbol to avoid index ambiguity
        merged = merged.join(ds.X[["date", "symbol"]].reset_index(drop=True))
        # Sort and drop NaNs
        m = merged.dropna(subset=["score", "y_true"]).copy()
        try:
            res = binary_calibration(m["y_true"].values.astype(int), m["score"].values.astype(float), n_bins=10)
            calib = {"brier": round(res.brier, 6), "logloss": round(res.logloss, 6)}
        except Exception:
            calib = None

    # Emit daily positions for the last date
    last_date = tw["date"].max()
    last_positions = tw[tw["date"] == last_date][["symbol", "target_w"]]
    metrics_report = {
        "total_return": round(metrics_bt.total_return, 6),
        "sharpe": round(metrics_bt.sharpe_ratio, 4),
        "max_drawdown": round(metrics_bt.max_drawdown, 6),
        "turnover": round(metrics_bt.turnover, 4),
        "dsr": round(dsr_res["dsr"], 4),
        "pbo": round(pbo_res["pbo"], 4),
    }
    if calib is not None:
        metrics_report.update({"brier": calib.get("brier"), "logloss": calib.get("logloss")})
    metrics_report["model_backend"] = model_backend
    metrics_report["onnx_exported"] = onnx_exported
    for mk, mv in (final_metrics or {}).items():
        if isinstance(mv, (int, float, np.floating)):
            metrics_report[f"model_{mk}"] = round(float(mv), 6)
    emit_daily(last_date, last_positions, metrics_report)

    return {
        "equity_curve": equity_curve,
        "backtest_metrics": metrics_bt,
        "dsr": dsr_res,
        "pbo": pbo_res,
        "target_weights": tw,
        "signals": signals[["date", "symbol", "target_quantity"]],
        "calibration": calib,
        "artifacts": {
            "dir": str(artifact_dir),
            **artifact_files,
            "model_path": str(final_model_path) if final_model_path else None,
            "onnx_path": str(onnx_path) if onnx_path else None,
            "backend": model_backend,
        },
        "fold_metrics": fold_metrics,
        "final_model_metrics": final_metrics,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run equities cross-sectional pipeline")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--symbols", nargs="+", required=True)
    parser.add_argument("--label", choices=["horizon", "triple_barrier"], default="horizon")
    parser.add_argument("--k", type=int, default=5, help="horizon days for horizon labels")
    parser.add_argument("--tp", type=float, default=2.0)
    parser.add_argument("--sl", type=float, default=1.0)
    parser.add_argument("--max_h", type=int, default=10)
    parser.add_argument("--folds", type=int, default=8)
    parser.add_argument("--embargo", type=int, default=10)
    parser.add_argument("--capital", type=float, default=1_000_000.0)
    parser.add_argument("--spread_bps", type=float, default=5.0)
    parser.add_argument("--gross_cap", type=float, default=1.0)
    parser.add_argument("--max_name", type=float, default=0.05)
    parser.add_argument("--kelly", type=float, default=1.0)

    args = parser.parse_args()

    cfg = PipelineConfig(
        start_date=args.start,
        end_date=args.end,
        universe=args.symbols,
        label_type=args.label,
        horizon_days=args.k,
        tp_sigma=args.tp,
        sl_sigma=args.sl,
        max_h=args.max_h,
        n_folds=args.folds,
        embargo_days=args.embargo,
        initial_capital=args.capital,
        spread_bps=args.spread_bps,
        gross_cap=args.gross_cap,
        max_name=args.max_name,
        kelly_fraction=args.kelly,
    )

    out = run_pipeline(cfg)
    print("\n[OK] Phase 2 pipeline completed.")
    print(f"Equity curve points: {len(out['equity_curve'])}")
    print(f"Sharpe: {out['backtest_metrics'].sharpe_ratio:.2f}")
    print(f"PBO: {out['pbo']['pbo']:.2%}")
    print(f"DSR: {out['dsr']['dsr']:.4f}")
