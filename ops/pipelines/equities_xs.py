"""
Equities cross-sectional pipeline (long-term name).

End-to-end steps:
1) Load curated prices for universe and date range
2) Build features and labels
3) Run CPCV and fit baseline model (logistic for triple-barrier; ridge for horizon)
4) Collect OOS predictions as scores
5) Build portfolio target weights from scores
6) Convert to target quantities using price and run backtest with costs
7) Report metrics incl. Sharpe, DSR; compute PBO over a small config grid
8) Emit daily positions JSON/MD for the last as-of date
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional
from datetime import date, datetime

import numpy as np
import pandas as pd
from loguru import logger
import yaml

from data_layer.curated.loaders import load_price_panel
from feature_store.equities.dataset import build_training_dataset
from models.equities_xs.baselines import train_logistic_regression, train_ridge_regression
from portfolio.build import build as build_portfolio
from backtest.engine.backtester import MinimalBacktester
from validation import CPCV, DSRCalculator, PBOCalculator
from validation.calibration import binary_calibration
from ops.reports.emitter import emit_daily


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
    # Load training config for regime/time-decay
    tcfg = _load_training_cfg()
    hl_months = float(((tcfg.get("time_decay") or {}).get("half_life_months", 0)))
    regime_cfg = (tcfg.get("regime_masks") or [])

    end_date = _to_date(cfg.end_date)
    all_weights = _time_decay_weights(X_sorted["date"], end=end_date, half_life_months=hl_months) if hl_months > 0 else np.ones(len(X_sorted))
    allow_mask = _regime_allow_mask(X_sorted["date"], regime_cfg)

    for fold_id, (tr, te) in enumerate(splits):
        X_tr = X_sorted.drop(columns=["date", "symbol"]).iloc[tr]
        y_tr = y_sorted.iloc[tr]
        X_te = X_sorted.drop(columns=["date", "symbol"]).iloc[te]
        meta_te = X_sorted.iloc[te][["date", "symbol"]].reset_index(drop=True)

        # Apply regime filter and time-decay weights on train set
        tr_mask = allow_mask[tr]
        X_tr_f = X_tr.iloc[tr_mask]
        y_tr_f = y_tr.iloc[tr_mask]
        w_tr = all_weights[tr][tr_mask]

        if clf_task:
            model, _ = train_logistic_regression(X_tr_f, y_tr_f, sample_weight=w_tr)
            score = model.predict_proba(X_te)[:, 1]
        else:
            model, _ = train_ridge_regression(X_tr_f, y_tr_f, sample_weight=w_tr)
            score = model.predict(X_te)

        fold_scores = pd.DataFrame({"date": meta_te["date"], "symbol": meta_te["symbol"], "score": score})
        fold_scores["fold"] = fold_id
        oos_rows.append(fold_scores)

    oos_scores = pd.concat(oos_rows, ignore_index=True)

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
    emit_daily(last_date, last_positions, metrics_report)

    return {
        "equity_curve": equity_curve,
        "backtest_metrics": metrics_bt,
        "dsr": dsr_res,
        "pbo": pbo_res,
        "target_weights": tw,
        "signals": signals[["date", "symbol", "target_quantity"]],
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
