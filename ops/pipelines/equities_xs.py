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

from data_layer.curated.loaders import load_price_panel
from feature_store.equities.dataset import build_training_dataset
from models.equities_xs.baselines import train_logistic_regression, train_ridge_regression
from portfolio.build import build as build_portfolio
from backtest.engine.backtester import MinimalBacktester
from validation import CPCV, DSRCalculator, PBOCalculator
from ops.reports.emitter import emit_daily


def _to_date(d: object) -> date:
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, date):
        return d
    if isinstance(d, str):
        return datetime.strptime(d, "%Y-%m-%d").date()
    raise TypeError(f"Unsupported date type: {type(d)}")


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

    # Prepare CPCV labels (date + horizon)
    groups = pd.DataFrame({
        "date": ds.X["date"],
        "horizon_days": ds.meta["horizon_days"],
    })

    # Choose baseline
    clf_task = (cfg.label_type == "triple_barrier")

    cv = CPCV(n_folds=cfg.n_folds, embargo_days=cfg.embargo_days)
    splits = cv.split(ds.X.assign(date=groups["date"]), groups)

    # Collect OOS predictions
    oos_rows: List[Dict] = []
    for fold_id, (tr, te) in enumerate(splits):
        X_tr = ds.X.drop(columns=["date", "symbol"]).iloc[tr]
        y_tr = ds.y.iloc[tr]
        X_te = ds.X.drop(columns=["date", "symbol"]).iloc[te]
        meta_te = ds.X.iloc[te][["date", "symbol"]].reset_index(drop=True)

        if clf_task:
            model, _ = train_logistic_regression(X_tr, y_tr)
            score = model.predict_proba(X_te)[:, 1]
        else:
            model, _ = train_ridge_regression(X_tr, y_tr)
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
                X_tr = ds.X.drop(columns=["date", "symbol"]).iloc[tr]
                y_tr = ds.y.iloc[tr]
                X_te = ds.X.drop(columns=["date", "symbol"]).iloc[te]
                y_te = ds.y.iloc[te]
                # Split train further for IS via a simple holdout: last 10% of tr as val
                cut = int(0.9 * len(tr))
                tr_idx = np.array(tr[:cut])
                val_idx = np.array(tr[cut:])
                Xt, yt = ds.X.drop(columns=["date", "symbol"]).iloc[tr_idx], ds.y.iloc[tr_idx]
                Xv, yv = ds.X.drop(columns=["date", "symbol"]).iloc[val_idx], ds.y.iloc[val_idx]
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
                X_tr = ds.X.drop(columns=["date", "symbol"]).iloc[tr]
                y_tr = ds.y.iloc[tr]
                X_te = ds.X.drop(columns=["date", "symbol"]).iloc[te]
                y_te = ds.y.iloc[te]
                cut = int(0.9 * len(tr))
                tr_idx = np.array(tr[:cut])
                val_idx = np.array(tr[cut:])
                Xt, yt = ds.X.drop(columns=["date", "symbol"]).iloc[tr_idx], ds.y.iloc[tr_idx]
                Xv, yv = ds.X.drop(columns=["date", "symbol"]).iloc[val_idx], ds.y.iloc[val_idx]
                model = Pipeline([("scaler", StandardScaler()), ("reg", Ridge(random_state=42, **params))])
                model.fit(Xt, yt)
                yv_pred = model.predict(Xv)
                yte_pred = model.predict(X_te)
                is_perf[ci, ti] = r2_score(yv, yv_pred) if len(np.unique(yv)) > 1 else 0.0
                oos_perf[ci, ti] = r2_score(y_te, yte_pred) if len(np.unique(y_te)) > 1 else 0.0

    pbo_calc = PBOCalculator(n_trials=oos_perf.shape[1])
    pbo_res = pbo_calc.calculate_pbo(is_perf, oos_perf)

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
