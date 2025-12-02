from __future__ import annotations

"""
Shadow trading logger and evaluator (SSOT v2 compliant).

Logs signals (weights) and evaluates multi-week realized PnL for
champion-challenger promotion decisions. Includes:
- Multi-week performance metrics (Sharpe, max drawdown, win rate)
- Decision logging for audit trail
- Integration with promotion flow

SSOT v2 §6.4: promote_if_beat_champion() promotes a challenger only if
shadow trading over N weeks shows consistent improvement.
"""

import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger


def log_signals(asof: date, weights: pd.DataFrame, out_dir: str = "ops/reports/shadow/equities_xs") -> str:
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"signals_{asof.isoformat()}.json"
    payload = {"asof": asof.isoformat(), "weights": weights.to_dict(orient="records")}
    path.write_text(json.dumps(payload, indent=2))
    logger.info(f"Shadow signals logged: {path}")
    return str(path)


def _load_price_panel(start: date, end: date) -> Optional[pd.DataFrame]:
    """Load curated close prices for shadow evaluation."""
    base = Path("data_layer/curated/equities_ohlcv_adj")

    # Try date-partitioned format first
    dates = []
    for p in base.glob("date=*"):
        try:
            ds = pd.to_datetime(p.name.split("=")[-1]).date()
            dates.append(ds)
        except Exception:
            continue

    panel = []
    for ds in dates:
        if not (start <= ds <= end):
            continue
        p = base / f"date={ds.isoformat()}" / "data.parquet"
        try:
            df = pd.read_parquet(p)
            close_col = "close_adj" if "close_adj" in df.columns else "close"
            panel.append(df[["symbol", close_col]].assign(date=ds).rename(columns={close_col: "close"}))
        except Exception:
            pass

    # Try per-symbol format if date-partitioned fails
    if not panel:
        for sym_file in base.glob("*.parquet"):
            try:
                df = pd.read_parquet(sym_file)
                if "date" not in df.columns:
                    continue
                df["date"] = pd.to_datetime(df["date"]).dt.date
                df = df[(df["date"] >= start) & (df["date"] <= end)]
                close_col = "close_adj" if "close_adj" in df.columns else "close"
                panel.append(df[["symbol", "date", close_col]].rename(columns={close_col: "close"}))
            except Exception:
                continue

    if not panel:
        return None

    return pd.concat(panel, ignore_index=True)


def evaluate_shadow(
    start: date,
    end: date,
    in_dir: str = "ops/reports/shadow/equities_xs",
    min_weeks: Optional[int] = None,
) -> Dict:
    """Evaluate shadow trading performance over a period.

    Returns comprehensive metrics for promotion decisions per SSOT v2 §6.4.

    Parameters
    ----------
    start, end : date
        Evaluation window
    in_dir : str
        Directory containing signals_*.json files
    min_weeks : int, optional
        Minimum weeks of data required; returns insufficient_data if not met

    Returns
    -------
    dict with keys:
        - status: "ok", "no_data", "no_prices", "no_pnl", "insufficient_data"
        - period_start, period_end: actual dates of first/last PnL
        - n_days, n_weeks: number of trading days and weeks evaluated
        - pnl_total: cumulative PnL over period
        - pnl_mean: mean daily PnL
        - pnl_std: std dev of daily PnL
        - sharpe: annualized Sharpe ratio (daily * sqrt(252))
        - max_dd: maximum drawdown as fraction (negative)
        - win_rate: fraction of profitable days
        - daily_pnl: list of {date, pnl} for each day
    """
    files = sorted(Path(in_dir).glob("signals_*.json"))
    if not files:
        return {"status": "no_data"}

    # Load curated closes
    px_df = _load_price_panel(start, end)
    if px_df is None or px_df.empty:
        return {"status": "no_prices"}

    px_pivot = px_df.pivot(index="date", columns="symbol", values="close").sort_index()
    dates_sorted = list(px_pivot.index)

    # Evaluate day-over-day returns since signals
    pnl_rows: List[Dict] = []
    for f in files:
        try:
            data = json.loads(f.read_text())
            asof = pd.to_datetime(data["asof"]).date()
        except Exception:
            continue

        if not (start <= asof < end):
            continue

        try:
            w = pd.DataFrame(data["weights"])
            w = w.set_index("symbol")["target_w"].to_dict()
        except Exception:
            continue

        # Next day return from asof to next trading day
        if asof not in dates_sorted:
            continue
        idx = dates_sorted.index(asof)
        if idx + 1 >= len(dates_sorted):
            continue

        dnext = dates_sorted[idx + 1]
        p0 = px_pivot.loc[asof]
        p1 = px_pivot.loc[dnext]
        rets = (p1 / p0 - 1.0).fillna(0.0)
        pnl = sum(rets.get(sym, 0.0) * w.get(sym, 0.0) for sym in w.keys())
        pnl_rows.append({"date": dnext, "pnl": float(pnl)})

    if not pnl_rows:
        return {"status": "no_pnl"}

    dfp = pd.DataFrame(pnl_rows).sort_values("date")
    n_days = len(dfp)
    n_weeks = n_days / 5.0  # approximate trading weeks

    # Check minimum weeks requirement
    if min_weeks is not None and n_weeks < min_weeks:
        return {
            "status": "insufficient_data",
            "n_days": n_days,
            "n_weeks": float(n_weeks),
            "min_weeks_required": min_weeks,
        }

    # Compute comprehensive metrics
    pnl_series = dfp["pnl"].values
    pnl_total = float(pnl_series.sum())
    pnl_mean = float(pnl_series.mean())
    pnl_std = float(pnl_series.std(ddof=1)) if len(pnl_series) > 1 else 0.0

    # Sharpe ratio (annualized)
    sharpe = 0.0
    if pnl_std > 1e-9:
        sharpe = float((pnl_mean / pnl_std) * np.sqrt(252))

    # Maximum drawdown
    cumulative = np.cumsum(pnl_series)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = cumulative - running_max
    max_dd = float(drawdowns.min()) if len(drawdowns) > 0 else 0.0

    # Win rate
    win_rate = float((pnl_series > 0).sum() / len(pnl_series)) if len(pnl_series) > 0 else 0.0

    return {
        "status": "ok",
        "period_start": dfp["date"].min().isoformat() if hasattr(dfp["date"].min(), "isoformat") else str(dfp["date"].min()),
        "period_end": dfp["date"].max().isoformat() if hasattr(dfp["date"].max(), "isoformat") else str(dfp["date"].max()),
        "n_days": n_days,
        "n_weeks": float(n_weeks),
        "pnl_total": pnl_total,
        "pnl_mean": pnl_mean,
        "pnl_std": pnl_std,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "win_rate": win_rate,
        "daily_pnl": dfp.to_dict(orient="records"),
    }


def log_shadow_decision(
    model_name: str,
    challenger_run_id: Optional[str],
    shadow_metrics: Dict,
    decision: str,
    reason: str,
    out_dir: str = "ops/reports/shadow/decisions",
) -> str:
    """Log a shadow trading promotion decision for audit trail.

    Per SSOT v2 §6.4, all promotion decisions must be logged with metrics.

    Parameters
    ----------
    model_name : str
        Name of the model family (e.g., "equities_xs")
    challenger_run_id : str, optional
        MLflow run ID of the challenger model
    shadow_metrics : dict
        Output from evaluate_shadow()
    decision : str
        One of: "promote", "reject", "extend_shadow", "error"
    reason : str
        Human-readable explanation of the decision
    out_dir : str
        Directory for decision logs

    Returns
    -------
    str : Path to the decision log file
    """
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.utcnow().isoformat() + "Z"

    payload = {
        "timestamp": timestamp,
        "model_name": model_name,
        "challenger_run_id": challenger_run_id,
        "shadow_metrics": shadow_metrics,
        "decision": decision,
        "reason": reason,
    }

    # Use timestamp-based filename for append-only audit trail
    filename = f"decision_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{model_name}.json"
    path = d / filename
    path.write_text(json.dumps(payload, indent=2, default=str))

    logger.info(f"Shadow decision logged: {decision} for {model_name} -> {path}")
    return str(path)


def check_shadow_promotion_bars(
    shadow_metrics: Dict,
    min_sharpe: float = 0.0,
    max_drawdown: float = -0.20,
    min_win_rate: float = 0.45,
    min_pnl_mean: float = 0.0,
) -> tuple[bool, str]:
    """Check if shadow metrics pass promotion bars.

    Per SSOT v2 §6.3 and §6.4, models must meet go/no-go bars.

    Parameters
    ----------
    shadow_metrics : dict
        Output from evaluate_shadow()
    min_sharpe : float
        Minimum Sharpe ratio (default 0 = just positive)
    max_drawdown : float
        Maximum allowed drawdown (negative, default -20%)
    min_win_rate : float
        Minimum win rate (default 45%)
    min_pnl_mean : float
        Minimum mean daily PnL (default 0 = positive)

    Returns
    -------
    tuple[bool, str] : (passes, reason)
    """
    if shadow_metrics.get("status") != "ok":
        return False, f"Shadow status not ok: {shadow_metrics.get('status')}"

    sharpe = shadow_metrics.get("sharpe", 0.0)
    max_dd = shadow_metrics.get("max_dd", 0.0)
    win_rate = shadow_metrics.get("win_rate", 0.0)
    pnl_mean = shadow_metrics.get("pnl_mean", 0.0)

    failures = []

    if sharpe < min_sharpe:
        failures.append(f"Sharpe {sharpe:.2f} < {min_sharpe:.2f}")

    if max_dd < max_drawdown:  # max_dd is negative, max_drawdown is threshold
        failures.append(f"Max DD {max_dd:.2%} worse than {max_drawdown:.2%}")

    if win_rate < min_win_rate:
        failures.append(f"Win rate {win_rate:.1%} < {min_win_rate:.1%}")

    if pnl_mean < min_pnl_mean:
        failures.append(f"Mean PnL {pnl_mean:.4f} < {min_pnl_mean:.4f}")

    if failures:
        return False, "; ".join(failures)

    return True, f"Passed: Sharpe={sharpe:.2f}, DD={max_dd:.2%}, WinRate={win_rate:.1%}"

