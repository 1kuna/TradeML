"""
Portfolio construction with full SSOT v2 compliance.

API (blueprint): portfolio.build(scores_df, risk_cfg) -> dict

Scores per date,symbol are converted to target weights with:
- Cross-sectional z-score per date
- Gross exposure cap
- Per-name cap
- Kelly fraction scaling
- Volatility targeting (scale portfolio to target annualized vol)
- Turnover constraints (limit daily weight changes)
- Shorting realism (cap shorts for unknown borrow names)

Output dict includes target_weights DataFrame and metadata.
"""

from __future__ import annotations

import os
from datetime import date as Date, timedelta
from typing import Dict, Optional, Set

import numpy as np
import pandas as pd
from loguru import logger


def _per_date_zscore(df: pd.DataFrame, score_col: str = "score") -> pd.Series:
    def z(g: pd.Series) -> pd.Series:
        mu = g.mean()
        sd = g.std(ddof=1)
        if sd == 0 or np.isnan(sd):
            return pd.Series(0.0, index=g.index)
        return (g - mu) / sd

    return df.groupby("date")[score_col].transform(z)


def _load_returns_for_vol(
    symbols: Set[str],
    end_date: Date,
    lookback_days: int = 60,
) -> pd.DataFrame:
    """Load historical returns for volatility estimation.

    Returns DataFrame with columns: date, symbol, ret (log return).
    """
    try:
        from data_layer.curated.loaders import load_price_panel
    except ImportError:
        logger.warning("Could not import curated loaders; volatility targeting disabled")
        return pd.DataFrame(columns=["date", "symbol", "ret"])

    start_date = end_date - timedelta(days=lookback_days + 30)  # buffer for weekends/holidays
    panel = load_price_panel(list(symbols), start_date, end_date)

    if panel.empty:
        return pd.DataFrame(columns=["date", "symbol", "ret"])

    # Compute log returns per symbol
    panel = panel.sort_values(["symbol", "date"])
    panel["ret"] = panel.groupby("symbol")["close"].transform(
        lambda x: np.log(x / x.shift(1))
    )
    panel = panel.dropna(subset=["ret"])

    return panel[["date", "symbol", "ret"]]


def _compute_realized_vol(
    returns_df: pd.DataFrame,
    symbols: Set[str],
    vol_lookback_days: int = 20,
    vol_floor: float = 0.05,
    annualization_factor: float = 252.0,
) -> Dict[str, float]:
    """Compute annualized realized volatility per symbol.

    Returns dict mapping symbol -> annualized vol (capped at vol_floor minimum).
    Uses median vol as fallback for symbols with insufficient data.
    """
    if returns_df.empty:
        return {s: vol_floor for s in symbols}

    # Get most recent N days of returns per symbol
    vol_map = {}
    for sym in symbols:
        sym_ret = returns_df[returns_df["symbol"] == sym]["ret"].tail(vol_lookback_days)
        if len(sym_ret) >= 10:  # minimum 10 observations
            daily_vol = sym_ret.std(ddof=1)
            ann_vol = daily_vol * np.sqrt(annualization_factor)
            vol_map[sym] = max(ann_vol, vol_floor)
        else:
            vol_map[sym] = None  # mark for fallback

    # Compute median for fallback
    valid_vols = [v for v in vol_map.values() if v is not None]
    median_vol = np.median(valid_vols) if valid_vols else vol_floor

    # Fill missing with median
    for sym in vol_map:
        if vol_map[sym] is None:
            vol_map[sym] = median_vol
            logger.debug(f"Using median vol {median_vol:.4f} for {sym} (insufficient data)")

    return vol_map


def _apply_vol_targeting(
    weights: pd.Series,
    vol_map: Dict[str, float],
    symbols: pd.Series,
    vol_target: float,
    gross_cap: float,
) -> pd.Series:
    """Scale weights to target annualized portfolio volatility.

    Uses simplified diagonal covariance (ignores correlations) for speed.
    portfolio_vol = sqrt(sum(w_i^2 * sigma_i^2))
    """
    # Build vol array aligned with weights
    vols = symbols.map(vol_map).fillna(vol_map.get("__median__", 0.15)).values
    w = weights.values

    # Compute portfolio vol (diagonal approximation)
    port_var = np.sum(w**2 * vols**2)
    port_vol = np.sqrt(port_var) if port_var > 0 else 0.0

    if port_vol < 1e-8:
        return weights  # no scaling needed if near-zero vol

    # Scale factor to hit target
    scale = vol_target / port_vol

    # Apply scaling but respect gross cap
    scaled_w = w * scale
    scaled_gross = np.abs(scaled_w).sum()

    if scaled_gross > gross_cap:
        # Reduce to gross cap
        scaled_w = scaled_w * (gross_cap / scaled_gross)
        logger.debug(f"Vol targeting: scaled {scale:.2f}x but capped by gross {gross_cap}")
    else:
        logger.debug(f"Vol targeting: scaled {scale:.2f}x, port_vol {port_vol:.4f} -> {vol_target:.4f}")

    return pd.Series(scaled_w, index=weights.index)


def _apply_turnover_constraint(
    new_weights: pd.DataFrame,
    prev_weights: Optional[pd.DataFrame],
    max_turnover: float,
) -> pd.DataFrame:
    """Constrain turnover by scaling weight deltas if needed.

    new_weights and prev_weights: DataFrames with columns: date, symbol, target_w
    """
    if prev_weights is None or prev_weights.empty or max_turnover >= 2.0:
        return new_weights

    result_frames = []

    for dt, grp in new_weights.groupby("date"):
        prev_day = prev_weights[prev_weights["date"] == dt]
        if prev_day.empty:
            # No previous weights for this date; check day before
            prev_dates = prev_weights["date"].unique()
            prev_dates = sorted([d for d in prev_dates if d < dt])
            if prev_dates:
                prev_day = prev_weights[prev_weights["date"] == prev_dates[-1]]

        if prev_day.empty:
            result_frames.append(grp)
            continue

        # Merge to align symbols
        merged = grp.merge(
            prev_day[["symbol", "target_w"]],
            on="symbol",
            how="left",
            suffixes=("", "_prev"),
        )
        merged["target_w_prev"] = merged["target_w_prev"].fillna(0.0)

        # Calculate turnover
        delta = (merged["target_w"] - merged["target_w_prev"]).abs()
        turnover = delta.sum()

        if turnover > max_turnover:
            # Scale deltas proportionally
            scale = max_turnover / turnover
            merged["target_w"] = merged["target_w_prev"] + (
                merged["target_w"] - merged["target_w_prev"]
            ) * scale
            logger.info(
                f"Turnover constraint: {dt} reduced from {turnover:.2%} to {max_turnover:.2%}"
            )

        result_frames.append(merged[["date", "symbol", "target_w"]])

    return pd.concat(result_frames, ignore_index=True)


def _apply_shorting_realism(
    weights: pd.Series,
    symbols: pd.Series,
    borrow_cap_unknown: float,
    easy_to_borrow: Optional[Set[str]] = None,
) -> pd.Series:
    """Apply shorting constraints for unknown borrow availability.

    Caps short positions at borrow_cap_unknown for symbols not in easy_to_borrow set.
    """
    if easy_to_borrow is None:
        easy_to_borrow = set()  # all treated as unknown

    w = weights.values.copy()
    syms = symbols.values
    constraints_bound = []

    for i, (sym, wt) in enumerate(zip(syms, w)):
        if wt < 0:  # short position
            if sym not in easy_to_borrow:
                if abs(wt) > borrow_cap_unknown:
                    constraints_bound.append(sym)
                    w[i] = -borrow_cap_unknown

    if constraints_bound:
        logger.warning(
            f"Shorting constraint: capped {len(constraints_bound)} positions at "
            f"{borrow_cap_unknown:.1%} (unknown borrow): {constraints_bound[:5]}..."
        )

    return pd.Series(w, index=weights.index)


def _load_easy_to_borrow(borrow_data_path: Optional[str] = None) -> Set[str]:
    """Load set of easy-to-borrow symbols from file if available."""
    if borrow_data_path is None:
        borrow_data_path = os.getenv("BORROW_DATA_PATH", "")

    if not borrow_data_path or not os.path.exists(borrow_data_path):
        return set()  # all treated as unknown

    try:
        df = pd.read_csv(borrow_data_path)
        if "symbol" in df.columns and "tier" in df.columns:
            easy = df[df["tier"].isin(["Easy", "easy", "EASY", "GC"])]["symbol"]
            return set(easy.str.upper())
        elif "symbol" in df.columns:
            return set(df["symbol"].str.upper())
    except Exception as e:
        logger.warning(f"Failed to load borrow data from {borrow_data_path}: {e}")

    return set()


def _load_rl_policy(model_path: str):
    """Load RL policy model from disk.

    Returns None if loading fails or dependencies missing.
    """
    if not model_path or not os.path.exists(model_path):
        return None

    try:
        import joblib
        return joblib.load(model_path)
    except ImportError:
        logger.debug("joblib not available; RL policy disabled")
        return None
    except Exception as e:
        logger.warning(f"Failed to load RL policy from {model_path}: {e}")
        return None


def _apply_rl_adjustment(
    weights_df: pd.DataFrame,
    rl_policy,
    features_df: Optional[pd.DataFrame] = None,
    gross_cap: float = 1.0,
) -> pd.DataFrame:
    """Apply RL policy adjustments to portfolio weights.

    Per SSOT v2 §4.8, RL modulates sizes/schedules; signals remain supervised.
    RL outputs multiplicative adjustments to portfolio weights.

    Parameters
    ----------
    weights_df : DataFrame
        Columns: date, symbol, target_w
    rl_policy : model object
        Trained RL policy with predict() method
    features_df : DataFrame, optional
        State features for RL. If None, uses default features.
    gross_cap : float
        Maximum gross exposure after adjustment

    Returns
    -------
    DataFrame with adjusted target_w
    """
    if rl_policy is None:
        return weights_df

    result_rows = []

    for dt, grp in weights_df.groupby("date"):
        # Build state features for RL
        if features_df is not None and "date" in features_df.columns:
            day_features = features_df[features_df["date"] == dt]
        else:
            # Use weights as minimal state
            day_features = grp.copy()

        if day_features.empty:
            result_rows.append(grp)
            continue

        try:
            # Build feature matrix for prediction
            # Default: use absolute weight, sign, and rank as state
            state_cols = [c for c in day_features.columns if c.startswith("state_")]
            if state_cols:
                X = day_features[state_cols].values
            else:
                # Create minimal state from weights
                w = grp["target_w"].values
                X = np.column_stack([
                    np.abs(w),
                    np.sign(w),
                    np.argsort(np.argsort(-np.abs(w))) / len(w),  # rank
                ])

            # Get RL adjustments (expected to be multipliers around 1.0)
            adjustments = rl_policy.predict(X)

            # Clip adjustments to reasonable range [0.5, 2.0]
            adjustments = np.clip(adjustments, 0.5, 2.0)

            # Apply multiplicative adjustment
            adjusted_w = grp["target_w"].values * adjustments

            # Renormalize to respect gross cap
            gross = np.abs(adjusted_w).sum()
            if gross > gross_cap:
                adjusted_w = adjusted_w * (gross_cap / gross)

            grp = grp.copy()
            grp["target_w"] = adjusted_w
            result_rows.append(grp)

        except Exception as e:
            logger.debug(f"RL adjustment failed for {dt}: {e}")
            result_rows.append(grp)

    return pd.concat(result_rows, ignore_index=True)


def build(
    scores_df: pd.DataFrame,
    risk_cfg: Optional[Dict] = None,
    prev_weights: Optional[pd.DataFrame] = None,
) -> Dict:
    """Convert scores to target weights with full SSOT v2 constraints.

    Parameters
    ----------
    scores_df : DataFrame
        Must have columns: date, symbol, score
    risk_cfg : dict, optional
        Configuration including:
        - gross_cap: float (default 1.0) - total |w| per date
        - max_name: float (default 0.05) - per-name cap
        - kelly_fraction: float (default 1.0) - Kelly scaling
        - vol_target: float (default None) - target annualized vol (e.g., 0.11 for 11%)
        - vol_lookback_days: int (default 20) - lookback for realized vol
        - vol_floor: float (default 0.05) - minimum vol estimate
        - max_turnover: float (default 2.0) - max daily turnover (2.0 = no constraint)
        - borrow_cap_unknown: float (default 0.02) - max short for unknown borrow
        - borrow_data_path: str (optional) - path to borrow availability CSV
        - rl_policy_path: str (optional) - path to RL policy for size adjustment
        - rl_enabled: bool (default False) - whether to apply RL adjustments
    prev_weights : DataFrame, optional
        Previous weights for turnover constraint. Columns: date, symbol, target_w

    Returns
    -------
    dict with target_weights DataFrame and metadata
    """
    if risk_cfg is None:
        risk_cfg = {}

    # Core parameters
    gross_cap = float(risk_cfg.get("gross_cap", 1.0))
    max_name = float(risk_cfg.get("max_name", 0.05))
    kelly_fraction = float(risk_cfg.get("kelly_fraction", 1.0))

    # Volatility targeting parameters (SSOT §4.7)
    vol_target = risk_cfg.get("vol_target")  # None = disabled
    vol_lookback_days = int(risk_cfg.get("vol_lookback_days", 20))
    vol_floor = float(risk_cfg.get("vol_floor", 0.05))

    # Turnover constraint (SSOT §4.7)
    max_turnover = float(risk_cfg.get("max_turnover", 2.0))  # 2.0 = disabled

    # Shorting realism (SSOT §4.7)
    borrow_cap_unknown = float(risk_cfg.get("borrow_cap_unknown", 0.02))
    borrow_data_path = risk_cfg.get("borrow_data_path")

    # RL adjustment (SSOT §4.8)
    rl_enabled = bool(risk_cfg.get("rl_enabled", False))
    rl_policy_path = risk_cfg.get("rl_policy_path")
    rl_policy = None
    if rl_enabled and rl_policy_path:
        rl_policy = _load_rl_policy(rl_policy_path)
        if rl_policy:
            logger.info(f"Loaded RL policy from {rl_policy_path}")

    df = scores_df.copy()
    if not {"date", "symbol", "score"}.issubset(df.columns):
        raise ValueError("scores_df must have columns: date, symbol, score")

    df["z"] = _per_date_zscore(df, "score")

    # Load volatility data if vol targeting is enabled
    vol_map = None
    if vol_target is not None and vol_target > 0:
        symbols = set(df["symbol"].unique())
        max_date = df["date"].max()
        if isinstance(max_date, str):
            max_date = Date.fromisoformat(max_date)
        elif hasattr(max_date, "date"):
            max_date = max_date.date()

        returns_df = _load_returns_for_vol(symbols, max_date, vol_lookback_days + 40)
        vol_map = _compute_realized_vol(
            returns_df, symbols, vol_lookback_days, vol_floor
        )

    # Load borrow availability data
    easy_to_borrow = _load_easy_to_borrow(borrow_data_path)

    # Convert z-scores to weights per date
    def scale_group(g: pd.DataFrame) -> pd.DataFrame:
        if (g["z"].abs().sum() == 0) or np.isnan(g["z"].abs().sum()):
            g["target_w"] = 0.0
            return g

        # Initial weights from z-scores
        w = g["z"] / g["z"].abs().sum() * gross_cap

        # Apply per-name cap
        w = w.clip(-max_name, max_name)
        if w.abs().sum() > 0:
            w = w / w.abs().sum() * gross_cap

        # Apply Kelly fraction
        w = w * kelly_fraction

        # Apply shorting realism
        w = _apply_shorting_realism(
            pd.Series(w.values, index=g.index),
            g["symbol"],
            borrow_cap_unknown,
            easy_to_borrow,
        )

        # Apply volatility targeting (per-date)
        if vol_map is not None:
            w = _apply_vol_targeting(
                pd.Series(w.values, index=g.index),
                vol_map,
                g["symbol"],
                vol_target,
                gross_cap,
            )

        g["target_w"] = w.values
        return g

    out = df.groupby("date", group_keys=False).apply(scale_group).reset_index(drop=True)
    out = out[["date", "symbol", "target_w"]]

    # Apply turnover constraint across dates
    if max_turnover < 2.0:
        out = _apply_turnover_constraint(out, prev_weights, max_turnover)

    # Apply RL adjustment (SSOT §4.8 - optional layer)
    rl_applied = False
    if rl_policy is not None:
        out = _apply_rl_adjustment(out, rl_policy, None, gross_cap)
        rl_applied = True
        logger.info("Applied RL size adjustment to portfolio weights")

    # Build result
    constraints_active = []
    if vol_target is not None:
        constraints_active.append(f"vol_target={vol_target:.1%}")
    if max_turnover < 2.0:
        constraints_active.append(f"max_turnover={max_turnover:.1%}")
    if borrow_cap_unknown < 1.0:
        constraints_active.append(f"borrow_cap={borrow_cap_unknown:.1%}")
    if rl_applied:
        constraints_active.append("rl_adjustment")

    result = {
        "target_weights": out,
        "gross_cap": gross_cap,
        "kelly_fraction": kelly_fraction,
        "max_name": max_name,
        "vol_target": vol_target,
        "max_turnover": max_turnover,
        "borrow_cap_unknown": borrow_cap_unknown,
        "rl_enabled": rl_applied,
        "constraints_active": constraints_active,
    }

    logger.info(
        f"Portfolio built: {out['date'].nunique()} dates, gross_cap={gross_cap}, "
        f"kelly={kelly_fraction}, max_name={max_name}, constraints={constraints_active}"
    )
    return result

