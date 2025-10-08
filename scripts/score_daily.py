#!/usr/bin/env python3
"""Daily scoring script using trained equities_xs artifacts."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from loguru import logger

try:
    import joblib
except Exception:  # pragma: no cover
    joblib = None  # type: ignore

from data_layer.curated.loaders import load_price_panel
from feature_store.equities.features import compute_equity_features
from models.equities_xs import predict_lgbm
from ops.reports.emitter import emit_daily
from portfolio.build import build as build_portfolio


def _load_feature_list(path: Path) -> List[str]:
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        return data.get("features", [])
    if isinstance(data, list):
        return data
    raise ValueError(f"Unsupported feature list format in {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score daily equities signals from saved model")
    parser.add_argument("--asof", required=True, help="As-of date YYYY-MM-DD")
    parser.add_argument("--model", default="models/equities_xs/artifacts/latest/model.pkl")
    parser.add_argument("--features", default="models/equities_xs/artifacts/latest/feature_list.json")
    parser.add_argument("--capital", type=float, default=1_000_000.0)
    parser.add_argument("--spread-bps", type=float, default=5.0)
    parser.add_argument("--gross-cap", type=float, default=1.0)
    parser.add_argument("--max-name", type=float, default=0.05)
    parser.add_argument("--kelly", type=float, default=1.0)
    parser.add_argument("--symbols", nargs="*", default=None, help="Universe symbols")
    parser.add_argument("--universe-file", type=str, default="data_layer/reference/universe_symbols.txt")
    return parser.parse_args()


def _resolve_universe(args: argparse.Namespace) -> List[str]:
    if args.symbols:
        return [s.strip().upper() for s in args.symbols if s]
    path = Path(args.universe_file)
    if path.exists():
        return [s.strip().upper() for s in path.read_text().splitlines() if s.strip()]
    raise ValueError("Universe not provided; pass --symbols or ensure universe file exists")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    load_dotenv(dotenv_path=repo_root / ".env", override=False)

    args = parse_args()
    universe = _resolve_universe(args)

    if joblib is None:
        raise ImportError("joblib is required to load trained models")

    model_path = Path(args.model)
    feature_path = Path(args.features)
    if not model_path.exists() or not feature_path.exists():
        raise FileNotFoundError("Model or feature list not found; run training first")

    model = joblib.load(model_path)
    feature_list = _load_feature_list(feature_path)

    feats = compute_equity_features(args.asof, universe)
    if feats.empty:
        raise RuntimeError("No features generated for scoring; ensure curated data is available")

    X = feats[[c for c in feats.columns if c.startswith("feature_")]].copy()
    missing = [f for f in feature_list if f not in X.columns]
    for col in missing:
        X[col] = 0.0
    X = X[feature_list]

    # Generate scores
    try:
        scores = predict_lgbm(model, X)
    except Exception:
        if hasattr(model, "predict_proba"):
            scores = model.predict_proba(X)[:, 1]
        else:
            scores = model.predict(X)

    score_df = pd.DataFrame({
        "date": pd.to_datetime(args.asof).date(),
        "symbol": feats["symbol"],
        "score": scores,
    })

    port_res = build_portfolio(
        score_df,
        {
            "gross_cap": args.gross_cap,
            "max_name": args.max_name,
            "kelly_fraction": args.kelly,
        },
    )
    weights = port_res["target_weights"]

    price_panel = load_price_panel(universe, args.asof, args.asof)
    px = price_panel[price_panel["date"] == pd.to_datetime(args.asof).date()][["symbol", "close"]]
    signals = weights.merge(px, on="symbol", how="left")
    if signals["close"].isna().any():
        logger.warning("Missing close prices for some symbols on %s", args.asof)
    signals["target_quantity"] = (signals["target_w"] * args.capital) / signals["close"].replace(0, np.nan)
    signals["target_quantity"] = signals["target_quantity"].fillna(0.0)

    metrics = {
        "status": "scored",
        "n_symbols": int(len(signals)),
        "gross_cap": args.gross_cap,
        "kelly_fraction": args.kelly,
    }
    emit_daily(date.fromisoformat(args.asof), signals[["symbol", "target_w"]], metrics)


if __name__ == "__main__":
    main()
