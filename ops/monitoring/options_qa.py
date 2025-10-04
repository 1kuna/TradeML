from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import date, datetime
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from loguru import logger

from data_layer.storage.s3_client import get_s3_client


_OCC_RE = re.compile(r"^([A-Z]{1,6})(\d{2})(\d{2})(\d{2})([CP])(\d{8})$")


def _parse_occ_symbol(sym: str) -> Optional[Tuple[str, str, float, str]]:
    m = _OCC_RE.match(sym or "")
    if not m:
        return None
    ul, yy, mm, dd, cp, strike = m.groups()
    year = int("20" + yy)
    exp = f"{year:04d}-{int(mm):02d}-{int(dd):02d}"
    strike_val = int(strike) / 1000.0
    return ul, exp, strike_val, cp


def _storage_backend() -> str:
    return os.getenv("STORAGE_BACKEND", "s3").lower()


def _s3_bucket() -> str:
    return os.getenv("S3_BUCKET", "ata")


def _read_parquet_local(path: Path) -> Optional[pd.DataFrame]:
    try:
        if not path.exists():
            return None
        return pd.read_parquet(path)
    except Exception as e:
        logger.warning(f"Local parquet read failed: {e}")
        return None


def _read_parquet_s3(key: str) -> Optional[pd.DataFrame]:
    try:
        s3 = get_s3_client()
        data, _ = s3.get_object(key)
        import pyarrow.parquet as pq
        import pyarrow as pa

        table = pq.read_table(BytesIO(data))
        return table.to_pandas()
    except Exception as e:
        logger.warning(f"S3 parquet read failed for {key}: {e}")
        return None


def _load_chain_df(vendor: str, day: date, underlier: str) -> Optional[pd.DataFrame]:
    day_str = day.isoformat()
    if _storage_backend() == "s3":
        key = f"raw/{vendor}/options_chain/date={day_str}/underlier={underlier}/data.parquet"
        return _read_parquet_s3(key)
    else:
        path = Path("data_layer/raw") / vendor / "options_chain" / f"date={day_str}" / f"underlier={underlier}" / "data.parquet"
        return _read_parquet_local(path)


def _normalize_alpaca_df(df: pd.DataFrame, underlier: str) -> pd.DataFrame:
    # Expect column 'contract' or fallback attempt from any present
    if "contract" not in df.columns:
        # If coming from a different mapping, try 'symbol' or index
        if "symbol" in df.columns:
            df = df.rename(columns={"symbol": "contract"})
        else:
            return pd.DataFrame()
    rows = []
    for _, r in df.iterrows():
        parsed = _parse_occ_symbol(str(r.get("contract", "")))
        if not parsed:
            continue
        ul, exp, strike, cp = parsed
        if ul != underlier:
            # In rare cases root differs; keep but tag
            pass
        bid = r.get("bid")
        ask = r.get("ask")
        rows.append({
            "underlier": ul,
            "expiry": exp,
            "strike": float(strike),
            "cp_flag": cp,
            "bid": float(bid) if pd.notna(bid) else None,
            "ask": float(ask) if pd.notna(ask) else None,
        })
    out = pd.DataFrame.from_records(rows)
    if out.empty:
        return out
    return out.sort_values(["expiry", "strike", "cp_flag"]).reset_index(drop=True)


def _normalize_finnhub_df(df: pd.DataFrame, underlier: str) -> pd.DataFrame:
    if df.empty:
        return df
    # Ensure expected columns exist
    cols = set(df.columns)
    need = {"underlier", "expiry", "strike", "cp_flag"}
    if not need.issubset(cols):
        # Attempt to rename if case differs
        ren = {}
        for c in list(need):
            for col in df.columns:
                if col.lower() == c:
                    ren[col] = c
        if ren:
            df = df.rename(columns=ren)
    # Keep only target underlier
    try:
        df = df[df["underlier"].str.upper() == underlier.upper()]
    except Exception:
        pass
    # Normalize types
    try:
        df["expiry"] = pd.to_datetime(df["expiry"]).dt.date.astype(str)
    except Exception:
        pass
    try:
        df["strike"] = df["strike"].astype(float)
        df["strike"] = (df["strike"] * 1000.0).round().astype(int) / 1000.0
    except Exception:
        pass
    if "cp_flag" in df.columns:
        df["cp_flag"] = df["cp_flag"].astype(str).str.upper().str[0]
    if "bid" in df.columns:
        df["bid"] = pd.to_numeric(df["bid"], errors="coerce")
    if "ask" in df.columns:
        df["ask"] = pd.to_numeric(df["ask"], errors="coerce")
    out = df[[c for c in ["underlier", "expiry", "strike", "cp_flag", "bid", "ask"] if c in df.columns]].copy()
    return out.sort_values(["expiry", "strike", "cp_flag"]).reset_index(drop=True)


def _mid(bid: Optional[float], ask: Optional[float]) -> Optional[float]:
    if bid is None or ask is None:
        return None
    if bid <= 0 or ask <= 0:
        return None
    return (bid + ask) / 2.0


def _mid_stats(alp: pd.DataFrame, fh: pd.DataFrame) -> Dict:
    # Join on key (expiry, strike, cp_flag)
    if alp.empty or fh.empty:
        return {}
    key = ["expiry", "strike", "cp_flag"]
    a = alp[key + ["bid", "ask"]].copy()
    a.columns = ["expiry", "strike", "cp_flag", "a_bid", "a_ask"]
    f = fh[key + ["bid", "ask"]].copy()
    f.columns = ["expiry", "strike", "cp_flag", "f_bid", "f_ask"]
    j = pd.merge(a, f, on=key, how="inner")
    if j.empty:
        return {}
    j["a_mid"] = j.apply(lambda r: _mid(r.a_bid, r.a_ask), axis=1)
    j["f_mid"] = j.apply(lambda r: _mid(r.f_bid, r.f_ask), axis=1)
    j = j.dropna(subset=["a_mid", "f_mid"])
    if j.empty:
        return {}
    j["abs_diff"] = (j["a_mid"] - j["f_mid"]).abs()
    j["rel_diff"] = j["abs_diff"] / j["f_mid"].replace(0, pd.NA)
    stats = {
        "count": int(len(j)),
        "abs_diff_mean": float(j["abs_diff"].mean()),
        "abs_diff_p95": float(j["abs_diff"].quantile(0.95)),
        "rel_diff_mean": float(j["rel_diff"].mean(skipna=True)) if j["rel_diff"].notna().any() else None,
        "rel_diff_p95": float(j["rel_diff"].quantile(0.95)) if j["rel_diff"].notna().any() else None,
        "gt_10pct": int((j["rel_diff"] > 0.10).sum(skipna=True)) if j["rel_diff"].notna().any() else 0,
    }
    return stats


def _write_report(day: date, underlier: str, payload: Dict):
    payload["asof"] = day.isoformat()
    payload["underlier"] = underlier
    if _storage_backend() == "s3":
        key = f"manifests/qa/options_chain/{day.isoformat()}/underlier={underlier}/report.json"
        s3 = get_s3_client()
        s3.put_json(key, payload)
    else:
        out = Path("manifests/qa/options_chain") / day.isoformat() / f"underlier={underlier}"
        out.mkdir(parents=True, exist_ok=True)
        (out / "report.json").write_text(json.dumps(payload, indent=2))


def chain_consistency_report(asof: date, underliers: List[str], feed: Optional[str] = None, max_underliers: int = 10):
    """Compare Alpaca vs Finnhub option chains for given date and write per-underlier reports.

    Notes are added when Alpaca feed is 'indicative' (delayed, modified quotes).
    """
    ul_list = underliers[: max_underliers or len(underliers)] if underliers else []
    if not ul_list:
        logger.info("options_qa: no underliers provided")
        return
    feed = feed or os.getenv("ALPACA_OPTIONS_FEED") or "indicative"
    for ul in ul_list:
        try:
            alp = _load_chain_df("alpaca", asof, ul) or pd.DataFrame()
            fh = _load_chain_df("finnhub", asof, ul) or pd.DataFrame()
            a_n = _normalize_alpaca_df(alp, ul)
            f_n = _normalize_finnhub_df(fh, ul)
            # Set comparisons
            a_keys = set(zip(a_n.get("expiry", []), a_n.get("strike", []), a_n.get("cp_flag", [])))
            f_keys = set(zip(f_n.get("expiry", []), f_n.get("strike", []), f_n.get("cp_flag", [])))
            only_alp = list(a_keys - f_keys)
            only_fh = list(f_keys - a_keys)
            # Mid stats
            mid_stats = _mid_stats(a_n, f_n)
            report = {
                "alpaca_count": int(len(a_keys)),
                "finnhub_count": int(len(f_keys)),
                "intersection": int(len(a_keys & f_keys)),
                "only_alpaca_sample": [list(x) for x in only_alp[:20]],
                "only_finnhub_sample": [list(x) for x in only_fh[:20]],
                "mid_stats": mid_stats,
                "notes": {
                    "alpaca_feed": feed,
                    "alpaca_indicative": str(feed).lower() == "indicative",
                    "alpaca_indicative_warning": "Indicative feed is delayed and quotes are modified; use comparisons as qualitative checks only.",
                },
            }
            _write_report(asof, ul, report)
            logger.info(f"options_qa: wrote report for {ul} ({asof})")
        except Exception as e:
            logger.warning(f"options_qa failed for {ul}: {e}")

