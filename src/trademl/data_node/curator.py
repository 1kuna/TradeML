"""Curate raw bars into adjusted OHLCV research tables."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


PRICE_COLUMNS = ["open", "high", "low", "close", "vwap"]


@dataclass(slots=True)
class CuratorResult:
    """Result of a curation pass."""

    frame: pd.DataFrame
    adjustment_log: pd.DataFrame


class Curator:
    """Apply split and dividend adjustments and persist curated parquet."""

    def apply_adjustments(self, raw_bars: pd.DataFrame, corp_actions: pd.DataFrame) -> CuratorResult:
        """Return a backward-adjusted OHLCV dataframe with an adjustment log."""
        if raw_bars.empty:
            return CuratorResult(frame=raw_bars.copy(), adjustment_log=pd.DataFrame())

        curated = raw_bars.copy()
        curated["date"] = pd.to_datetime(curated["date"])
        curated = curated.sort_values(["symbol", "date"]).reset_index(drop=True)
        actions = corp_actions.copy()
        if actions.empty:
            curated["curated_at"] = pd.Timestamp.now(tz="UTC")
            return CuratorResult(frame=curated, adjustment_log=pd.DataFrame(columns=["symbol", "date", "event_type", "ratio", "source"]))

        actions["ex_date"] = pd.to_datetime(actions["ex_date"])
        adjustment_log: list[dict[str, object]] = []

        for symbol, symbol_actions in actions.sort_values("ex_date").groupby("symbol"):
            symbol_mask = curated["symbol"] == symbol
            symbol_frame = curated.loc[symbol_mask].copy()
            if symbol_frame.empty:
                continue

            for action in symbol_actions.itertuples(index=False):
                prior_mask = symbol_frame["date"] < action.ex_date
                if not prior_mask.any():
                    continue

                if action.event_type == "split":
                    ratio = float(action.ratio)
                    symbol_frame.loc[prior_mask, PRICE_COLUMNS] = symbol_frame.loc[prior_mask, PRICE_COLUMNS] * ratio
                    symbol_frame.loc[prior_mask, "volume"] = symbol_frame.loc[prior_mask, "volume"] / ratio
                    adjustment_log.append(
                        {"symbol": symbol, "date": action.ex_date.date(), "event_type": "split", "ratio": ratio, "source": action.source}
                    )
                elif action.event_type == "dividend":
                    pre_ex_close = symbol_frame.loc[symbol_frame["date"] < action.ex_date, "close"].iloc[-1]
                    dividend_amount = float(action.amount if getattr(action, "amount", None) is not None and pd.notna(action.amount) else action.ratio)
                    dividend_ratio = float((pre_ex_close - dividend_amount) / pre_ex_close)
                    symbol_frame.loc[prior_mask, PRICE_COLUMNS] = symbol_frame.loc[prior_mask, PRICE_COLUMNS] * dividend_ratio
                    adjustment_log.append(
                        {
                            "symbol": symbol,
                            "date": action.ex_date.date(),
                            "event_type": "dividend",
                            "ratio": dividend_ratio,
                            "source": action.source,
                        }
                    )

            curated.loc[symbol_mask, symbol_frame.columns] = symbol_frame.values

        curated["date"] = curated["date"].dt.date
        curated["curated_at"] = pd.Timestamp.now(tz="UTC")
        return CuratorResult(frame=curated, adjustment_log=pd.DataFrame(adjustment_log))

    def write_curated(
        self,
        *,
        raw_bars: pd.DataFrame,
        corp_actions: pd.DataFrame,
        output_root: Path,
        changed_dates: list[str] | None = None,
        adjustment_log_path: Path | None = None,
    ) -> CuratorResult:
        """Write curated parquet partitioned by date."""
        result = self.apply_adjustments(raw_bars=raw_bars, corp_actions=corp_actions)
        output_root.mkdir(parents=True, exist_ok=True)
        changed_date_set = {pd.Timestamp(day).strftime("%Y-%m-%d") for day in changed_dates} if changed_dates else None
        for day, day_frame in result.frame.groupby("date"):
            day_key = pd.Timestamp(day).strftime("%Y-%m-%d")
            if changed_date_set is not None and day_key not in changed_date_set:
                continue
            partition = output_root / f"date={pd.Timestamp(day).strftime('%Y-%m-%d')}"
            partition.mkdir(parents=True, exist_ok=True)
            day_frame.to_parquet(partition / "data.parquet", index=False)
        if adjustment_log_path is not None:
            adjustment_log_path.parent.mkdir(parents=True, exist_ok=True)
            result.adjustment_log.to_parquet(adjustment_log_path, index=False)
        return result
