# Autopilot Roadmap

This is the north-star operating model for TradeML. `SSOT.md` remains canonical; this file is the short roadmap for execution order.

## 1. Pi Capture Loop

- Run the data node continuously on the Raspberry Pi.
- Keep canonical EOD integrity first: forward bars, repair tasks, and frozen-window coverage gates.
- Saturate independent quota lanes every poll cycle: Alpaca minute bars, Tiingo/Finnhub news, SEC filings/companyfacts, FRED vintages, and low-rate Massive/Twelve Data fillers.
- Burn down unused daily reserve near reset so current/free entitlements do not idle.
- Persist raw minute/news/event archives indefinitely with watermarks; do not expose them to Phase 1 features until PIT-safe curation exists.

## 2. Mac Mini Research Loop

- Consume only curated PIT-safe tables from the NAS.
- Run deterministic walk-forward experiments with costs always on.
- Compare new architectures against the accepted incumbent using IC stability, decile spread, drawdown, turnover, and cost stress.
- Save every run config, feature list, metric file, and promotion decision under `models/{model_name}/run_{timestamp}/`.

## 3. Drift And Health

- Alert when recent live or retrain metrics fall materially below the incumbent's accepted OOS envelope.
- Track feature-distribution drift, GREEN/AMBER/RED coverage drift, vendor 429s, entitlement failures, rows per credit, and idle quota.
- Treat drift as a research signal, not an automatic trading permission change.

## 4. Incumbent Promotion

- Phase 1 promotion remains manual.
- A challenger must pass leakage checks, replayable walk-forward evidence, placebo tests, and cost stress before replacing the incumbent.
- No model can be promoted from raw archive-only lanes until those lanes have PIT-safe curated outputs.
