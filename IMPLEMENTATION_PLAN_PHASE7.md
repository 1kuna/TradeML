# Phase 6 – Edge Collector Gating, Validation, and Universe Hygiene

Status: Planned + In Progress

Goals
- Reduce wasted API calls via universal EOD gating and a persistent bad‑symbols cache.
- Pre‑prune the top‑1000 universe from `IWB_holdings.csv` using cheap bulk endpoints, then continuously validate in‑node.
- Ensure daily vendors don’t hammer “today” before end‑of‑day; fetch once after cutoff.

Scope
- scripts/edge_collector.py: add universal EOD gating (Alpaca/Polygon/FRED) and FRED day‑window iteration. Persist bad symbols.
- utils/bad_symbols.py: simple S3/local JSON cache for invalid tickers per vendor with TTL.
- scripts/validate_universe.py: pre‑prune script to generate `universe_validity.parquet` + `universe_symbols_valid.txt` from `IWB_holdings.csv` using FMP (bulk) + an Alpaca day check (batched).

Design Decisions
- Universal EOD gating: skip “today” until `EOD_TODAY_CUTOFF_UTC_HOUR` (default 22 UTC) controlled by existing helper `_should_fetch_eod_for_day`.
- FRED: instead of only “today”, iterate from bookmark → gated `end_day`; write one partition per day and only advance bookmark on real data.
- Bad‑symbol cache: persisted at `manifests/vendor_bad_symbols.json` on S3 (fallback: local `data_layer/manifests/vendor_bad_symbols.json`), default TTL 14 days; read on startup, write on update.
- Pre‑prune: use low‑cost bulk listings (FMP `stock/list`) to filter obvious invalids; confirm a recent trading day via one Alpaca day request in 100‑symbol batches; options coverage validated incrementally by node (Finnhub) to avoid heavy preflight.

Operational Flow
1) Pre‑prune: `python scripts/validate_universe.py --input IWB_holdings.csv` produces:
   - `data_layer/reference/universe_validity.parquet`
   - `data_layer/reference/universe_symbols_valid.txt`
   - Optionally copy to `data_layer/reference/universe_symbols.txt`.
2) Reset bucket artifacts if needed (only `raw/` + `manifests/`), then restart node.
3) Edge node:
   - Schedules Alpaca/Polygon over bookmark windows with “today” gated.
   - Schedules FRED over window with “today” gated; one partition per day.
   - Finnhub options: rotates a limited per‑cycle subset; updates bad‑symbols cache on explicit invalids.
4) Continuous validation: nightly (future) validator pass with small budgets to catch drift (new listings/delistings).

Milestones
- [x] Plan documented (this file)
- [ ] Universal EOD gating in Alpaca producer
- [ ] FRED day‑window producer with gating
- [ ] Persistent bad‑symbols cache + integration (Polygon/Finnhub)
- [ ] Pre‑prune validator script (FMP + Alpaca day check)
- [ ] Swap universe to pruned list; rerun node

Notes on FMP
- FMP is leveraged for cheap symbol coverage (`stock/list`) and delisted/price as reference. We keep Alpaca/Polygon as primary bar sources; FMP augments universe and corp‑actions/reference.

