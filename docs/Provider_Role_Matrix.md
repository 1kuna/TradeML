# Provider Role Matrix

This matrix is the docs-backed operating policy for TradeML collection. It should stay aligned with `src/trademl/data_node/capabilities.py`.

## Canonical bars

| Vendor | Role | Why | Saturation policy | QC role |
| --- | --- | --- | --- | --- |
| Alpaca | Primary forward + recent-history bars | Best multi-symbol batching, same-day edge, stable assets/corp-actions support | Saturate first for forward + recent history | Primary only, not backup QC |
| Tiingo | Deep-history bars | Best long history depth of current live vendors | Keep focused on canonical backfill only | Optional tertiary QC |
| Twelve Data | Supplemental bar overflow | Docs support batch requests; also useful for downstream corp-actions/statements | Use after Alpaca/Tiingo, then spend residual credits on reference/event lanes | Preferred secondary QC |
| Massive | Independent bar validator + overflow | Strong independent OHLCV/reference surface, but free minute cap is tight | Use as QC first, overflow second | Preferred independent QC |
| Finnhub | Disabled for canonical bars | Current entitlement is not reliable enough for candles | Do not spend on canonical bars unless re-audited | Disabled |

## Security master / corp actions / events

| Vendor | Best use | Notes |
| --- | --- | --- |
| Alpaca | Assets, recent corp actions | Good corroborating security-master lane and recent action coverage |
| Alpha Vantage | Listing status, low-rate corp actions | Slow but useful corroboration |
| Massive | Tickers, splits, dividends | Strong independent reference validation |
| Twelve Data | Stocks, dividends, splits, earnings, statements | Broadest non-bar supplemental surface on current keys |
| FMP | Delistings, earnings calendar | Keep to stable free endpoints |
| SEC EDGAR | Company tickers, filings, companyfacts | Authoritative PIT event timeline and issuer linkage |
| FRED / ALFRED | Macro observations + vintages | Independent of equity vendors; should always stay full |
| Finnhub | Company profile, earnings calendar | Useful event/reference lane even though bars stay disabled |

## Documented but not primary today

- Tiingo: fundamentals and supported-ticker metadata are documented, but remain lower priority than deep-history bars.
- Twelve Data: price targets and insider transactions are documented, but remain research-only until Phase 1 data gates are healthy.
- Finnhub: peers, recommendation trends, company news, and financials-reported are documented, but remain non-core until the training-critical corpus is complete.
- SEC EDGAR: bulk submissions/companyfacts archives are documented and should be the next scale-up path for PIT filing coverage.
- FMP: symbol changes and broader historical-price lanes stay audit-gated because current entitlement behavior is inconsistent.

## Runtime rules

1. Canonical bars stay `core-first`.
2. Vendors with `canonical_first` or `canonical_only` policies do not spend auxiliary capacity while canonical bars remain missing.
3. Cross-vendor price QC uses independent backups only:
   - Massive
   - Twelve Data
   - Tiingo (optional tertiary check)
4. Finnhub is not used for canonical bar collection or canonical bar QC unless a fresh live audit restores candle entitlement.
