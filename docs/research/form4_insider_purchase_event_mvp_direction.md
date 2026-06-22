# Form 4 Insider Purchase Event MVP

Last updated: 2026-05-06

Parent source of truth:

- [Public Event Research Pivot SSOT](public_event_research_pivot_ssot.md)

## Decision

TradeML should pivot its alpha-search roadmap from generic daily cross-sectional
price prediction toward a source-first public-event research pipeline. The
current Pi/Mac/NAS stack should stay in place, but it should become the
infrastructure for falsifiable event hypotheses rather than the main source of
alpha by itself.

The first event MVP should be:

**SEC Form 4 open-market insider purchases, long-only, common stock only.**

This is not because insider buying is guaranteed to be novel alpha. It is
because it is the smallest clean test of the system TradeML actually needs:

- public source ingestion;
- as-of timestamps;
- raw append-only storage;
- deterministic parsing of SEC ownership XML only, because the source fields are
  structured and schema-backed;
- CIK-to-ticker resolution;
- event schemas;
- minute/daily entry and forward-label logic;
- realistic costs and liquidity filters;
- negative controls;
- event-study reporting;
- paper/shadow operational validation.

If this simple event class cannot survive honest timing, controls, costs, and
liquidity, the right answer is to kill it and move on to the next event class.

## First Historical MVP Verdict

Status: implemented and evaluated on the first bounded 2025 historical slice.

The first strict MVP result was **`KILL_OR_REWORK`** for the naive deterministic
Form 4 open-market-buy event rule. The bounded diagnostic rework gate then
evaluated the fixed variant set and produced
**`FORM4_KILLED_BASELINE_COMPLETE`**.

Latest artifacts:

- `/Volumes/dev/TradeML/control/cluster/state/research/form4_ingest/latest.json`
- `/Volumes/dev/TradeML/control/cluster/state/research/form4_market_backfill/latest.json`
- `/Volumes/dev/TradeML/control/cluster/state/research/form4_event_labels/latest.json`
- `/Volumes/dev/TradeML/control/cluster/state/research/form4_event_study/latest.json`
- `/Volumes/dev/TradeML/control/cluster/state/research/form4_baseline_status/latest.json`
- `/Volumes/dev/TradeML/control/cluster/state/research/form4_rework_study/latest.json`
- `/Volumes/dev/TradeML/reports/research/form4_event_study/latest.md`
- `/Volumes/dev/TradeML/reports/research/form4_rework_study/latest.md`

Run scope:

- 2025 SEC Form 4/4-A manifest slice, first 12,000 index rows;
- 12,000 parsed successfully, 0 failed;
- 11,343 candidate issuer events;
- 376 strict eligible open-market common-stock buy candidates;
- 315 primary events labeled after minute/daily market-data coverage;
- 6,652 mechanical acquisition-control labels;
- 1,863 sales-placebo labels.

Primary 5-trading-day abnormal net-return result after the 50 bps round-trip
cost model:

- mean: `0.002507365413593847`;
- median: `-0.00646093227804026`;
- hit rate: `0.4603174603174603`;
- bootstrap 95% CI for mean: `[-0.005802112108748498, 0.011099090268429717]`;
- top-5 absolute contribution: `0.07374135331073463`.

Failed gates:

- `mean_abret_5d_net<=50bps`;
- `median_abret_5d_net<=0`;
- `q4_minus_q1<=75bps`;
- `mechanical_acquisition_codes_separation<=75bps`.

Interpretation:

- The simple Form 4 `P/A` common-stock buy rule is not promotable as a trading
  candidate from this first slice.
- The deterministic strength score is not monotonic: Q4 did not beat Q1.
- Mechanical acquisition controls are too close to the primary signal, so the
  naive rule does not isolate convincing insider-conviction drift.
- Sales placebo separation barely clears the raw 75 bps threshold, but this is
  not enough to rescue the strategy given the primary, median, monotonicity, and
  mechanical-control failures.

This does **not** prove that all insider-buy research is dead. It does mean the
first simple version should not be promoted or paper-traded as an alpha engine.

## Bounded Rework Verdict

The rework gate intentionally tested only fixed, pre-registered variants:

- `clean_timely_unmixed`;
- `officer_director_only`;
- `ceo_cfo_only`;
- `clustered_buyers`;
- `large_purchase_q4`.

Final verdict: **`FORM4_KILLED_BASELINE_COMPLETE`**.

No variant passed all continuation gates. The most interesting slice was
`ceo_cfo_only`, which had positive mean and median abnormal returns plus strong
control separation, but only `47` labeled events. It fails the minimum `n >= 100`
gate and is therefore not enough to continue automated Form 4 search.

Form 4 remains useful as an implemented source-first event-pipeline benchmark,
but it is not a current alpha candidate and must not feed paper/live entries
without a new documented hypothesis and a fresh gate.

The next active child MVP is:

- [SEC 8-K Item Event MVP](sec_8k_item_event_mvp_direction.md)

Implementation lessons from the first run:

- SEC historical ingestion needs raw artifact caching and polite retry/resume
  behavior; both were added.
- Market backfill needs date-specific minute symbol batches and chunked daily
  symbol batches; both were added.
- SEC issuer symbols can be ambiguous multi-symbol strings; those are now
  excluded until a proper security-master resolver exists.
- Event/control label generation must use indexed minute/daily lookups; repeated
  pandas scans became a severe bottleneck at the control-set size.

## Why This Is A Better Near-Term Direction

The current research loop is strong infrastructure, but it is mostly asking a
hard generic question:

> Can a supervised daily ranker predict short-horizon relative returns from
> broad price/liquidity/news/filing aggregates?

That is useful as a sentinel baseline, but it is probably not the best
battlefield for a solo trader. Large liquid names and generic public-news
signals are heavily competed. The more plausible edge is narrow public
information that is structured enough to verify, annoying enough to be missed,
small enough to be ignored by large funds, and tradable only with strict
capacity limits.

Form 4 open-market insider purchases fit the first MVP because they are:

- structured SEC filings;
- public and timestamped;
- long-only by default;
- easy to control against non-conviction transaction codes;
- compatible with current SEC, minute, daily, source-quality, and paper/shadow
  infrastructure;
- simple enough to falsify without LLM extraction or options execution.

This is a structured-source exception. It does not license regex/keyword
semantic extraction for news, 8-K exhibits, company IR, contracts, court records,
or other narrative sources. Narrative public-source semantics must use an LLM or
purpose-built model with schema validation and evidence spans/source offsets.

## Current Stack To Keep

The existing stack should not be ripped out. It becomes the foundation:

- Raspberry Pi collector:
  - vendor capability registry;
  - planner-native queue;
  - raw/archive parquet writes;
  - ingestion ledger;
  - scheduler/controller telemetry;
  - source/data-quality artifacts.
- NAS:
  - raw/source archives;
  - curated market data;
  - research/modeling/control artifacts.
- Mac Mini research loop:
  - experiment registry;
  - strict gates;
  - negative controls;
  - candidate evidence;
  - paper/shadow output;
  - watchdog/audit/Codex feed.
- Generic ranker:
  - keep as a market-context sentinel;
  - keep as a leakage/overfit detector;
  - do not use it as the first Form 4 entry engine.

## Current Gap In The Repo

The repo already has SEC filing index and companyfacts support, but it does not
yet have a first-class event research spine.

Observed current state from code inspection:

- `src/trademl/connectors/sec_edgar.py` supports SEC submissions, company
  tickers, companyfacts, and filing index rows.
- The SEC filing index currently filters to `8-K`, `10-K`, and `10-Q`; Form 4
  is not part of that path.
- There are no first-class `events/` or `entities/` modules yet.
- Existing SEC/news modeling features are aggregate recency/count features, not
  parsed event objects.
- Existing backtesting is portfolio/rank oriented, not event-triggered from
  `first_seen_at` / `tradable_at`.

That means the right implementation is not another model lane. The right
implementation is a small event pipeline.

## Historical Retrieval Path At Scale

Raw source prompt:

- [GPT Pro Form 4 Historical Retrieval Path](source_prompts/2026-05-05_gpt_pro_form4_historical_retrieval_path.md)

Use a hybrid manifest plus raw XML path. Do not rely only on issuer-level SEC
submissions JSON for historical discovery.

The retrieval backbone is:

```text
SEC full-index / daily-index -> canonical accession + archive-path CIK
EDGAR submissions JSON       -> enrichment + acceptanceDateTime + primaryDocument
SEC Archives raw XML         -> authoritative parsed filing body
Complete .txt fallback       -> SGML header + XML extraction when primary XML fails
```

Primary historical manifest:

- build from SEC quarterly `master.idx` or `form.idx`, not current tickers;
- include exact forms `4` and `4/A`;
- start at `2006Q1` for the XML MVP;
- store `archive_cik`, `index_filename`, `accession`,
  `accession_no_dashes`, `filed_date`, `index_year`, `index_quarter`,
  `index_file_hash`, and `index_crawled_at`.

Critical CIK rule:

- `archive_cik` comes from the SEC index filename path and is used for URL
  construction;
- `issuer_cik` comes from ownership XML and is used for company/entity mapping;
- `owner_cik` comes from ownership XML and is used for insider identity;
- never construct archive URLs from accession prefix, `issuerCik`,
  `issuerTradingSymbol`, or `company_tickers.json` CIK unless that archive URL
  was already verified.

Raw archive URL construction must use:

```text
/Archives/edgar/data/{archive_cik_without_leading_zeros}/{accession_without_dashes}/
```

The complete `.txt` fallback URL from an index row is:

```text
https://www.sec.gov/Archives/{index_filename}
```

For accepted timestamps, use this precedence:

1. `submissions.acceptanceDateTime`;
2. SGML `<ACCEPTANCE-DATETIME>`;
3. accession index `Accepted` value.

Store the raw accepted string and normalized ET/UTC values. For trading labels,
treat SEC accepted times as America/New_York market-clock timestamps unless the
parsed source proves a real timezone offset.

Use SEC-compliant request headers and global throttling. The implementation
should use a project user agent and stay below the SEC fair-access ceiling;
target `3-5` requests/second for backfills.

## Raw SEC Archive Storage

Store raw retrieval artifacts append-only before parsing:

```text
raw/sec/form4_manifest/year=YYYY/qtr=Q/
  manifest.parquet

raw/sec/archives/archive_cik={archive_cik}/accession={accession_no_dashes}/
  primary.xml
  complete.txt
  header.sgml
  accession_index.html
  directory_index.json
  metadata.json
```

Curated parser outputs should be separated by grain:

```text
curated/sec/form4/submissions/
curated/sec/form4/reporting_owners/
curated/sec/form4/nonderiv_transactions/
curated/sec/form4/deriv_transactions/
curated/sec/form4/footnotes/
curated/events/form4_open_market_buy_candidates/
```

Do not overwrite amendments into originals. Raw accession rows are immutable.
Build resolved views for amendment replacement logic and a primary event view
that excludes `4/A`.

## MVP Hypothesis

Pre-register this before running the test:

> Top-quartile SEC Form 4 open-market insider purchase events in tradable
> small/mid-cap common stocks produce positive 5-trading-day abnormal net
> returns after public availability, realistic entry timing, costs, liquidity
> filters, duplicate suppression, and negative controls.

Primary failure mode to test:

> The apparent alpha exists only in close-to-close historical labels and
> disappears when entry is delayed until after the Form 4 is actually public.

## Event Definition

Eligible raw transaction rows must satisfy:

- `documentType == "4"`;
- `nonDerivativeTransaction`;
- `transactionCode == "P"`;
- `acquiredDisposedCode == "A"`;
- `transactionFormType == "4"`;
- common stock / ordinary shares only;
- `transactionShares > 0`;
- `transactionPricePerShare > 0`;
- no private-placement, unit, SPAC sponsor, PIPE, warrant, or derivative flags;
- no `notSubjectToSection16` flag;
- issuer maps to a listed tradable common equity at event time.

Important: SEC code `P` means open-market **or private** purchase. The MVP is
trying to test open-market insider conviction buys, so `P` is necessary but not
sufficient. Private purchases, units, warrants, PIPEs, SPAC sponsor purchases,
and derivative-table `P` rows must be parsed and stored but excluded from the
strict primary sample.

Do not include in MVP:

- sales;
- `4/A` amendments as primary signals;
- private-placement or unit purchases;
- derivative exercises;
- option grants;
- awards;
- tax withholding;
- 10b5-1 sales;
- options trading expression;
- LLM sentiment.

## Raw Transaction Table

Add a durable artifact/table equivalent to `form4_transaction_raw`.

Fields:

```python
{
    "accession": str,
    "accession_no_dashes": str,
    "archive_cik": str,
    "index_filename": str,
    "manifest_source": str,
    "form_type": str,
    "document_type": str,
    "issuer_cik": str,
    "issuer_name": str,
    "issuer_trading_symbol_raw": str,
    "owner_cik_set": tuple[str, ...],
    "reporting_owner_cik": str,
    "reporting_owner_name": str,
    "is_director": bool,
    "is_officer": bool,
    "is_ten_percent_owner": bool,
    "is_other": bool,
    "officer_title": str | None,
    "accepted_at_raw": str | None,
    "accepted_at_source": str | None,
    "accepted_at_et": datetime | None,
    "accepted_at_utc": datetime,
    "first_seen_at_utc": datetime,
    "filed_date": date,
    "period_of_report": date | None,
    "date_of_original_submission": date | None,
    "transaction_date": date,
    "transaction_date_raw": str,
    "security_title": str,
    "security_title_normalized": str,
    "transaction_code": str,
    "transaction_form_type": str,
    "acquired_disposed": str,
    "transaction_shares": Decimal,
    "transaction_price": Decimal,
    "transaction_value": Decimal,
    "post_transaction_shares": Decimal | None,
    "direct_or_indirect": str | None,
    "ownership_nature": str | None,
    "field_footnote_ids": dict[str, list[str]],
    "footnotes_text": str | None,
    "probably_private_or_unit_purchase": bool,
    "same_filing_has_sales": bool,
    "same_owner_same_day_has_sales": bool,
    "primary_document": str | None,
    "raw_xml_url": str | None,
    "raw_xml_path": str,
    "raw_xml_hash": str,
    "complete_txt_url": str | None,
    "complete_txt_path": str | None,
    "complete_txt_hash": str | None,
    "primary_xml_source": str,
    "parser_version": str,
    "schema_version": str,
    "source_quality_flags": list[str],
}
```

Use `Decimal`, not float, for shares, prices, post-transaction holdings, and
transaction values. Ownership XML can contain weighted-average prices with
several decimal places, and rounding before event scoring/backtesting would be
unforced data loss.

Footnotes are field-level. Preserve per-field footnote references in
`field_footnote_ids`; do not only concatenate all footnotes into one remarks
blob.

Multiple reporting owners are valid. Preserve `owner_cik_set` as a sorted set
and do not assume the first owner is the primary insider.

## Amendment Policy

Parse and store `4/A`, but do not treat it as a fresh ordinary buy signal.

Primary policy:

- `documentType == "4"` can create `FORM4_OPEN_MARKET_INSIDER_BUY` if all
  strict rules pass.
- `documentType == "4/A"` is parsed, linked to the original, and excluded from
  the primary backtest signal.

Secondary bucket only:

- create `FORM4A_CORRECTION_DISCLOSED_BUY` only when an amendment reveals a
  missing or materially changed buy that was not known from the original filing;
- keep that bucket separate from the primary Form 4 open-market buy sample.

Amendment linking order:

1. exact original accession if available;
2. `issuer_cik + owner_cik_set + dateOfOriginalSubmission`;
3. `issuer_cik + owner_cik_set + periodOfReport + overlapping transaction dates`;
4. `issuer_cik + owner_cik_set + similar row hash within +/-30 calendar days`.

## Issuer-Level Event Table

Add a durable artifact/table equivalent to `insider_purchase_event`.

Fields:

```python
{
    "event_id": str,
    "issuer_cik": str,
    "ticker": str,
    "primary_security_id": str,
    "accessions": list[str],
    "event_type": "FORM4_OPEN_MARKET_INSIDER_BUY",
    "accepted_at_utc": datetime,
    "first_seen_at_utc": datetime,
    "tradable_at_utc": datetime,
    "n_insiders_buying": int,
    "n_directors_buying": int,
    "n_officers_buying": int,
    "ceo_buy": bool,
    "cfo_buy": bool,
    "ten_percent_owner_buy": bool,
    "total_shares_bought": float,
    "total_dollar_value": float,
    "max_single_purchase_value": float,
    "median_purchase_price": float,
    "purchase_value_to_market_cap": float | None,
    "purchase_value_to_adv20": float | None,
    "purchase_value_to_prior_holdings": float | None,
    "days_since_transaction": int,
    "cluster_7d_purchase_value": float,
    "cluster_30d_purchase_value": float,
    "prior_90d_net_insider_purchase_value": float,
    "prior_20d_return": float,
    "prior_60d_return": float,
    "prior_20d_vol": float,
    "prior_20d_adv": float,
    "market_cap_bucket": str,
    "liquidity_bucket": str,
    "news_nearby_flag": bool,
    "earnings_nearby_flag": bool,
    "offering_nearby_flag": bool,
    "mna_nearby_flag": bool,
    "eligibility_pass": bool,
    "exclusion_reasons": list[str],
    "event_strength_score": float,
    "parse_confidence": float,
    "source_quality_score": float,
}
```

## Entity Resolution Scope

Keep v1 deliberately boring:

- issuer CIK to ticker;
- ticker to primary common security;
- issuer CIK to companyfacts entity;
- issuer CIK to sector/SIC where available;
- reporting owner CIK to owner identity.

Rules:

- CIK is the primary issuer key.
- Ticker is the tradable instrument key after point-in-time resolution.
- Preserve archive-path CIK separately from issuer CIK. Archive-path CIK is not
  an entity-resolution key; it is only a retrieval key.
- Do not use fuzzy matching in the MVP.
- Do not infer subsidiaries in the MVP.
- Do not force ambiguous mappings.
- Store unresolved/ambiguous mappings with reasons.

## Parser Edge Cases And Quality Flags

The parser must be namespace-agnostic and defensive around SEC ownership XML.

Required edge-case handling:

- boolean variants: `0/1`, `true/false`, `True/False`, `TRUE/FALSE`;
- transaction dates with timezone suffixes such as `2015-01-26-05:00`, parsed
  as dates without shifting the transaction date;
- field-level footnotes on individual `<value>` elements;
- multiple reporting owners with no assumed primary owner;
- multiple signatures that are not mapped to reporting owners;
- supporting documents such as `EX-24`, `EX-99`, graphics, or HTML docs,
  inventoried as telemetry but not parsed as transactions;
- derivative-table `P` rows, which must never become common-stock buy events;
- mixed `P` and `S` filings, flagged separately from clean buy-only filings;
- missing/zero transaction prices, excluded from the strict primary sample;
- private-placement, PIPE, unit, SPAC sponsor, warrant, and subscription
  language in security titles, remarks, or footnotes.

Quality flags should include:

- `archive_cik_differs_from_issuer_cik`;
- `accession_prefix_differs_from_archive_cik`;
- `raw_primary_404`;
- `used_complete_txt_fallback`;
- `document_type_mismatch`;
- `amendment`;
- `missing_date_of_original_submission`;
- `multiple_reporting_owners`;
- `supporting_documents_present`;
- `graphic_documents_present`;
- `mixed_p_and_s`;
- `derivative_p_present`;
- `private_or_unit_purchase_flag`;
- `missing_price`;
- `zero_price`;
- `non_common_security_title`;
- `otc_symbol`;
- `late_report`.

## Trading-Time Semantics

Event time:

```text
t0 = max(SEC accepted_at_utc, repo first_seen_at_utc)
```

Entry time:

```text
tradable_at = first regular-hours minute >= t0 + 5 minutes
```

Rules:

- after-market filing: next regular session open + 5 minutes;
- pre-market filing: same session open + 5 minutes;
- in-session filing: next minute bar open after `t0 + 5 minutes`;
- if minute bars are missing, record blocker instead of silently falling back to
  impossible close-to-close timing.

Primary entry price:

```text
entry_px = minute_open at tradable_at
```

## Primary Label

Use a 5-trading-day net abnormal return as the primary label:

```text
ret_5d_net = log(exit_close_day5 / entry_px) - cost_model
abret_5d_net = stock_ret_5d_net - benchmark_ret_5d
```

MVP benchmark:

- IWM for micro/small-cap buckets;
- SPY for larger buckets;
- sector ETF later only if mapping is clean.

Secondary labels:

- `ret_1d_net`;
- `ret_10d_net`;
- `ret_20d_net`;
- `abret_1d_net`;
- `abret_10d_net`;
- `abret_20d_net`;
- max adverse excursion;
- max favorable excursion;
- entry gap from prior close;
- volume reaction.

## Cost And Liquidity

Eligibility:

- price at entry >= 2 dollars;
- ADV20 >= 500,000 dollars;
- common stock only;
- listed NYSE, Nasdaq, or NYSE American;
- exclude OTC;
- exclude ETFs/funds/SPAC units/warrants/preferreds;
- exclude same-day earnings;
- exclude contaminated major same-day news in the primary clean bucket.

Punitive MVP cost:

```text
round_trip_cost_bps = max(50 bps, conservative spread/liquidity proxy)
```

If quotes are available, use quoted spread. If only minute OHLCV is available,
mark the spread estimate as weak and use a conservative proxy.

## Event Strength Score

Do not train the first strength score. Make it deterministic:

```python
score = (
    log1p(total_dollar_value)
    + 0.75 * log1p(n_insiders_buying)
    + 0.50 * ceo_buy
    + 0.50 * cfo_buy
    + 0.25 * director_buy
    + 0.75 * log1p(cluster_30d_purchase_value)
    + 0.50 * percentile(purchase_value_to_market_cap)
    - 0.50 * earnings_nearby_flag
    - 0.50 * offering_nearby_flag
)
```

Bucket by event strength:

- Q1 weakest;
- Q2;
- Q3;
- Q4 strongest.

The first result should test whether Q4 beats Q1 and controls. It should not
depend on a black-box model.

## Controls

Run controls immediately:

1. Transaction-code control:
   - `A`, `M`, and `F` rows;
   - expected: P buys outperform mechanical acquisition/exercise/tax rows.
2. Sales placebo:
   - code `S`;
   - expected: sales do not produce the same positive 5d abnormal return.
3. Timestamp placebo:
   - shift event time back 20 trading days;
   - expected: effect disappears.
4. Matched ticker placebo:
   - same date bucket, sector/SIC bucket, market-cap bucket, liquidity bucket;
   - expected: real event beats matched fake event.
5. Strength monotonicity:
   - expected: Q4 beats Q1.

## Pass Criteria

Continue only if:

- eligible historical sample size > 300;
- Q4 mean `abret_5d_net` > +50 bps after 50 bps round-trip cost;
- Q4 median `abret_5d_net` > 0;
- Q4 minus Q1 > +75 bps;
- Q4 P-buy minus matched A/M/F control > +75 bps;
- result remains positive after excluding price < 2 dollars and ADV20 <
  500,000 dollars;
- top 5 events contribute < 35 percent of total return;
- source-quality failures <= 1 percent of eligible events.

## Kill Criteria

Kill as a trading candidate if:

- sample size < 300 eligible events;
- Q4 mean or median is <= 0 after costs;
- Q4 does not beat Q1;
- P buys do not beat A/M/F controls;
- timestamp placebo performs similarly;
- matched ticker placebo performs similarly;
- top 5 events dominate returns;
- effect exists only in untradeable names;
- entry gap captures most of the move before `tradable_at`;
- paper/shadow detection is too late relative to SEC accepted time.

Pause rather than kill if:

- effect exists only in tradable but very small-cap/liquidity-sensitive names and
  the current execution/liquidity model is too crude.

## Fixture Gate Before Backtesting

Do not trust any historical Form 4 result until the parser passes a small,
weird-fixture suite.

Minimum fixture set:

| Fixture | Accession | Required assertion |
| --- | --- | --- |
| Amazon / Indra Nooyi | `0001127602-20-013168` | Preserve many `P/A` rows, Decimal weighted-average prices, field-level footnotes, indirect ownership, and purchase-value aggregation. |
| Sinclair / David Smith | `0001250842-25-000026` | Accept Class A common stock and use filing acceptance time, not transaction date, as event time. |
| Tiptree / Goldman | `0000769993-15-000534` | Use archive CIK from index path, not issuer CIK; handle multiple reporting owners, mixed `P/S`, and timezone-suffixed transaction dates. |
| Immediatek / Radical / Mark Cuban | `0001209191-06-060213` | Parse but exclude from primary sample because `4/A` plus OTC; inventory supporting docs. |
| Super Micro / Sara Liu | `0001758554-19-000046` | Do not create a P-buy event from mechanical `M/F` rows; keep rows for negative controls. |
| Archimedes Tech SPAC | `0001437749-25-003569` | Exclude private/unit/SPAC sponsor language and ignore derivative warrant `P` rows. |
| Bioject / Logomasini | `0000810084-13-000003` | Exclude derivative warrants/preferred rows and zero-price rows. |
| Eledon private-placement-style filing | `0001593968-24-000563` | Trigger private-placement flag and exclude from strict open-market sample. |
| Ares / Ressler | `0001025978-25-000011` | Produce no buy event; preserve as sales-control fixture. |
| Very late filing | `0001528597-26-000004` | Compute `days_since_transaction` and exclude or bucket late reports separately. |

Minimum pass/fail tests:

- raw XML retrieval works from `archive_cik + accession + primaryDocument`;
- Tiptree fixture fails if issuer CIK is wrongly used for archive path;
- `4/A` filings parse but do not create primary signals;
- derivative `P` rows do not become common-stock buy events;
- private-placement/unit/SPAC rows are flagged and excluded;
- weighted-average footnotes are preserved;
- Decimal prices are not rounded;
- multiple reporting owners become an owner set;
- mixed `P/S` filings are flagged;
- transaction-date timezone suffixes parse without shifting the date.

## Implementation Sequence

### Pass 1: Direction And Contracts

- Update SSOT/autopilot roadmap to say event research is now the primary alpha
  roadmap.
- Keep generic ranker as sentinel infrastructure.
- Add provider/source contract entries for SEC Form 4 ownership filing XML.
- Add tests around Form 4 capability and timestamp contract.

### Pass 2: SEC Form 4 Raw Ingestion

- Extend SEC connector/runtime to discover Form 4 and 4/A filings from SEC
  full-index/daily-index manifests.
- Store `archive_cik` from the SEC index filename path separately from
  `issuer_cik` parsed from ownership XML.
- Enrich manifest rows from submissions JSON when available, but do not depend
  on submissions JSON for historical coverage or archive URL construction.
- Fetch and archive primary ownership XML by `archive_cik + accession +
  primaryDocument`.
- Fetch and archive complete `.txt` fallback when primary XML lookup fails.
- Extract ownership XML from SGML document blocks when needed.
- Store raw XML path/hash, complete `.txt` path/hash, accession, accepted time,
  first seen time, parser version, schema version, and source-quality flags.
- Never overwrite raw XML; revised/amended filings create linked versions.

### Pass 3: Parser And Event Tables

- Add `src/trademl/events/` and `src/trademl/entities/`.
- Parse ownership XML into `form4_transaction_raw`.
- Preserve Decimal prices/shares, field-level footnotes, multiple owner sets,
  supporting-document telemetry, and all exclusion flags.
- Aggregate eligible P buys into `insider_purchase_event`.
- Add CIK-to-PIT-ticker resolution using existing reference/security-master
  artifacts.
- Preserve ambiguous/unresolved mappings with reasons.
- Run the fixture gate before any backtest.

### Pass 4: Event Labels

- Resolve `tradable_at`.
- Resolve entry price from minute bars.
- Build 1d/5d/10d/20d labels, abnormal labels, MAE/MFE, entry gap, and volume
  reaction.
- Block rather than fake labels when minute data is missing.

### Pass 5: Event Study And Backtest

- Build event-study outputs by event strength bucket.
- Add naive trade simulation:
  - buy at `tradable_at`;
  - hold 5 trading days;
  - equal weight;
  - max 10 concurrent positions;
  - skip if already holding same ticker.
- Add capacity-aware tiny-trader simulation:
  - position notional = min(2,000 dollars, 1 percent of ADV20);
  - skip if position notional < 500 dollars.
- Add controls and outlier-removal reports.

### Pass 6: Shadow/Paper Operational Validation

- Detect new live Form 4 P events.
- Resolve ticker and eligibility.
- Write hypothetical order payloads only.
- Track latency, entry bar availability, source quality, and future labels.
- Do not submit orders.

## Result Packet

The first MVP report must contain:

- event class;
- historical period;
- eligible/excluded event counts and reasons;
- primary label definition;
- Q1/Q2/Q3/Q4 outcomes;
- Q4 mean/median/hit rate;
- Q4 minus Q1;
- Q4 minus A/M/F control;
- Q4 minus matched ticker placebo;
- cost sensitivity at 20/50/100 bps;
- top 5 contribution;
- worst 5 events;
- liquidity bucket performance;
- market-cap bucket performance;
- source-quality failure rate;
- continue/pause/kill decision.

## Questions To Ask GPT Pro If Needed

Ask for implementation-risk details, not more strategy philosophy:

```text
For a SEC Form 4 insider-buy MVP using EDGAR submissions plus filing XML,
what exact historical retrieval path should we use at scale, including URL
construction from accession/primaryDocument, amendment handling for 4/A, XML
field edge cases, and a minimum fixture set of weird filings we should test
before running the backtest?
```

## Boundary Conditions

- No live trading.
- No live Alpaca endpoint use.
- No options expression in MVP.
- No LLM buy/sell calls.
- No LLM materiality claims in MVP.
- No generic architecture churn as a substitute for event evidence.
- If the event fails, say so and move to the next source-first hypothesis.
