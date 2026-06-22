# GPT Pro Form 4 MVP Response

Captured: 2026-05-05

The following is copied from the ChatGPT thread context. It is preserved as a
source artifact for iterative review.

---

The smallest MVP I’d build is **Form 4 open-market insider purchases, long-only, common stock only**.

Not because it is the most original edge. It is not. Because it is the fastest clean test of your whole event-research stack: SEC ingestion, as-of timestamps, entity resolution, event schema, labels, negative controls, paper/shadow validation, and source-quality telemetry. It avoids options-fill fantasy, borrow constraints, LLM hallucination, court/entity mess, and government-contract materiality hell.

The falsifiable question:

> After a Form 4 open-market insider purchase is publicly available, do selected small/mid-cap common stocks show positive abnormal forward returns net of realistic costs, and does that effect survive negative controls?

SEC says Form 4 reports insider transactions, including amount purchased/sold and price per share, and is generally due within two business days after the transaction date; transaction code **P** means a purchase on an exchange or from another person. ([Investor][1])

---

# MVP choice

## Event type

**Insider open-market purchase event**

Strict definition:

```text
SEC Form 4
nonDerivativeTransaction only
transactionCode == "P"
securityTitle maps to common stock / ordinary shares
acquiredDisposedCode == "A"
transactionShares > 0
transactionPricePerShare > 0
issuer has tradable listed common equity
```

Do **not** start with sales. Sales are polluted by diversification, tax, 10b5-1 plans, liquidity needs, vesting, and automatic selling. Do **not** start with derivative exercises. They are mostly compensation mechanics.

This MVP is long-only:

```text
signal direction = bullish
trade expression = buy common stock
```

No options. No shorts. No LLM-generated trades.

---

# Why this is the right first MVP

Because it is boring enough to falsify.

A lot of public-event ideas fail because the data is ambiguous before the strategy is even tested. This one has structured XML, known transaction codes, known filing timestamps, and obvious negative controls.

It also tests a real small-trader hypothesis:

> Insider purchases in small, undercovered, low-attention names may be processed slowly enough that a small trader can capture post-filing drift without needing institutional speed or capacity.

The edge may be gone. That is fine. You want a fast answer.

---

# Exact data sources

## Primary event source

Use SEC ownership filings:

```text
Form types:
  - 4
  - 4/A only for correction handling, not fresh primary signal by default

Fields:
  - accession number
  - accepted datetime
  - filing date
  - period of report
  - issuer CIK
  - issuer trading symbol
  - reporting owner CIK
  - reporting owner name
  - reporting owner relationship flags
  - nonDerivativeTransaction rows
  - footnotes
```

Use your SEC collector against SEC submissions/filing docs. SEC’s `data.sec.gov` APIs expose public EDGAR data without authentication/API keys, including submissions history and XBRL data, with JSON structures updated throughout the day as submissions are disseminated. ([SEC][2])

## Market data

Use your existing minute data:

```text
minute OHLCV
daily OHLCV
corporate actions / split adjustment
delisting status if available
exchange/listing metadata
```

Minimum viable price source requirements:

```text
event timestamp → first tradable minute
entry price → minute open/close after latency buffer
exit prices → close+1, close+5, close+20
volume/ADV → from prior 20 trading days
volatility → prior 20 trading days
```

## Companyfacts / fundamentals

Use only simple filters and scaling variables:

```text
market cap proxy
latest shares outstanding if available
revenue
assets
cash
sector/SIC
filer size proxy
```

Do **not** over-model fundamentals in MVP.

## News data

Use news only as contamination / awareness telemetry:

```text
was there ticker-specific news within [-1d, +1d]?
was there earnings news?
was there M&A news?
was there offering/dilution news?
was there bankruptcy/delisting news?
```

For MVP, news should not generate alpha features. It should tell you whether the Form 4 event is isolated.

---

# Universe

Start narrow.

```text
US-listed common stocks only
exchanges: NYSE, Nasdaq, NYSE American
exclude OTC
exclude ADRs
exclude ETFs, funds, SPAC units, warrants, preferreds
exclude closed-end funds
exclude stocks with price < $2 at entry
exclude prior 20-day median dollar volume < $500k
exclude market cap > $20B for primary test
exclude same-day earnings announcement window
```

Why cap market cap? Because Apple-director-buying-50k is not your edge. The plausible edge is undercovered names.

Suggested primary bucket:

```text
$50M to $5B market cap
price >= $2
20-day median dollar volume >= $500k
```

Secondary buckets:

```text
micro: $25M–$300M
small: $300M–$2B
mid: $2B–$10B
large: >$10B
```

If the effect only exists in untradeable microcaps with trash liquidity, kill it as a trading idea but keep it as a data-quality benchmark.

---

# Entity resolution requirements

Keep this brutally simple.

## Required mappings

```text
issuer CIK → point-in-time ticker
ticker → primary common security
security → minute price series
issuer CIK → companyfacts entity
issuer CIK → sector/SIC
reporting owner CIK → insider identity
```

## Rules

Use **CIK as the primary company key**, not ticker.

Ticker is only the tradable instrument key after point-in-time resolution.

Handle these cases:

```text
same CIK, ticker changed:
  use ticker valid at filing timestamp

multiple securities for one issuer:
  choose primary common stock only

class A / class B:
  use class matching issuerTradingSymbol when available;
  otherwise choose highest-dollar-volume common share

amended Form 4:
  link 4/A to original accession if possible;
  do not create a new signal unless original was missing/wrong

duplicate filing rows:
  aggregate by accession + owner + transaction date + security title + code

multiple insiders same day:
  aggregate into one issuer-day event, but retain person-level rows
```

Do not infer subsidiaries. Do not fuzzy-match company names in this MVP. That is how you inject bad links.

---

# Event schema

Use two tables: raw transaction rows and issuer-level event rows.

## `form4_transaction_raw`

```python
{
  "accession": str,
  "form_type": "4" | "4/A",
  "issuer_cik": str,
  "issuer_name": str,
  "issuer_trading_symbol_raw": str,
  "reporting_owner_cik": str,
  "reporting_owner_name": str,

  "is_director": bool,
  "is_officer": bool,
  "is_ten_percent_owner": bool,
  "is_other": bool,
  "officer_title": str | None,

  "accepted_at_utc": datetime,
  "filed_date": date,
  "period_of_report": date | None,
  "transaction_date": date,

  "security_title": str,
  "transaction_code": str,
  "acquired_disposed": "A" | "D",
  "transaction_shares": float,
  "transaction_price": float,
  "transaction_value": float,

  "post_transaction_shares": float | None,
  "direct_or_indirect": "D" | "I" | None,
  "ownership_nature": str | None,

  "footnotes_text": str | None,
  "raw_xml_hash": str,
  "parser_version": str,
  "source_quality_flags": list[str]
}
```

## `insider_purchase_event`

Aggregate to issuer/event level.

```python
{
  "event_id": str,                 # issuer_cik + accepted_at + accession group
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
  "source_quality_score": float
}
```

---

# Event-strength score

Do not train this initially. Make it deterministic.

```python
score =
    log1p(total_dollar_value)
  + 0.75 * log1p(n_insiders_buying)
  + 0.50 * CEO_buy
  + 0.50 * CFO_buy
  + 0.25 * director_buy
  + 0.75 * log1p(cluster_30d_purchase_value)
  + 0.50 * percentile(purchase_value_to_market_cap)
  - 0.50 * earnings_nearby_flag
  - 0.50 * offering_nearby_flag
```

Then bucket it:

```text
Q1 weakest
Q2
Q3
Q4 strongest
```

Your first falsifiable result should not depend on a black-box model. First test whether stronger insider-buy events have stronger forward abnormal returns.

---

# Labels

Use several labels, but pre-register one primary label.

## Event time

```text
t0 = max(SEC accepted_at, repo first_seen_at)
```

For historical backtest:

```text
tradable_at = first regular-hours minute >= t0 + 5 minutes
```

If filed after market close:

```text
tradable_at = next session open + 5 minutes
```

If filed before market open:

```text
tradable_at = same session open + 5 minutes
```

If filed during regular session:

```text
tradable_at = next minute bar open after t0 + 5 minutes
```

Use `+5 minutes` even if your collector is faster. This avoids pretending you got instant EDGAR-to-order execution.

## Entry price

Primary:

```text
entry_px = minute_open at tradable_at
```

Conservative alternative:

```text
entry_px = max(minute_open, minute_vwap_proxy)
```

If you do not have true intraminute VWAP, do not pretend.

## Primary label

```text
ret_5d_net = log(exit_close_day5 / entry_px) - cost_model
```

Primary benchmark-adjusted label:

```text
abret_5d_net = stock_ret_5d_net - beta_adj_benchmark_ret_5d
```

Use simple beta adjustment:

```text
beta = rolling 252d daily beta vs SPY/IWM
benchmark = IWM for small/micro, SPY for large, or sector ETF if available
```

If sector ETF mapping is messy, use IWM/SPY only for MVP.

## Secondary labels

```text
ret_1d_net
ret_10d_net
ret_20d_net
abret_1d_net
abret_10d_net
abret_20d_net
max_adverse_excursion_5d
max_favorable_excursion_5d
close_to_close_gap_from_prior_close_to_entry
volume_reaction_1d
```

## Cost model

For MVP, use deliberately punitive costs.

For liquid names:

```text
round_trip_cost_bps = max(20 bps, 0.25 * median_intraday_high_low_spread_bps)
```

For less liquid names:

```text
round_trip_cost_bps = max(50 bps, 0.50 * median_intraday_high_low_spread_bps)
```

If you have quotes/spreads, use quoted spread. If you only have minute OHLCV, use a conservative proxy and clearly mark it as weak.

A strategy that dies under 50 bps round-trip in small caps probably was not real.

---

# Backtest rules

## Eligibility

Event is eligible only if:

```text
transaction_code == P
acquired_disposed == A
non-derivative common stock
transaction_value >= $25,000
accepted_at is present
mapped ticker is valid at event time
price >= $2
ADV20 >= $500,000
not OTC
not halted at entry
not same-day earnings
not same-day major news contamination unless testing contaminated bucket
```

## Signal formation

One issuer can generate multiple Form 4s close together. Aggregate to avoid double-counting.

```text
issuer-event window = 3 calendar days
```

If the same issuer has multiple Form 4 open-market buys within 3 calendar days:

```text
combine into one event
event timestamp = first accepted_at
total value = sum purchases known by each timestamp
```

For live/paper, only use information available as each filing arrives. For historical aggregation, do not let later Form 4s inflate the first signal’s strength.

Simpler implementation:

```text
create event at each Form 4 accepted_at
but suppress duplicate same-issuer entries for next 5 trading days
```

That is easier and safer.

## Portfolio simulation

For research MVP, run three versions.

### Version A: event-study only

No portfolio constraints.

```text
measure all eligible event outcomes
group by strength decile/bucket
compare to controls
```

### Version B: naive trade simulation

```text
buy at tradable_at
hold 5 trading days
equal weight each event
max 10 concurrent positions
skip if already holding same ticker
position size = min(1/N, 2% notional)
```

### Version C: capacity-aware tiny trader

```text
position_notional = min($2,000, 1% of ADV20)
hold 5 trading days
skip if position_notional < $500
```

This tells you whether the edge is only theoretical.

## Controls

You need controls immediately. Otherwise you will fool yourself.

### Control 1: transaction-code negative control

Same pipeline, same universe, but use:

```text
transactionCode in ["A", "M", "F"]
```

These are grants/exercises/tax-related mechanics, not open-market conviction buys. SEC’s code list distinguishes code P open-market/private purchase from award/exercise/tax-related codes like A, M, and F. ([SEC][3])

Expected result:

```text
P buys should outperform A/M/F controls.
```

If A/M/F performs the same, your “alpha” is probably just insider-filing attention, small-cap beta, or leakage.

### Control 2: sales placebo

Use code `S`, but do not necessarily expect symmetric negative returns.

Expected result:

```text
S should not produce the same positive 5d abnormal return as P.
```

If sales are also bullish, your model is picking post-filing drift/attention, not insider conviction.

### Control 3: timestamp placebo

Shift the event timestamp backward:

```text
t0_placebo = t0 - 20 trading days
```

Expected result:

```text
effect disappears.
```

If it survives, your effect is likely momentum/selection.

### Control 4: ticker placebo

Randomly assign each event to a matched ticker from same:

```text
market cap bucket
liquidity bucket
sector/SIC bucket
date bucket
```

Expected result:

```text
real event > matched fake event.
```

### Control 5: strength monotonicity

Bucket by deterministic event-strength score.

Expected result:

```text
Q4 strongest P-buy events > Q1 weakest P-buy events.
```

If there is no monotonicity, be suspicious.

---

# Falsifiable hypotheses

Pre-register these before running.

## H1: primary effect

```text
Top-quartile insider purchase events produce positive 5-trading-day abnormal net returns.
```

Pass threshold:

```text
mean abret_5d_net > +50 bps
median abret_5d_net > 0
bootstrap 95% CI for mean excludes 0
```

Do not require fantasy Sharpe. This is an event-study MVP.

## H2: monotonicity

```text
abret_5d_net increases from weak to strong event-strength buckets.
```

Pass threshold:

```text
Q4 - Q1 > +75 bps net
Q4 hit rate > Q1 hit rate by at least 5 percentage points
```

## H3: negative control separation

```text
P-buy events outperform A/M/F mechanical acquisition controls.
```

Pass threshold:

```text
Q4 P-buy abret_5d_net - matched A/M/F abret_5d_net > +75 bps
```

## H4: tradability

```text
Effect survives punitive costs and duplicate suppression.
```

Pass threshold:

```text
still positive after 50 bps round-trip cost
not driven by stocks below $2
not driven by ADV20 < $500k
not driven by top 5 events
```

---

# Kill criteria

Kill the idea as a trading candidate if any of these happen:

```text
sample size < 300 eligible historical events after filters
top-quartile mean abret_5d_net <= 0 after costs
median abret_5d_net <= 0
Q4 does not outperform Q1
P-buy events do not beat A/M/F controls
timestamp placebo performs similarly to real events
matched ticker placebo performs similarly to real events
top 5 events explain > 50% of total PnL
effect only exists below $2 price or below $500k ADV20
entry gap captures most of the move before tradable_at
paper/shadow signals arrive too late vs SEC accepted_at
source-quality failures exceed 1% of events
```

Pause, not kill, if:

```text
the edge exists only in microcaps but liquidity model is too crude
```

That means the next MVP is execution/liquidity validation, not more ML.

---

# What to leave out

Do **not** use the generic supervised ranker to pick trades yet.

Leave out:

```text
LLM sentiment
news embeddings
deep time-series models
options expression
cross-asset features
macro regime detector
portfolio optimizer
strategy synthesis
```

This MVP should answer one question:

> Does this one public event class have tradable post-publication drift?

If yes, then model it. If no, move on.

---

# What existing generic ranker work should remain

Keep the generic ranker, but demote it to **sentinel infrastructure**.

It should not drive entries. It should act as an overfitting/leakage detector.

## Keep these ranker components

### 1. Market-context sentinel

A generic ranker using only pre-event market features:

```text
prior returns
prior volatility
liquidity
market cap
sector
volume spike before filing
spread proxy
```

Purpose:

```text
Detect whether insider-buy “alpha” is just momentum, reversal, liquidity, or small-cap beta.
```

If the generic market-context ranker explains all the returns, the event is not adding much.

### 2. Source-quality sentinel

Keep your source-quality telemetry fully active:

```text
SEC accepted_at present?
first_seen_at present?
raw XML parse success?
transaction rows parsed?
CIK/ticker mapping confidence?
minute bars available?
corporate action adjustment valid?
duplicate detection triggered?
amendment handling valid?
```

This matters more than model quality.

### 3. Negative-control sentinel

Your existing negative controls should run automatically on every event-research batch:

```text
transaction-code controls
timestamp-shift controls
ticker-shuffle controls
matched-event controls
random-label controls
future-feature scan
```

This is the most valuable part of the current autopilot.

### 4. Paper/shadow sentinel

Keep paper/shadow mode, but only for operational validation:

```text
Would signal have fired?
When?
What ticker?
What entry bar?
Was the stock tradable?
Was there a price?
Was the event duplicate-suppressed?
Was there contamination news?
```

Do not use paper PnL as proof yet.

### 5. Backtest/live discrepancy sentinel

For each new live/shadow event, log:

```text
historical expected return bucket
actual signal latency
actual tradable_at
actual spread/liquidity proxy
actual next-day return
actual 5-day return
whether event matched schema
```

This lets you find silent production drift.

---

# Minimal implementation sequence

## Days 1–2: parser and event table

Build:

```text
Form 4 XML parser
transaction row table
issuer-level event aggregation
CIK/ticker join
common-stock filter
```

Manual QA:

```text
random 100 filings
all top 50 transaction_value events
all parser failures
all weird securityTitle values
```

## Days 3–4: labels

Build:

```text
tradable_at resolver
entry price resolver
1d/5d/10d/20d labels
abnormal return labels
MAE/MFE labels
cost model
```

Sanity checks:

```text
no label uses prices before tradable_at
after-hours filings enter next session
pre-market filings enter open+5m
in-session filings enter t0+5m
```

## Days 5–7: event-study and controls

Run:

```text
all P buys
top quartile strength
A/M/F controls
S controls
timestamp placebo
matched ticker placebo
```

Output one table:

```text
N
mean net abret
median net abret
hit rate
bootstrap CI
Q4-Q1 spread
top-5 contribution
worst 5 events
liquidity bucket performance
market-cap bucket performance
```

## Week 2: paper/shadow deployment

Turn on live shadow:

```text
detect Form 4 P event
resolve ticker
score event
decide eligible/ineligible
write hypothetical order
track forward labels
```

No money yet unless the historical result is strong and execution assumptions are sane.

---

# Exact “result packet” after 1–2 weeks

The MVP should produce a result like this:

```text
Event class:
  Form 4 open-market insider purchases

Historical period:
  YYYY-MM-DD to YYYY-MM-DD

Eligible events:
  N total
  N top quartile
  N excluded and why

Primary label:
  5-trading-day abnormal net return from tradable_at

Result:
  Q4 mean abret_5d_net
  Q4 median abret_5d_net
  Q4 hit rate
  Q4 - Q1
  Q4 - A/M/F matched control
  Q4 - ticker placebo
  cost sensitivity: 20/50/100 bps

Failure analysis:
  return captured before tradable_at?
  driven by illiquid names?
  driven by microcaps?
  driven by outliers?
  contaminated by earnings/news?
  does generic ranker explain it?

Decision:
  continue / pause / kill
```

That is falsifiable.

---

# What would count as “continue”

Continue only if you get something like:

```text
N eligible > 300
Q4 mean abret_5d_net > +50 bps after 50 bps cost
Q4 median > 0
Q4 > Q1 by +75 bps
Q4 > matched controls by +75 bps
effect still positive excluding microcaps below $500k ADV
top 5 events contribute < 35% of total return
```

Then the next version can add:

```text
insider role modeling
purchase size relative to wealth/holdings
repeat buyer history
cluster detection
company distress filters
news contamination modeling
filing-time latency model
position sizing
```

---

# What would count as “kill”

Kill it fast if:

```text
event effect is gone after realistic entry timing
signal is just the entry gap
P buys do not beat A/M/F controls
generic market-context ranker performs as well
effect is only in impossible-liquidity names
returns come from one biotech/meme outlier
paper/shadow pipeline misses or delays events
```

The most likely failure mode is that the apparent alpha exists only in historical close-to-close labels and disappears when you enter after the Form 4 is actually public.

---

# Bottom line

Build **Form 4 open-market insider-buy event research** first.

It is the cleanest 1–2 week falsification target because it is:

```text
structured
public
timestamped
long-only
cheap to parse
easy to control
easy to label
compatible with your current SEC/news/minute stack
```

The generic ranker stays, but only as a sentinel: leakage alarm, baseline comparator, source-quality monitor, and negative-control runner. The event-specific MVP should be dumb, explicit, and hard to fool.

[1]: https://www.investor.gov/introduction-investing/general-resources/news-alerts/alerts-bulletins/investor-bulletins-69 "Updated Investor Bulletin: Insider Transactions and Forms 3, 4, and 5 | Investor.gov"
[2]: https://www.sec.gov/search-filings/edgar-application-programming-interfaces "SEC.gov | EDGAR Application Programming Interfaces (APIs)"
[3]: https://www.sec.gov/edgar/searchedgar/ownershipformcodes.html "Ownership Form Codes"

