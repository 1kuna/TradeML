# Public Event Research Pivot SSOT

Last updated: 2026-05-20

Current control note, 2026-05-20:

- Active path-forward and restart criteria now live in
  [TradeML Path Forward SSOT](PATH_FORWARD_SSOT.md).
- This document remains the historical pivot detail and source-first policy
  reference. For current next steps, killed candidates, and runtime restart
  criteria, `PATH_FORWARD_SSOT.md` wins.

## Status

This is the historical source-of-truth document for TradeML's research-direction
pivot. It is subordinate to `PATH_FORWARD_SSOT.md` for current restart and
next-gate decisions.

Repo-wide implementation details still live in `SSOT.md`, `DEV_GUIDE.md`, and
the code. This document owns the research thesis and roadmap direction for the
next alpha-search phase. If implementation work creates a conflict between this
document and `SSOT.md`, update the repo-wide SSOT in the same pass rather than
letting both drift.

Child MVPs under this pivot:

- [Form 4 Insider Purchase Event MVP](form4_insider_purchase_event_mvp_direction.md)
- [SEC 8-K Item Event MVP](sec_8k_item_event_mvp_direction.md)

Current child-MVP status:

- The first naive Form 4 open-market insider-buy rule has been implemented and
  evaluated.
- The bounded Form 4 rework gate has also been evaluated.
- Verdict: `FORM4_KILLED_BASELINE_COMPLETE`, not promotable.
- Keep the source-first event pipeline work; do not use the naive Form 4 rule as
  a paper/live entry engine.
- Do not spend more automated search budget on Form 4 unless a new documented
  hypothesis changes the event definition, data revision, or controls.
- The deterministic SEC 8-K item-family MVP has been implemented and evaluated
  through a 1,000-filing April 2025 slice.
- Verdict: `BROAD_SEC8K_ITEM_FAMILIES_KILLED`, not promotable, with
  `move_forward=false`.
- Keep the SEC/event spine; do not spend more automated search budget on broad
  8-K item families without a narrower documented hypothesis.
- The next public-event continuation should be either a narrower 8-K
  exhibit/materiality hypothesis or a different source-first event class.

The raw GPT Pro / GPT-5.5 source material behind this pivot is preserved here:

- [Source Prompt Archive](source_prompts/README.md)

## Decision

TradeML should pivot from generic architecture search over daily cross-sectional
price prediction toward a source-first public-event research foundry.

The current Pi/Mac/NAS stack should stay in place. It is valuable
infrastructure. The pivot is about what the system tries to discover, not about
throwing away the collection, validation, telemetry, or research-control plane.

The new research target is:

> Find narrow, lawful, public-information event classes where the information is
> structured enough to audit, annoying enough to be under-processed, small
> enough to be ignored by large funds, and tradable after realistic timing,
> spread, slippage, liquidity, and risk constraints.

## Why We Are Pivoting

The current supervised cross-sectional ranker is an important baseline, but it
is probably not the best solo-trader battlefield by itself. Major liquid markets
and generic public-news signals are heavily competed. If an edge exists for this
project, it is more likely to come from:

- as-of-correct public source ingestion;
- entity linking;
- structured event extraction;
- materiality and novelty scoring;
- first-tradable-time labeling;
- realistic liquidity/cost filters;
- hard negative controls;
- disciplined event-level backtesting;
- paper/shadow live-vs-backtest validation.

The edge is not expected to come from a more exotic model guessing tomorrow's
return from generic features.

## Small-Trader Edge Thesis

TradeML should not try to compete with institutional firms on easy-to-parse,
high-capacity, liquid public signals. The project should intentionally search
where a small trader might have an advantage:

- the trade is too small or illiquid for large funds to size;
- the public data is annoying, fragmented, or document-heavy;
- entity linking is hard;
- the event is material but not broadly digested;
- market awareness is delayed;
- execution is possible with conservative liquidity filters;
- the edge survives realistic spread, slippage, and risk limits.

The durable edge, if one exists, is expected to come from the combination of
ugly public data, better entity linking, source-quality discipline, event
normalization, low-capacity small-name focus, and fast kill criteria. It is not
expected to come from a monolithic financial world model.

## Architecture Stance

Do not build one all-market AI trader. Build shared infrastructure plus narrow
strategy modules.

Preferred shape:

```text
raw public data
  -> append-only as-of data lake
  -> entity resolution + event extraction
  -> strategy-specific feature stores
  -> hypothesis testing / backtest engine
  -> paper/shadow operations
  -> risk engine + execution logger
  -> PnL attribution + kill criteria
```

The project should prefer:

- separate strategy-specific models over a single latent world model;
- deterministic validators, scoring, and gates after semantic extraction;
- simple rules, linear/logistic models, gradient boosting, and calibrated
  baselines before deep learning;
- LLMs for extraction, classification, diffing, and entity-disambiguation
  suggestions only;
- constraint-first risk controls over fragile expected-return optimizers.

LLMs must not allocate capital, directly choose buy/sell decisions, invent
materiality claims, or explain away failed evidence.

## Semantic Extraction Policy

Regex, keyword rules, and other deterministic string matching are allowed only
for stable source structure:

- SEC accession metadata;
- SGML/XML/HTML tags and document tables;
- item-section boundaries such as literal `Item 1.01` headings;
- timestamps, archive paths, hashes, and source identifiers;
- exact schema fields from structured formats.

They are not allowed to classify or extract narrative meaning from news,
filings, exhibits, press releases, contracts, articles, local reports, or
regulatory/legal text. Do not use regex or keyword heuristics to decide:

- sentiment;
- materiality;
- dilution or financing toxicity;
- contract awards;
- auditor trouble;
- customer loss;
- default/covenant/liquidity stress;
- guidance changes;
- M&A or strategic transaction substance;
- legal/regulatory outcome meaning;
- any other economic event interpretation.

Semantic classification/extraction must use an LLM or purpose-built model into a
strict schema. The deterministic layer then validates the model output:

- required fields present;
- event type from an allowed taxonomy;
- evidence snippets and source offsets present;
- numeric fields supported by the cited text;
- timestamps and PIT availability preserved;
- categorical certainty and ambiguity recorded, with no numeric confidence
  scores;
- impossible or unsupported claims rejected;
- ambiguous/insufficient cases marked `needs_review` or `no_event`, never silently
promoted.

This policy applies to news, 8-K exhibits, company IR, federal contracts, court
records, regulatory releases, and every other narrative public source. A
header-only or regex-only event classifier may be used as a negative-control
baseline, but not as the alpha candidate for semantic events.

Current semantic gate:

- `trademl research sec-event-semantic-gate`;
- `trademl research sec-event-semantic-classify`;
- `trademl research sec-event-semantic-study`;
- default local model candidate: `qwen3.5-9b-mlx` through LM Studio;
- artifact:
  `control/cluster/state/research/sec_event_semantic_fixture_gate/latest.json`;
- report: `reports/research/sec_event_semantic_fixture_gate/latest.md`;
- semantic MVP artifacts:
  `control/cluster/state/research/sec_event_semantic_classification/latest.json`,
  `control/cluster/state/research/sec_event_semantic_study/latest.json`,
  and `reports/research/sec_event_semantic_study/latest.md`.

This gate enforces schema validity, categorical certainty only, exact evidence
quotes, and exact materiality evidence for contract awards. Auxiliary extracted
fields are recorded for inspection, but non-exact auxiliary values are warnings
and cannot feed downstream numeric features until a stricter extraction gate
passes. Passing this gate means the model is eligible for a bounded semantic MVP
fixture/canary; it does not make any event class paper/live-ready.

The SEC 8-K semantic MVP classifies archived 8-K item and exhibit snippets,
promotes only validated economic event classes, labels them with the existing
first-tradable-time market pipeline, and decides
`CONTINUE_SEMANTIC_8K`, `MORE_DATA_REQUIRED`, `SEMANTIC_8K_KILLED`, or
`BLOCKED_DATA_COVERAGE`. Paper/live remains blocked in every verdict.

## What Stays

The existing system remains the foundation:

- Raspberry Pi data node:
  - continuous public/market data capture;
  - provider capability registry;
  - planner-native scheduling;
  - ingestion ledger;
  - archive write telemetry;
  - source availability and data-quality artifacts;
  - saturation/controller telemetry.
- NAS:
  - raw append-only archives;
  - curated market data;
  - modeling artifacts;
  - research and control-plane state.
- Mac Mini research loop:
  - experiment registry;
  - canary/preflight machinery;
  - strict objective gates;
  - negative controls;
  - candidate evidence;
  - paper/shadow artifacts;
  - fleet watchdog and Codex issue feed.
- Generic daily ranker:
  - stays as a sentinel and comparator;
  - stays useful for leakage detection, market-context controls, and baseline
    performance;
  - should not be the primary entry engine for the first event MVP.

## What Changes

The primary research unit changes from:

```text
feature matrix + model architecture + cross-sectional daily label
```

to:

```text
public source record -> parsed event -> entity-linked tradable instrument ->
first-tradable-time label -> event-study/backtest -> paper/shadow evidence
```

This means new work should prioritize:

- raw source completeness and first-seen timestamps;
- event schemas;
- entity resolution;
- LLM/model semantic extraction with schema validation;
- deterministic evidence validation and event-strength scores after extraction;
- event-level research datasets;
- controls and kill criteria;
- operational shadow validation.

Model complexity is intentionally secondary.

## Research Methodology

Every event hypothesis must be written before testing:

```text
Who is mispricing what?
Why does the mispricing exist?
Why might it persist?
Who is on the other side?
Why can a small trader trade it while a large fund may ignore it?
What destroys the edge?
What is the expected holding period?
What are the costs?
What data would falsify it?
What live result would kill it?
```

Every event MVP must produce a result packet:

- event class;
- historical period;
- raw source coverage;
- eligible event count;
- excluded event count and reasons;
- primary label;
- entry timing rule;
- cost assumptions;
- liquidity assumptions;
- main effect result;
- negative-control results;
- outlier dependence;
- liquidity bucket performance;
- market-cap bucket performance;
- paper/shadow readiness;
- continue / pause / kill decision.

## Required Event Pipeline Objects

### Raw Source Records

Every raw source record should preserve:

- source id;
- source name;
- source URL when applicable;
- raw payload or raw payload path;
- raw hash;
- source timestamp when present;
- vendor timestamp when present;
- ingestion timestamp;
- first-seen timestamp;
- revision/amendment timestamp when applicable;
- parser version;
- schema version;
- collection job id;
- provider/feed/plan metadata when known;
- error metadata for failed fetches.

Raw source records must be append-only. Revisions create new versions linked to
prior versions.

### Entity Records

The first entity layer should be simple and auditable:

- issuer CIK;
- ticker history;
- primary common security;
- exchange/listing status;
- company name and aliases;
- sector/SIC/industry when available;
- reporting-owner identities where relevant;
- source provenance;
- manual overrides with versioning.

CIK should be the preferred company key when SEC data is involved. Ticker is the
tradable instrument key after point-in-time resolution.

### Event Records

Every event should be auditable back to source:

- event id;
- event type/subtype;
- entity id or issuer CIK;
- ticker/security id;
- source accession or source id;
- event timestamp;
- first-seen timestamp;
- tradable-at timestamp;
- model-extracted or structured-source parsed fields;
- model-extracted and deterministically validated strength/materiality/novelty
  fields;
- source-quality flags;
- eligibility status;
- exclusion reasons;
- parser/extractor version;
- categorical certainty/ambiguity and deterministic parse-quality flags.

## First MVP

The first MVP is SEC Form 4 open-market insider purchases:

- Form 4;
- non-derivative transaction rows;
- transaction code `P`;
- acquired code `A`;
- common stock only;
- long-only event study and paper/shadow validation;
- no options;
- no LLM-generated trade decisions.

See:

- [Form 4 Insider Purchase Event MVP](form4_insider_purchase_event_mvp_direction.md)

## Initial Event Backlog

After the Form 4 MVP is falsified or promoted to further research, candidate
next tracks are:

1. SEC 8-K / exhibit novelty events.
2. Company IR press-release materiality events.
3. Federal contract award materiality events.
4. Undercovered PEAD / revision drift.
5. Earnings options implied-move mispricing, only after equities event research
   and execution assumptions are credible.

Do not start with options, raw text models, transformers, RL, GNNs, or broad
strategy synthesis.

## Source-First Feed Backlog

This catalog exists so future implementation passes do not lose useful public
source lanes. It is a backlog, not permission to enable everything at once. Each
source must get its own adapter, source contract, timestamp semantics,
rate-limit/terms check, fixtures, quality checks, and result packet before it is
allowed to affect research.

### Tier 0: Existing / First-Path Sources

These are closest to the current TradeML stack and should be hardened first:

- SEC EDGAR submissions API:
  - filing history;
  - accepted timestamps;
  - accession/primary document metadata;
  - 8-K, 10-K, 10-Q, Form 4, and 4/A ownership forms;
  - amendments/revisions linked to original accessions.
- SEC filing documents and exhibits:
  - ownership XML;
  - 8-K exhibits;
  - text/html filing documents;
  - exhibit-level raw hash and parser version.
- SEC company facts / XBRL:
  - revenue/assets/cash/share-count scale variables;
  - simple filters and materiality denominators;
  - not a black-box alpha source in early MVPs.
- Company IR press releases and RSS/pages:
  - source-of-truth company announcements;
  - materiality/event source after SEC MVPs are working.
- Existing market-data/provider archives:
  - daily OHLCV;
  - minute bars;
  - trades/quotes where available;
  - corporate actions;
  - listing/security-master metadata;
  - news as awareness/contamination before alpha features.

### Tier 1: High-Value Public Event Sources

These are strong candidates after SEC/Form 4 proves the event spine:

- USAspending API:
  - federal awards;
  - recipient names;
  - agencies;
  - award values;
  - award/modification distinctions;
  - materiality relative to revenue/market cap.
- SAM.gov:
  - opportunities;
  - awards/solicitations where accessible;
  - contract vehicle context.
- Federal Register API:
  - agency rules;
  - proposed/final rule events;
  - comment deadlines;
  - regulatory calendars.
- FDA RSS / openFDA:
  - approvals;
  - warning letters;
  - recalls;
  - enforcement events;
  - biotech/medtech-specific event tracks.
- ClinicalTrials.gov API:
  - study status changes;
  - completion/readout timing;
  - sponsor/condition/drug mappings;
  - trial-event calendars.
- CourtListener and RECAP where usable:
  - opinions;
  - docket events;
  - legal/regulatory case updates;
  - patent/product-liability/antitrust/bankruptcy event tracks.
- USPTO / PatentsView:
  - patents;
  - assignees;
  - pre-grant publications;
  - litigation/context features;
  - slower thematic/event context rather than first short-horizon alpha.

### Tier 2: Awareness / Supplementary Sources

These are useful mainly for awareness, contamination, novelty, and coverage
checks until a specific hypothesis justifies deeper use:

- General news APIs already available through current providers:
  - Alpaca News;
  - Finnhub company news;
  - Alpha Vantage NEWS_SENTIMENT;
  - FMP stock/press news;
  - any paid/entitlement-blocked lane must stay audit-only until approved.
- Local/regional public news where accessible and terms allow:
  - plant openings/closures;
  - layoffs;
  - permits;
  - municipal contracts;
  - environmental incidents.
- Official agency/company social feeds only when API access and terms allow:
  - agency X/Twitter or equivalent official channels;
  - company official accounts;
  - never scrape in a way that violates terms.
- App/web/social trend data:
  - backlog only;
  - high narrative-overfit risk;
  - must be treated as awareness/context unless a separate MVP proves value.

### Provider Adapter Families

As sources are added, keep adapters grouped by what they produce:

- `FilingProvider`;
- `OwnershipFilingProvider`;
- `CompanyPressReleaseProvider`;
- `AwardProvider`;
- `LegalEventProvider`;
- `RegulatoryProvider`;
- `ClinicalTrialProvider`;
- `PatentProvider`;
- `NewsSourceProvider`;
- `MarketDataProvider`;
- `EntityResolverProvider`.

Every provider adapter must expose:

- capabilities;
- coverage;
- current entitlement/free-vs-paid limitation;
- rate limits;
- auth requirements;
- timestamp semantics;
- data-delay semantics;
- supported asset classes/entities;
- error handling behavior;
- parser/schema version;
- test fixtures.

### Capture Rule

The collector should capture raw public source data before the research loop can
use it, but raw capture alone is not enough. A source is research-eligible only
after it has:

- append-only raw storage;
- source hash;
- first-seen timestamp;
- parser version;
- source-quality checks;
- entity-resolution path;
- event schema or feature contract;
- leakage/control tests;
- explicit inclusion in a hypothesis/result packet.

## First Three Research Tracks

The near-term research roadmap should stay narrow:

1. Public-event drift in undercovered listed equities.
   - SEC filings, company IR, government contracts, legal/regulatory events,
     and other public sources.
   - Equity expression first.
   - Options only after equities event evidence is credible.
2. PEAD and analyst/revision drift in undercovered names.
   - Earnings surprise, guidance surprise, estimate dispersion, and revision
     clusters.
   - Days-to-weeks holding periods.
   - Strict controls for hidden beta and liquidity.
3. Earnings options implied-volatility mispricing.
   - Deferred until equities event research and execution accounting are sane.
   - Defined-risk structures only.
   - No market orders, no naked short options, and no midpoint-fill fantasy.

Everything else is backlog until one small, boring, measurable edge survives.

## Validation Rules

Every event MVP must check:

- lookahead bias;
- first-seen timestamp correctness;
- SEC/vendor timestamp semantics;
- source revisions and amendments;
- point-in-time ticker mapping;
- survivorship and delisting effects;
- corporate actions;
- realistic first-tradable entry;
- spread/slippage/cost assumptions;
- liquidity and capacity limits;
- event clustering and duplicate suppression;
- matched placebo events;
- timestamp-shift placebo;
- ticker-shuffle or matched-ticker placebo;
- outlier dependence;
- regime/period dependence;
- paper/shadow live-vs-backtest discrepancy.

No strategy should continue if it only works with impossible fills, stale ticker
links, untradeable liquidity, future data, or a few outlier events.

## Self-Deception Checklist

TradeML should assume automated research will fool itself unless these failure
modes are actively tested:

- lookahead bias;
- survivorship bias;
- options midpoint-fill fantasy;
- selection bias from too many trials;
- regime luck;
- hidden correlation to small-cap beta, liquidity, credit, volatility, or theme
  exposure;
- miscalibrated model confidence;
- ignoring capacity;
- borrow/financing blindness for any future short strategy;
- narrative overfitting from LLM-generated explanations;
- confusing public availability with market neglect;
- mistaking volatility or outlier wins for edge.

The experiment registry, candidate evidence, negative controls, event result
packets, paper/shadow logs, and Codex issue feed should exist to keep these
failure modes visible.

## Options And Execution Stance

Options are not a first MVP. They may become useful later only when an event
edge has credible equities evidence and the option chain is liquid enough to
trade honestly.

Future options rules:

- defined-risk only;
- no naked short gamma;
- no market orders;
- reject contracts with bad spread, open interest, volume, or stale IV;
- model assignment/dividend/exercise risks before trade approval;
- record NBBO, mid, submitted limit, fill, cancel/replace path, exit quote, and
  realized slippage;
- treat paper options fills as operational telemetry, not proof of executable
  edge.

Execution logging matters before live trading. Even paper/shadow signals should
record signal id, strategy id, event id, intended instrument, decision time,
quote/price data, hypothetical order terms, exit reason, and realized label/PnL
when mature.

## Role Of The Generic Ranker

The generic ranker stays, but it is demoted from primary alpha engine to
sentinel infrastructure:

- market-context baseline;
- leakage detector;
- source-quality monitor;
- negative-control runner;
- paper/shadow operational comparator;
- evidence that an event adds value beyond momentum, reversal, liquidity,
  sector, and small-cap beta.

If a generic market-context baseline explains all event returns, the event edge
is probably not real.

## Non-Goals

- No live trading.
- No live Alpaca endpoint use.
- No options in the first MVP.
- No LLM direct buy/sell decisions.
- No LLM position sizing.
- No model architecture churn as a substitute for event evidence.
- No HFT/order-book market making.
- No OTC-first strategy.
- No paid-data assumption unless explicitly approved later.

## Definition Of Done For The Pivot Foundation

The pivot foundation is ready when TradeML can:

- ingest one public event source append-only;
- segment stable source structure deterministically;
- classify/extract narrative semantics with an LLM or purpose-built model into a
  validated schema;
- link events to tradable instruments with evidence;
- build event labels from first tradable time;
- run event-study and simple trade simulations;
- run mandatory controls;
- emit a continue/pause/kill result packet;
- write paper/shadow signal artifacts without live orders;
- surface source-quality and research blockers in the Codex issue feed.

The first concrete test of this foundation is the Form 4 insider-purchase MVP.
