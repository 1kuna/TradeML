# GPT Pro Original Brutal Take

Captured: 2026-05-05

The following is copied from the ChatGPT thread context. It is preserved as a
source artifact for iterative review.

---

## Brutal take

Your hypothesis is directionally right but way too “institutional AI lab” and not enough “find one ugly edge that survives costs.” The dangerous part is the fantasy of a unified financial world model. The useful part is the control-system framing: point-in-time data, narrow strategy families, execution accounting, risk limits, drift monitoring, and kill criteria.

Markets are **extremely efficient where the same public signal is easy to parse, high-capacity, liquid, and monetizable by large firms**: mega-cap news, index futures, liquid ETFs, most macro headlines, obvious earnings beats, SPX direction, and liquid options surface arbitrage. Markets are **less efficient where the signal is public but annoying, low-capacity, entity-linkage-heavy, document-heavy, legally fragmented, liquidity-constrained, or behaviorally distorted**.

The options market is now enormous and hypercompetitive: OCC reported **15.207B total listed options contracts in 2025**, up **24.4%** from 2024, with 2025 options ADV around **60.6M contracts** by year-end stats; Cboe says 2025 was the sixth straight record year, with SPX 0DTE averaging **2.3M contracts daily** and representing **59%** of SPX volume. ([OCC][1]) SEC staff also notes that options breadth and complexity exploded: unique underliers rose **144%** since 2012, listed option securities increased more than sevenfold, OPRA messages peaked at **247B/day** in early 2025, top 10 underliers account for **31%** of volume, and 0DTE is over **28%** of total options volume. ([SEC][2]) That means “AI trading options” is not a greenfield opportunity. It is a knife fight where the only sane target is narrow, measurable, low-capacity mispricing.

---

# 1. Harsh critique of the proposed architecture

## The biggest flaw: it is too broad to falsify

The architecture tries to ingest everything, model everything, and trade everything. That is how you build an impressive demo and a useless trading system.

A profitable small-trader system should not start with:

> “Let’s model equities, ETFs, options, futures, FX, rates, vol, social media, filings, court records, weather, shipping, and order books.”

It should start with:

> “Here is one repeatable market behavior, here is why it exists, here is who is on the other side, here is why they do not arbitrage it away, here is how much survives execution.”

Your architecture has too many degrees of freedom. A system that can explain every result after the fact will eventually “discover” fake alpha.

## “Financial world model” is mostly a trap

A common latent market/world state sounds elegant, but in practice it creates a giant uninspectable dependency graph. Markets do not have one stable latent state. They have many partial, conflicting states:

* fundamental state,
* liquidity state,
* dealer inventory state,
* factor/risk state,
* sentiment state,
* macro-policy state,
* event-risk state,
* positioning state.

These interact, but forcing them into one shared representation will often reduce interpretability and increase model risk. The system will look smarter while becoming harder to debug.

The more realistic approach is **separate strategy-specific models with shared infrastructure**, not one grand world model.

## Time-series foundation model: likely overkill

A foundation model for returns/vol/correlation may help with representation learning, but it will not magically create tradable alpha. Liquid returns are low signal-to-noise, non-stationary, reflexive, and adversarial.

Use time-series models for:

* volatility forecasting,
* liquidity forecasting,
* regime clustering,
* risk estimation,
* execution/slippage prediction.

Do not use them as primary “which stock goes up” machines.

## LLM/news model: useful, but not as an alpha oracle

LLMs are useful for:

* extracting structured events from messy text,
* classifying filings,
* diffing risk factors,
* mapping entities,
* detecting novelty,
* summarizing legal/regulatory developments.

They are dangerous for:

* sentiment scores as direct trading signals,
* hallucinated causal explanations,
* “this sounds bullish” trade generation,
* overfitting narratives.

The LLM should produce **auditable structured event objects**, not trades.

Example:

```text
event_type: federal_contract_award
company: X
award_value: $182M
award_type: new / modification / IDIQ / option exercise
materiality_to_revenue: 18%
source_timestamp: 2026-02-03 14:22:10 ET
market_awareness_proxy: no major newswire within 2h
eligible_strategy: small-cap public-event drift
```

That is useful. “Bullish contract news, buy stock” is trash.

## GNN/entity graph: useful plumbing, weak alpha

An entity graph is valuable for mapping:

* ticker ↔ CIK ↔ subsidiaries,
* suppliers/customers,
* competitors,
* ETFs/indexes,
* court parties,
* government-award recipients,
* patents,
* products,
* executives.

But a graph neural network is not automatically better than a clean entity database plus simple graph features. The edge is usually in **entity resolution**, not in fancy graph embeddings.

## Causal inference layer: mostly marketing unless narrowed

“Distinguish X caused Y from X correlated with Y” is almost impossible as a general market module. Markets are adaptive, confounded, and full of simultaneous information shocks.

Better version:

* use causal thinking during hypothesis design,
* run placebo tests,
* compare treated vs matched controls,
* test event windows,
* use difference-in-differences where plausible,
* pre-register hypotheses,
* avoid pretending observational ML has identified causality.

Causal inference should be a **research discipline**, not a magical layer.

## Regime detector: usually late

Regime detectors tend to identify regime change after the damage. The useful version is not “the AI detects regimes.” It is:

* rolling live-vs-backtest discrepancy,
* drawdown by strategy,
* slippage drift,
* hit-rate/payoff drift,
* feature distribution drift,
* signal calibration drift,
* kill switches.

Do not ask a regime detector, “Should I trade?” Ask, “Has this strategy’s live behavior stopped matching its known distribution?”

## Bayesian uncertainty layer: good, but not enough

Probabilistic outputs are correct. But Bayesian-looking uncertainty is often fake precision. Your model may say:

```text
expected return: 1.8%
confidence interval: [-0.3%, +4.2%]
```

But the real uncertainty includes:

* stale data,
* wrong timestamp,
* liquidity disappearing,
* hidden correlation,
* changed market participant behavior,
* broker routing differences,
* option fill impossibility,
* borrow recall,
* overnight gap,
* event calendar error.

Use probabilistic outputs, but assume the tails are under-modeled.

## Strategy synthesis layer is p-hacking at scale

Automated strategy discovery is where systems go to die. It will search thousands of combinations and find beautiful garbage.

Bailey and López de Prado’s Deflated Sharpe Ratio work exists because modern researchers can test huge numbers of strategies and keep the lucky ones; they specifically warn that backtest optimizers inflate performance through multiple testing and non-normal returns. ([SSRN][3]) Their Probability of Backtest Overfitting framework also argues ordinary holdout methods can be unreliable for investment backtests. ([SSRN][4])

So the strategy layer should not “synthesize strategies.” It should:

* test human-legible hypotheses,
* record every trial,
* penalize multiple testing,
* require economic rationale,
* require live micro-validation.

## Portfolio optimizer: fragile because expected returns are fragile

Mean-variance optimization on noisy alpha estimates is a footgun. Kelly is also a footgun if your edge estimate is wrong, which it probably is.

For a small trader, use:

* hard exposure limits,
* simple risk budgets,
* scenario stress,
* max loss per trade,
* max loss per strategy,
* max correlation cluster,
* liquidity constraints,
* capped convexity where possible,
* volatility-adjusted sizing.

Optimization should be **constraint-first**, not “maximize expected utility from garbage inputs.”

## Execution engine: retail reality is uglier

“Smart order routing, venue selection, dark/lit decisioning, queue-position estimation” is mostly institutional infrastructure. A small trader using retail/prosumer broker APIs often cannot control the routing stack meaningfully.

Also, options execution is brutal. The SEC staff notes retail options execution is dominated by consolidator/PFOF economics, with top consolidators controlling most individual-customer order flow; it also notes that electronic limit order books often dominate retail executions with minimal price improvement, while auctions can help under wide spreads. ([SEC][2])

For a small trader, execution edge is more likely:

* never using market orders in options,
* posting near-mid limits,
* measuring fill quality,
* avoiding contracts with fake tight markets,
* rejecting trades where spread consumes edge,
* using defined-risk structures,
* choosing contracts where you can actually exit.

## Missing pieces

You need these before “world model” anything:

* point-in-time security master,
* ticker/CIK/CUSIP/option-root mapping,
* corporate actions,
* delisted symbols,
* point-in-time index/ETF membership,
* actual source arrival timestamps,
* vendor publication timestamps,
* broker quote-at-order timestamps,
* option symbology handling,
* dividend/exercise/assignment model,
* borrow availability and borrow cost history,
* spread/volume/open-interest liquidity filters,
* experiment registry,
* trial-count tracking,
* live-vs-backtest reconciliation,
* fill simulator calibrated to your own orders.

As-of correctness is not a feature. It is the foundation. Without it, everything else is cosplay.

---

# 2. Architecture I would propose instead

Do **not** build an all-market AI trader. Build a **public-information alpha research foundry** with a few production strategy modules.

## Core architecture

```text
Raw public data
    ↓
Append-only as-of data lake
    ↓
Entity resolution + event extraction
    ↓
Strategy-specific feature stores
    ↓
Hypothesis testing / backtest engine
    ↓
Paper ops + live micro-sizing
    ↓
Risk engine + execution logger
    ↓
PnL attribution + kill criteria
```

## Layer 1: append-only as-of data lake

Every record needs:

```text
source_id
source_url / source_name
source_timestamp
vendor_timestamp
ingestion_timestamp
first_seen_timestamp
revision_timestamp
raw_hash
parsed_version
parser_version
```

Never overwrite. Corrections become new rows.

This matters especially for:

* SEC filings,
* earnings calendars,
* transcripts,
* analyst revisions,
* court dockets,
* government awards,
* macro releases,
* options chains,
* corporate actions.

SEC’s EDGAR APIs provide company submissions and XBRL financial data, with JSON structures updated throughout the day as submissions are disseminated; that is exactly the kind of source where “first seen” timestamps matter. ([SEC][5])

## Layer 2: entity graph, not GNN-first

Build a deterministic entity graph:

```text
company
  ├── ticker history
  ├── CIK
  ├── subsidiaries
  ├── products
  ├── government-award names
  ├── court-party names
  ├── patent assignees
  ├── competitors
  ├── customers/vendors
  ├── ETFs/indexes
  └── options roots
```

This is where a small trader can actually beat lazy processing. Public data is full of mismatched names, old tickers, subsidiaries, abbreviations, and legal entities.

## Layer 3: event store

Convert messy public information into normalized events:

```text
event_id
entity_id
event_type
event_subtype
event_timestamp
first_seen_timestamp
materiality_score
novelty_score
source_reliability
market_awareness_score
affected_instruments
expected_holding_period
eligible_strategies
```

Examples:

* 8-K covenant breach,
* new shelf offering,
* auditor resignation,
* government contract award,
* court ruling,
* FDA decision,
* patent litigation update,
* major customer loss,
* insider Form 4 cluster,
* analyst revision cluster,
* abnormal job posting decline,
* plant closure local news.

## Layer 4: strategy modules

Each strategy gets its own model, features, risk rules, and kill criteria.

Good modules:

1. **Public-event drift module**

   * small/mid-cap events,
   * materiality scoring,
   * newswire coverage filter,
   * liquidity filter,
   * post-event return model.

2. **Earnings volatility module**

   * implied move vs realized move,
   * historical event distributions,
   * IV crush forecast,
   * liquidity/spread filter,
   * defined-risk options structures.

3. **PEAD/revisions module**

   * earnings surprise,
   * guidance surprise,
   * analyst revisions,
   * estimate dispersion,
   * underreaction over days/weeks.

4. **Legal/regulatory module**

   * court dockets,
   * FDA/regulatory calendars,
   * agency decisions,
   * materiality and uncertainty scoring.

5. **Execution/liquidity module**

   * quote quality,
   * spread dynamics,
   * fill probability,
   * adverse selection,
   * order placement rules.

## Layer 5: model stack

Start boring:

* logistic regression,
* gradient boosting,
* calibrated random forest / XGBoost / LightGBM,
* hierarchical Bayesian models for event types,
* isotonic calibration,
* simple volatility models,
* robust baselines.

Use LLMs for extraction and classification. Do not let the LLM allocate capital.

## Layer 6: risk engine

Risk engine should be deterministic and mean:

```text
No trade unless:
  expected net edge > spread + fees + slippage + borrow + financing + error buffer
  liquidity supports entry and exit
  max loss is known or bounded
  correlation cluster budget available
  strategy drawdown state allows trade
  event calendar is checked
  data freshness passes
```

For options:

```text
Reject if:
  spread / mid > threshold
  OI too low
  volume too low
  IV estimate stale
  event date uncertain
  assignment/dividend risk not modeled
  short convexity uncapped
  exit liquidity questionable
```

## Layer 7: execution

Small-trader execution should be humble:

* limit orders only for options,
* midpoint or better when possible,
* staged price improvement,
* never cross wide spreads unless edge is huge,
* cancel stale orders,
* record NBBO at submit/fill/cancel,
* measure realized spread paid,
* avoid opening trades during chaotic first minutes unless strategy specifically needs it,
* avoid forced liquidation near close.

SEC adopted Rule 605 amendments to expand and modernize public execution-quality reporting, including broader covered entities and summary execution reports, which means execution-quality measurement is becoming more transparent — but that does not make retail execution good by default. ([SEC][6])

---

# 3. Where a small trader can realistically have edge

## Efficient zones

Do not waste time here first:

* SPY/QQQ direction from public news,
* mega-cap earnings headline reaction,
* obvious analyst upgrades,
* Fed statement first-order interpretation,
* high-liquidity ETF arbitrage,
* liquid index futures reaction,
* simple moving-average strategies,
* generic sentiment models,
* SPX 0DTE directional gambling,
* HFT order-book prediction.

Institutions dominate these because they have better data, faster execution, lower costs, better routing, and more scale.

## Less efficient zones

Small trader edge is more plausible where:

1. **Capacity is tiny**

   * A $20k–$200k trade can matter to you but not to a fund.
   * A $5M edge is too small for a large institution.

2. **Information is public but annoying**

   * SEC exhibits,
   * court filings,
   * government-award databases,
   * local news,
   * regulatory calendars,
   * patent assignments,
   * state-level permits,
   * obscure transcripts.

3. **Entity linking is hard**

   * Award goes to subsidiary, not ticker.
   * Lawsuit names operating entity, not parent.
   * Patent assignee changed name.
   * Supplier/customer relationship is buried in an old 10-K.

4. **Market awareness is delayed**

   * No Bloomberg headline.
   * No major newswire.
   * No analyst coverage.
   * No social-media attention yet.

5. **Large firms avoid the trade**

   * too illiquid,
   * too operationally annoying,
   * too legally/compliance-sensitive,
   * too hard to size,
   * too short-lived,
   * too idiosyncratic.

6. **Retail flow distorts options**

   * lottery demand,
   * meme calls,
   * earnings-vol overpayment,
   * 0DTE impulse trading,
   * wide-spread contracts.

Cboe estimated that in May 2025 retail made up **54%** of total SPX 0DTE volume, with 0DTE over **61%** of May SPX option volume. ([Cboe Global Markets][7]) Retail flow is not automatically dumb, but it can create predictable demand pressure in specific option habitats.

---

# 4. Best candidate domains for a nobody to research first

Scores: **5 = high**, **1 = low**. For difficulty, competition, execution difficulty, capital requirement, and ruin risk, lower is better.

| Rank | Domain                                                     | Edge potential | Difficulty | Data availability | Capital need | Execution difficulty | Competition | Ruin risk | Backtest reliability | Verdict                                        |
| ---: | ---------------------------------------------------------- | -------------: | ---------: | ----------------: | -----------: | -------------------: | ----------: | --------: | -------------------: | ---------------------------------------------- |
|    1 | Public-event drift in small/mid caps                       |              4 |          3 |                 4 |            1 |                    3 |           3 |         3 |                    3 | Best starting point. Ugly data + low capacity. |
|    2 | PEAD + revisions in undercovered names                     |              3 |          2 |                 5 |            2 |                    2 |           4 |         2 |                    4 | Less sexy, more testable.                      |
|    3 | Earnings options IV mispricing                             |              4 |          4 |                 3 |            2 |                    4 |           4 |         4 |                    2 | Real edge possible, but fills/tails brutal.    |
|    4 | Legal/regulatory/FDA/court events                          |              4 |          4 |                 3 |            2 |                    3 |           3 |         4 |                    2 | High upside, hard labels, binary risk.         |
|    5 | Government contracts/grants                                |              3 |          3 |                 5 |            1 |                    3 |           2 |         3 |                    2 | Great data, hard materiality.                  |
|    6 | ETF/index rebalance small-name mechanics                   |              3 |          3 |                 4 |            2 |                    3 |           5 |         2 |                    4 | Crowded but still niche in tiny names.         |
|    7 | Options liquidity provision in wide-but-tradable contracts |              3 |          4 |                 2 |            2 |                    5 |           4 |         4 |                    1 | Live-fill skill matters more than backtest.    |
|    8 | Supply-chain/job-posting/patent slow signals               |              3 |          5 |                 3 |            1 |                    2 |           3 |         2 |                    1 | Research-heavy, slow payoff.                   |
|    9 | Retail/meme-flow options distortions                       |              3 |          5 |                 2 |            2 |                    5 |           4 |         5 |                    1 | Can work, but easy to become exit liquidity.   |
|   10 | Liquid index/ETF vol surface                               |              2 |          5 |                 3 |            3 |                    4 |           5 |         4 |                    3 | Pros live here. Avoid first.                   |
|   11 | Macro event trading                                        |              2 |          5 |                 5 |            3 |                    3 |           5 |         3 |                    3 | Public info instantly digested.                |
|   12 | HFT/order-book market making                               |              1 |          5 |                 1 |            5 |                    5 |           5 |         4 |                    1 | Do not start here.                             |

My first serious research target would be **public-event drift in undercovered equities**, with options used only when liquidity and event-vol pricing justify it.

---

# 5. Options trading: where repeatable edges actually exist

Options traders who consistently make money are usually not “predicting direction.” They are harvesting or exploiting one of these:

1. **Volatility risk premium**
2. **Event-vol mispricing**
3. **Skew demand**
4. **Term-structure dislocations**
5. **Retail demand pressure**
6. **Liquidity provision**
7. **Inventory/risk-transfer compensation**
8. **Better execution**
9. **Better sizing**
10. **Better tail-risk control**

## IV vs realized volatility

The basic idea:

```text
If implied volatility is too high relative to future realized volatility,
short volatility earns money.

If implied volatility is too low relative to future realized volatility,
long volatility earns money.
```

But the naive version loses because:

* future realized volatility is hard to forecast,
* IV is high for a reason,
* tail events erase carry,
* transaction costs are large,
* short options create nonlinear losses,
* realized vol can arrive as one gap.

The edge is not “sell high IV.” The edge is:

```text
Sell IV only when:
  implied move > modeled event move
  liquidity is good
  structure is defined-risk
  skew/term structure supports it
  max loss is acceptable
  position survives gap scenarios
```

Or:

```text
Buy IV only when:
  implied move is underpriced
  catalyst is real
  expected move distribution is fat-tailed
  option spread is not insane
  timing is correct
```

## Earnings volatility premium

This is one of the more real retail-distortion areas. De Silva, Smith, and So find retail investors concentrate option purchases before earnings announcements, especially when expected abnormal volatility is high; they report retail losses of **5–9%** on average around earnings options and **10–14%** for high expected-volatility announcements, driven by overpaying relative to realized volatility, wide spreads, and slow post-announcement response. ([SSRN][8])

That does **not** mean blindly short every earnings straddle. It means there may be edge in carefully selected cases where:

* retail demand inflated IV,
* expected move is historically overstated,
* spreads are manageable,
* skew creates a better structure,
* trade is capped-risk,
* you do not hold naked short convexity through existential events.

Better structures than naked short straddles:

* iron flies,
* iron condors,
* calendars,
* diagonals,
* ratio spreads with defined disaster risk,
* post-earnings IV crush trades,
* long/short vol pairs across similar names.

## Volatility crush

Vol crush is real, but everyone knows it. The question is whether the crush is **larger than the realized move and transaction costs**.

Retail mistake:

```text
Buy call before earnings because company seems good.
Stock goes up.
Call still loses because IV collapses.
```

Skilled version:

```text
Estimate implied move.
Estimate realized move distribution.
Estimate post-event IV.
Choose structure where the scenario PnL is favorable across multiple paths.
```

## Skew

Skew exists because different options have different demand and risk profiles.

Common patterns:

* index downside puts are often expensive because institutions hedge crashes,
* meme/single-name upside calls can become expensive because retail buys lottery tickets,
* biotech/event names can have distorted both-tail pricing,
* hard-to-borrow names can distort puts/calls through stock loan and synthetic financing.

Potential edge:

* sell overpriced wing demand with capped risk,
* buy underpriced opposite wing,
* trade call-put relative value,
* use verticals instead of outright options.

Bad version:

* “puts are expensive, sell puts.”
* This works until the exact crash the put buyer feared.

## Term structure

Term structure matters around events.

Possible edges:

* front-month too expensive before earnings,
* back-month underprices continuing uncertainty,
* post-event IV remains elevated longer than market expects,
* short-dated options overpriced by retail urgency,
* calendar spreads misprice event decay.

But term-structure trades are subtle. You can be right on direction and still lose from skew, vol-of-vol, assignment risk, or liquidity.

## Gamma exposure and dealer hedging flows

Dealer gamma/GEX models are useful context, not standalone alpha.

Problems:

* public open interest is stale,
* trade direction is uncertain,
* dealer inventory is not directly observable,
* OTC positions are missing,
* hedging behavior changes with volatility/liquidity,
* models disagree.

Use GEX to ask:

```text
Could flows amplify or dampen moves near certain strikes?
Could pinning happen?
Could a break trigger hedging acceleration?
```

Do not use it as:

```text
GEX says market goes up.
```

That is astrology with Greeks.

## Unusual options activity

Most unusual options activity is garbage as a signal.

It can mean:

* informed flow,
* hedging,
* closing trade,
* spread leg,
* market-maker inventory transfer,
* retail chase,
* newsletter pump,
* pre-earnings speculation,
* mechanical roll.

Useful filters:

* opening vs closing,
* ask-side vs bid-side,
* sweep vs block,
* OI change next day,
* repeated buyer,
* catalyst proximity,
* ticker liquidity,
* IV reaction,
* stock reaction,
* relation to known news,
* whether the trade is directional or part of a spread.

Even then, “follow unusual activity” is usually late.

## Liquidity traps

Options liquidity is deceptive. A chain can show a tight quote and still be bad.

Watch:

* quoted spread / mid,
* size at NBBO,
* depth behind NBBO,
* OI,
* same-day volume,
* quote flicker,
* penny vs nickel increments,
* whether mid fills actually happen,
* exit liquidity after event,
* broker routing quality.

A backtest using mid prices in options is usually fiction. A backtest using last prices is often worse.

## Why most retail options traders lose

Main reasons:

* they buy expensive convexity after attention spikes,
* they trade short expiries with huge theta,
* they ignore implied move,
* they ignore spreads,
* they over-leverage,
* they hold through IV crush,
* they confuse payoff diagrams with probability,
* they chase screenshots,
* they size as if max loss is unlikely,
* they trade direction when the market is pricing distribution.

The trader-level retail options study by Bogousslavsky and Muravyev finds retail option trades are concentrated in a few underlyings, especially S&P 500 products, and dominated by short-term purchases rather than covered calls/protective puts. ([SSRN][9]) That is exactly the habitat where a sophisticated counterparty wants retail to show up.

## What separates skilled options traders from gamblers

Skilled options traders:

* trade distributions, not vibes,
* know whether they are long/short delta, gamma, vega, theta, skew, correlation,
* model event moves,
* care obsessively about execution,
* size small,
* cap tail risk,
* know when not to trade,
* understand assignment/exercise/dividend risk,
* track realized vs implied volatility,
* analyze PnL by Greek and by catalyst,
* survive long enough for edge to matter.

Gamblers:

* buy weekly calls because they “feel cheap,”
* sell naked premium because “theta gang,”
* average down options,
* ignore bid/ask,
* celebrate wins without expected value,
* size based on desired profit instead of risk.

---

# 6. Public information that is underused because it is annoying

The best public-info edge is usually not “secret data.” It is **public data that is operationally painful**.

## SEC filings

Useful:

* 8-K item classification,
* exhibit diffing,
* risk-factor changes,
* customer concentration changes,
* going-concern language,
* auditor resignation,
* debt covenant changes,
* ATM/shelf registrations,
* insider Form 4 clusters,
* amended filings,
* revenue recognition changes,
* backlog/order-book language.

SEC EDGAR APIs provide submissions and XBRL company financial data, updated through the day as submissions are disseminated; the data is available without API keys. ([SEC][5]) The edge is not getting the filing. The edge is parsing the right exhibit, linking the right entity, and knowing what is material before the market fully processes it.

## Federal contracts and grants

Useful:

* contract awards,
* modifications,
* renewals,
* IDIQ vehicles,
* agency concentration,
* award size vs company revenue,
* competitor displacement,
* geographic concentration.

USAspending’s API exposes public federal spending data, including contracts, grants, recipients, agencies, and geographies. ([USAspending API][10]) The hard part is materiality. A $200M award may be huge for one small contractor and irrelevant for a prime with $50B revenue.

## Court records

Useful:

* patent litigation,
* injunctions,
* antitrust rulings,
* product liability,
* bankruptcy dockets,
* regulatory enforcement,
* contract disputes,
* appellate decisions.

CourtListener is a free legal research site with millions of federal and state legal opinions and tools for tracking case law and raw data. ([CourtListener][11]) The edge is mapping cases to public companies and detecting material docket movement before broad coverage.

## Patent data

Useful:

* assignee trends,
* technology direction,
* competitor R&D,
* litigation exposure,
* product pipeline hints,
* supplier/customer innovation links.

USPTO’s PatentsView turns raw patent records into research-grade datasets using AI/ML/NLP methods and includes downloads for granted patents and pre-grant publications. ([USPTO][12]) Patent data is usually too slow for short-term trading, but useful for thematic baskets, competitive mapping, and event context.

## Local/regional news

Useful:

* plant closures,
* facility openings,
* permitting,
* union votes,
* layoffs,
* environmental incidents,
* lawsuits,
* municipal contract approvals,
* local subsidies.

Large funds can process this, but many do not care unless capacity is large. Small traders can.

## Regulatory calendars

Useful:

* FDA panels,
* EPA rules,
* FERC decisions,
* FCC auctions,
* ITC patent/import cases,
* state utility commission decisions,
* congressional hearings,
* agency comment deadlines.

The edge is not the calendar. It is the probability model plus the instrument selection.

## Job postings

Useful:

* hiring acceleration/deceleration,
* geographic expansion,
* product initiatives,
* salesforce buildout,
* AI/cloud/security staffing,
* sudden freezes.

Weakness: job postings are noisy, delayed, and easy to overinterpret.

## Supply chain and shipping

Useful:

* import/export records,
* port delays,
* commodity flows,
* vessel tracking,
* crop/weather impacts,
* inventory/channel checks.

Hard part: connecting physical flow to earnings impact and timing.

## App/web/social trend data

Useful:

* consumer apps,
* gaming,
* streaming,
* fintech,
* retail brands,
* viral product launches.

Danger: everyone sees the same social trend, and attention can reverse violently.

---

# 7. Concrete edge discovery loop

## Step 1: Hypothesis generation

Every hypothesis must have this form:

```text
Who is mispricing what?
Why does the mispricing exist?
Why does it persist?
Who is on the other side?
Why can I trade it but larger players may not?
What destroys it?
What is the expected holding period?
What are the costs?
```

Bad hypothesis:

```text
LLM sentiment predicts returns.
```

Better hypothesis:

```text
Small-cap companies receiving material federal contract awards
with no major newswire coverage show delayed positive drift over 1–5 trading days,
because the award is public but poorly linked to the ticker.
```

## Step 2: Data collection

Collect:

* raw source,
* source timestamp,
* first-seen timestamp,
* parsed fields,
* entity mapping,
* market data at first-seen time,
* tradable instruments,
* liquidity at decision time.

No timestamp, no trade.

## Step 3: Feature construction

Example features for contract-award strategy:

```text
award_value / trailing_revenue
award_value / market_cap
new_award_vs_modification
agency
contract_type
prior_award_history
company_size
analyst_coverage
short_interest
newswire_coverage
social_mentions
spread
dollar_volume
options_liquidity
sector beta
```

## Step 4: Labeling

Labels must match tradable reality:

* next 1-day, 3-day, 5-day return,
* market/sector-adjusted return,
* max adverse excursion,
* max favorable excursion,
* close-to-close and intraday labels,
* option return using executable bid/ask assumptions,
* event-window realized volatility,
* post-event IV crush.

## Step 5: Leakage checks

Run these:

* source timestamp audit,
* vendor delay audit,
* restatement detection,
* corporate-action adjustment check,
* survivorship check,
* delisted-name inclusion,
* randomized event timestamp placebo,
* shuffled ticker placebo,
* future-feature scan,
* post-event news contamination check.

## Step 6: Backtesting

Use:

* walk-forward testing,
* purged/embargoed splits,
* event-level cross-validation,
* sector-neutral tests,
* liquidity buckets,
* market-regime buckets,
* transaction-cost stress,
* slippage stress,
* borrow/financing stress,
* options bid/ask fill model.

Never accept a strategy that only works at midpoint fills in illiquid options.

## Step 7: Statistical correction

Track every trial.

Require:

* out-of-sample performance,
* Deflated Sharpe / multiple-testing correction,
* PBO estimate,
* parameter stability,
* monotonic feature behavior,
* performance by time period,
* performance by liquidity bucket,
* performance after doubled costs.

## Step 8: Paper trading

Paper trading is for:

* operational bugs,
* timestamp correctness,
* signal generation,
* order workflow,
* risk checks.

Paper trading is **not** reliable for options fills.

## Step 9: Live micro-sizing

Start absurdly small:

* equities: tiny share size,
* options: one-lot only,
* defined-risk only,
* limit orders only.

Record:

```text
signal
expected edge
quote at signal
quote at order
order price
fill price
cancel attempts
NBBO
mid
spread
exit quote
realized slippage
reason for entry/exit
```

## Step 10: Scaling

Scale only if:

* live fills match modeled fills,
* live PnL attribution matches backtest,
* edge persists after costs,
* drawdowns remain inside expected range,
* no hidden correlation blowups,
* strategy still works after excluding top winners.

## Step 11: Kill criteria

Kill or pause if:

* live expectancy negative after 30–50 independent trades,
* slippage exceeds model by 2x,
* drawdown breaches pre-set limit,
* feature distribution drifts materially,
* edge only comes from one outlier,
* liquidity vanishes,
* market awareness catches up,
* strategy underperforms random/placebo variant,
* live hit/payoff profile breaks.

---

# 8. Top ways the system fools itself

## 1. Lookahead bias

Most common. The model uses data that technically existed later.

Examples:

* earnings calendar revised after the event,
* filing parsed using later metadata,
* analyst revision timestamp from vendor not original release,
* options IV computed from closing chain after signal time,
* index membership known after rebalance announcement.

## 2. Survivorship bias

Using only current tickers makes small-cap strategies look amazing because the dead names vanish.

## 3. Options midpoint fantasy

Midpoint backtests print fake money. Real fills do not.

Especially bad in:

* low-volume weeklies,
* small-cap options,
* post-event exits,
* far OTM contracts,
* multi-leg spreads.

## 4. Selection bias

You test 500 ideas and remember the 3 that worked.

Without trial tracking, the research process lies.

## 5. Regime luck

A strategy found during a bull market may just be disguised beta. A short-vol strategy found during calm markets may just be hidden crash exposure.

## 6. Hidden correlation

“Diversified” event strategies can all be long:

* liquidity,
* small-cap beta,
* retail risk appetite,
* short-vol,
* AI theme,
* credit conditions.

Then one macro shock hits everything.

## 7. Miscalibrated confidence

ML probabilities often look precise but are not. A “70% win probability” model may mean nothing if trained on leaky, non-stationary data.

## 8. Ignoring capacity

Small edges disappear once you scale. In illiquid options, your own second contract may be enough to move the quote.

## 9. Borrow and financing blindness

Short-equity strategies can be ruined by borrow costs, recalls, hard-to-borrow constraints, and T+1 settlement mechanics. The U.S. moved to T+1 settlement in May 2024, reducing settlement time but also tightening operational timing. ([SEC][13])

## 10. Narrative overfitting

LLMs are especially dangerous here. They make plausible explanations for noise.

## 11. Confusing public availability with market neglect

A public datapoint is not underused just because you found it. The question is whether it is:

* material,
* tradeable,
* not already priced,
* not too costly,
* not too late.

## 12. Mistaking volatility for edge

Options strategies can produce huge wins and huge losses. A few big wins do not prove positive expectancy.

---

# 9. Minimal viable version for one person

Build less.

The MVP should be:

```text
One as-of data store
Three research tracks
One backtest engine
One execution logger
One risk engine
No autonomous strategy synthesis
No giant world model
```

## First 3 research tracks

### Track 1: Public-event drift in undercovered equities

Focus:

* SEC 8-Ks,
* contract awards,
* court/legal events,
* small/mid-cap material announcements,
* no major newswire coverage,
* low analyst coverage.

Trade expression:

* equity first,
* options only if liquid.

Why this first:

* public data,
* low capacity can be an advantage,
* edge is explainable,
* LLM extraction helps,
* easier risk than naked options.

### Track 2: Earnings options IV mispricing

Focus:

* implied move vs historical realized event move,
* IV crush,
* retail attention,
* liquidity filters,
* defined-risk structures.

Trade expression:

* iron flies,
* calendars,
* verticals,
* small debit spreads,
* occasionally long volatility when implied move is too cheap.

Avoid:

* naked short straddles,
* illiquid weeklies,
* binary biotech,
* “cheap” OTM lotto tickets.

### Track 3: PEAD + revisions in small/mid caps

Focus:

* earnings surprise,
* guidance surprise,
* analyst revision clusters,
* estimate dispersion,
* low coverage,
* post-event drift over 3–30 days.

PEAD is one of the rare public-information anomalies with a long literature; a 2021 review describes it as price drift in the direction of earnings surprise after announcements rather than full immediate adjustment. ([ScienceDirect][14]) It is also crowded in large caps, so the research target should be undercovered names and better feature construction.

## First datasets

Minimum:

* point-in-time daily/intraday equity prices,
* corporate actions and delistings,
* SEC submissions/XBRL,
* earnings dates,
* options EOD chain with bid/ask/OI/volume/IV,
* short interest,
* borrow availability/cost if accessible,
* USAspending awards,
* CourtListener/legal events,
* basic news coverage timestamps,
* broker execution logs.

## First backtesting framework

Use Python, but keep it boring:

```text
Polars/Pandas for research
DuckDB/Postgres for storage
Parquet for immutable raw data
custom event-driven backtester
separate options fill simulator
experiment registry
```

Backtest outputs:

* gross return,
* net return,
* spread paid,
* slippage,
* turnover,
* beta/sector exposure,
* max drawdown,
* hit rate,
* payoff ratio,
* tail loss,
* capacity estimate,
* performance by liquidity bucket,
* performance by event type.

## First models

Use:

* rules + linear/logistic models,
* gradient boosted trees,
* isotonic calibration,
* simple Bayesian shrinkage by event type,
* LLM extraction with validation,
* no end-to-end deep trading model.

Model outputs:

```text
probability of positive net return
expected net return
expected max adverse excursion
expected holding period
liquidity-adjusted capacity
confidence bucket
```

## First risk rules

For MVP:

* max loss per trade: tiny,
* max daily loss,
* max weekly loss,
* no uncapped short options,
* no illiquid options,
* no trading if spread/mid too high,
* no position where exit liquidity is uncertain,
* no more than a few correlated positions,
* no averaging down options,
* no event trade without event calendar verification,
* pause strategy after predefined drawdown.

For options specifically:

```text
Max premium at risk per trade.
Defined-risk spreads only.
No naked short gamma.
No market orders.
Reject contracts with bad spread/OI/volume.
One-lot live testing until fill model is proven.
```

## First live/paper setup

Paper trade for operations. Live micro-size for execution truth.

Record every order:

```text
timestamp
signal ID
model version
NBBO
mid
order limit
fill price
fill time
cancel/replace path
exit reason
realized slippage
realized PnL
Greek exposure
```

The execution journal is as important as the alpha model.

## Metrics to continue or kill

Continue only if:

* live net expectancy is positive,
* fill model error is acceptable,
* signal calibration is stable,
* results survive doubled-cost stress,
* PnL is not one outlier,
* drawdowns match expected distribution,
* strategy works in multiple time slices,
* strategy has a believable counterparty.

Kill if:

* only midpoint fills make it profitable,
* live trades underperform backtest materially,
* alpha decays after a few weeks,
* edge depends on illiquid exits,
* performance disappears after removing top 5 trades,
* explanation changes every time it loses.

---

# 10. Strongest alternative thesis

The most realistic path for a small AI-assisted trader in 2026 is **not** “AI predicts markets.”

It is:

> Build an AI-assisted public-information processing pipeline that finds low-capacity, undercovered, legally public events faster and more reliably than normal humans, then express only the cleanest opportunities through simple equity trades or defined-risk option structures, with obsessive as-of data control and live execution measurement.

The durable edge is not the model. The edge is the combination of:

* ugly public data,
* better entity linking,
* faster event structuring,
* materiality scoring,
* low-capacity small-name focus,
* disciplined options selection,
* realistic fills,
* hard risk limits,
* fast kill criteria.

The “nobody” advantage is real only in narrow places:

* trades too small for institutions,
* names too illiquid for funds,
* data too fragmented for casual traders,
* signals too operationally annoying for discretionary investors,
* options distortions created by retail demand,
* public events that diffuse slowly.

The system should be a **research-and-execution machine**, not an omniscient trader. Start with three narrow edges, prove one survives live costs, then scale slowly. Most of the proposed architecture should be deleted until a small, boring, measurable edge exists.

[1]: https://www.theocc.com/newsroom/views/2026/01-05-occ-annual-2025-and-december-2025-volume "OCC - OCC Annual 2025 and December 2025 Volume"
[2]: https://www.sec.gov/files/roundtable-options-market-structure.pdf "Roundtable on Options Market Structure"
[3]: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551 "The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting and Non-Normality by David H. Bailey, Marcos Lopez de Prado :: SSRN"
[4]: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253 "The Probability of Backtest Overfitting by David H. Bailey, Jonathan Borwein, Marcos Lopez de Prado, Qiji Jim Zhu :: SSRN"
[5]: https://www.sec.gov/search-filings/edgar-application-programming-interfaces "SEC.gov | EDGAR Application Programming Interfaces (APIs)"
[6]: https://www.sec.gov/newsroom/press-releases/2024-32 "SEC.gov | SEC Adopts Amendments to Enhance Disclosure of Order Execution Information"
[7]: https://www.cboe.com/insights/posts/spx-0-dte-options-jump-to-61-share-on-retail-resurgence/ "SPX® 0DTE Options Jump to 61% Share on Retail Resurgence | Cboe"
[8]: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4050165 "Losing is Optional: Retail Option Trading and Expected Announcement Volatility by Tim de Silva, Kevin Smith, Eric C. So :: SSRN"
[9]: https://papers.ssrn.com/sol3/Delivery.cfm/4682388.pdf?abstractid=4682388&mirid=1 "An Anatomy of Retail Option Trading by Vincent Bogousslavsky, Dmitriy Muravyev :: SSRN"
[10]: https://api.usaspending.gov/ "USAspending API"
[11]: https://www.courtlistener.com/ "Non-Profit Free Legal Search Engine and Alert System – CourtListener.com"
[12]: https://www.uspto.gov/ip-policy/economic-research/patentsview "PatentsView | USPTO"
[13]: https://www.sec.gov/newsroom/press-releases/2024-62 "SEC.gov | SEC Chair Gensler Statement on Upcoming Implementation of T+1 Settlement Cycle"
[14]: https://www.sciencedirect.com/science/article/pii/S2214635020303750 "A review of the Post-Earnings-Announcement Drift - ScienceDirect"

