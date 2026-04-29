# GPT Pro Architecture Recommendation - TradeML

Source: https://chatgpt.com/c/69f230d6-446c-83ea-a088-0d1dbe2a3b4d
Captured: 2026-04-29

The following is copied from the GPT Pro Extended Thinking response in the ChatGPT thread.

---

I’ll treat this as a system-design and research-governance problem, not a search for the flashiest model. I’ll prioritize practical local-compute choices, point-in-time safety, leakage control, costs, nonstationarity, and disciplined promotion rules.

Early conclusion: start with a point-in-time feature factory, supervised cross-sectional ranking, regularized linear/rank baselines, boosted trees, and cost-aware portfolio construction before deep/RL experiments.

Thought for 8m 21s
Bottom line recommendation

Build this as supervised cross-sectional alpha prediction + cost-aware portfolio construction + strict incumbent/challenger research automation. Do not start with reinforcement learning, large transformers, GNNs, or end-to-end news sequence models. Those are later-stage challengers after the data, validation, cost model, and model tournament are already hard to fool.

The practical ranking is:

Regularized linear/rank baselines + gradient boosted trees

Ensembles, calibration, meta-labeling, and cost-aware portfolio construction

Text/event embeddings as engineered features into tabular models

Small temporal CNN/TCN feature extractors for minute-derived features

Tabular deep learning / FT-Transformer-style challengers

TFT / sequence transformers only after the feature store and evaluation loop are mature

GNNs only when you have point-in-time relationship graphs

Time-series foundation models as auxiliary forecasters/features, not the core alpha engine

RL only for constrained allocation/execution research later, not as the first alpha architecture

Your biggest risk is not model weakness. It is leakage, survivorship bias, false discovery, costs, turnover, free-data quality, and fooling yourself with thousands of backtests. Empirical asset-pricing research does support using ML for cross-sectional return prediction, with trees and neural nets showing gains from nonlinear interactions; it also finds that dominant useful signal families are still mundane things like momentum, liquidity, volatility, and valuation-related variables. It also notes shallow learning can outperform deeper learning in this setting because asset-pricing data are low-signal and limited relative to vision/NLP datasets. 
OUP Academic

1. Ranked model architecture families
Tier 0: the non-negotiable “model” is the research stack

Before architecture selection, build a system that can answer: “Would this model have made this decision using only data available at that timestamp?”

Use:

Immutable raw data lake: vendor, endpoint, request time, vendor timestamp, ingestion timestamp, payload hash.

Point-in-time curated tables: prices, corporate actions, fundamentals, filings, news, universe membership.

Feature store with feature_available_at, not just feature_date.

Experiment registry: code commit, data snapshot, feature version, model version, label definition, costs, constraints, hyperparameter search count.

Incumbent/challenger registry.

Paper signal ledger and replayable backtests.

Without this, every advanced model becomes a sophisticated leakage amplifier.

Tier 1: regularized linear/rank models

Use first as baselines and sanity checks.

Architectures:

Ridge / ElasticNet regression on winsorized, cross-sectionally normalized features.

Logistic regression for top/bottom quantile classification.

Pairwise or listwise rank models.

Simple factor composites: momentum, reversal, quality, value, volatility, liquidity.

Why:

Fast on Mac Mini.

Interpretable.

Harder to overfit than flexible deep models.

Excellent leakage detector: if a simple model shows absurd performance, your pipeline is probably leaking.

Gives a baseline incumbent that every complex model must beat.

Failure modes:

Misses nonlinear interactions.

Can become a disguised factor model with no edge after costs.

Sensitive to normalization and universe construction.

Looks bad if labels are noisy and unresidualized.

Use it to answer: “Is there any signal at all?”

Tier 2: gradient boosted trees

This should be the first real workhorse.

Use LightGBM or XGBoost-style models, probably LightGBM first for local CPU speed.

Preferred variants:

Regression on forward residual return.

Rank objective grouped by date.

Binary classifier for top/bottom decile membership.

Quantile regression for uncertainty.

Separate models by horizon: 1-day, 5-day, 20-day.

Why:

Strong on tabular data.

Handles nonlinear interactions without needing huge compute.

Works well with mixed feature types.

Easier to inspect with permutation importance, SHAP-style diagnostics, and feature ablation.

Empirical asset-pricing work finds trees and neural nets among the strongest ML methods because they capture nonlinear interactions missed by simpler methods. 
OUP Academic

Failure modes:

Overfits stale microcap/liquidity artifacts.

Learns vendor quirks.

Learns universe-selection leakage.

Produces unstable feature importance.

High turnover if the objective ignores costs.

Practical verdict: main incumbent candidate by day 30–60.

Tier 3: ensemble/meta-labeling

Add early, but only after individual models are clean.

Use:

Equal-weight model average across linear, GBDT, and simple factor models.

Stacking only with out-of-fold predictions.

Meta-labeling to decide whether to act on a primary signal.

Confidence model for bet sizing: “when is the alpha model likely to be right?”

Regime-conditioned ensemble weights.

Why:

Ensemble forecasts in Gu/Kelly/Xiu’s asset-pricing study improved over individual models in some cases, and model averaging is often more robust than trying to choose the single best architecture. 
OUP Academic

Meta-labeling can reduce turnover and false positives.

Useful for self-improving systems because challengers can be added without replacing the whole stack.

Failure modes:

Stacking leakage.

Meta-model overfitting to validation periods.

Hidden multiple-testing problem.

“Ensemble” becomes a way to keep bad models alive.

Practical verdict: use as the promotion layer, not as a magic alpha source.

Tier 4: temporal CNNs / TCNs

Second-stage architecture for minute-derived patterns, not first-stage end-to-end trading.

Use:

1D CNN / temporal convolution over recent intraday bars.

Inputs: returns, volume, realized volatility, spread proxy, time-of-day bucket.

Output: compact features such as “intraday accumulation,” “late-day reversal probability,” “opening instability,” “execution risk,” “next-day volatility.”

Why:

Cheaper than transformers.

Can learn local temporal motifs from minute bars.

Useful for execution timing and cost modeling.

Easier to train locally than large recurrent/attention models.

Failure modes:

Minute bars are noisy.

Free-plan minute data may have gaps, delayed access, or inconsistent coverage.

End-of-day signal can accidentally use post-decision intraday information.

Learns microstructure artifacts that vanish after costs.

Practical verdict: use as a feature extractor after the daily GBDT stack is working.

Tier 5: tabular deep learning

Worth testing later, but not the first bet.

Architectures:

MLP with residual connections.

FT-Transformer-style tabular model.

TabNet only as a benchmark, not a priority.

Autoencoder/pretraining for representation learning if you have many unlabeled rows.

Why:

Can capture smooth interactions.

Can combine dense embeddings with numeric features.

May help once you have large, clean, multi-modal features.

But: tabular deep learning has no universal advantage over gradient boosted trees. A major benchmark paper on tabular deep learning compared strong deep models with GBDTs and concluded there is still no universally superior solution. 
arXiv

Failure modes:

Requires more tuning.

Less stable on low-signal financial labels.

Can underperform GBDT while consuming far more research time.

Harder to debug leakage.

Practical verdict: test in the 60–90 day window as challengers, not incumbents.

Tier 6: TFT / sequence transformers

Powerful-looking, but not the best first move.

TFT is designed for multi-horizon forecasting with static covariates, historical exogenous inputs, known future inputs, recurrent local processing, attention for longer dependencies, feature selection, and gating. 
ScienceDirect

Why it could help:

Multi-horizon forecasts.

Joint handling of static fundamentals, historical bars, macro features, and event features.

Potentially useful for predicting volatility, regime, or multi-day return distributions.

Why it is dangerous early:

Huge leakage surface.

Hard to validate.

Hard to compare fairly.

Needs a lot of clean data.

Often worse than GBDT on tabular cross-sectional alpha after costs.

Practical verdict: later challenger, not first architecture.

Tier 7: graph neural nets

Only later, only with real point-in-time graphs.

Useful graphs:

Sector/industry taxonomy.

Supply chain.

Customer/vendor relationships.

Common ownership.

ETF/index membership.

News co-mentions.

Board/executive links.

Correlation networks, but be careful because rolling correlations are unstable and can leak if computed incorrectly.

Why:

Cross-sectional equity prediction is naturally relational.

Events can propagate across suppliers, competitors, and sector peers.

Failure modes:

Relationship data is rarely truly point-in-time on free plans.

Static graphs encode hindsight.

Correlation graphs become regime-specific noise.

GNN complexity hides weak data.

Practical verdict: do not touch until you can prove the graph edges existed before the prediction date.

Tier 8: time-series foundation models

Use as auxiliary features or benchmarks, not core alpha.

Models like Chronos tokenize time-series values and train transformer language-model architectures on those tokens; the authors report comparable or occasionally superior zero-shot performance on new forecasting datasets relative to models trained specifically on them. 
arXiv

Possible uses:

Volatility forecasts.

Regime/anomaly scores.

Imputation.

Feature generation for price/volume histories.

Benchmark against simpler statistical forecasts.

Do not assume:

Zero-shot price forecasting equals tradable equity alpha.

A foundation model understands transaction costs.

A general time-series model knows cross-sectional stock selection.

Practical verdict: interesting later, especially for volatility/regime features; not your day-1 alpha engine.

Tier 9: reinforcement learning

Do not start here.

RL is attractive because it can combine prediction, position sizing, transaction costs, liquidity, and risk into one objective. Surveys correctly note this appeal. 
EconStor

But finance RL faces exactly the conditions that make RL brittle: noisy rewards, non-stationarity, heavy-tailed outcomes, difficult MDP modeling, explainability issues, and robustness problems. 
arXiv

The trap:

The RL agent does not learn the market. It learns your simulator.

If the simulator has optimistic fills, stale spreads, survivorship bias, or missing market impact, the agent optimizes those errors.

Practical use later:

Execution scheduling.

Threshold selection.

Allocation among already-validated strategies.

Contextual bandit for ensemble weighting.

Risk-budget adjustment.

Practical verdict: later optimizer, not first alpha researcher.

2. Best framing: supervised learning + portfolio construction, with limited online adaptation
Primary framing

Use:

Supervised cross-sectional prediction/ranking → cost-aware optimizer → paper/shadow execution → drift monitoring → incumbent/challenger promotion.

Each day or decision time:

Build point-in-time feature vector x_i,t for each stock.

Predict expected residual return or rank score s_i,t.

Convert scores into portfolio weights with costs, risk, liquidity, and exposure constraints.

Paper-trade.

Compare realized outcomes to incumbent, random portfolios, and benchmarks.

This is more likely to work than RL because it decomposes the system into testable parts.

Label choices

Use multiple horizons:

1D: next close-to-close or open-to-close residual return.

5D: short swing.

20D: slower cross-sectional factor horizon.

Better labels:

Market/sector residual return.

Beta-neutral residual return.

Cross-sectional z-score by date.

Top/bottom quantile class.

Pairwise ranking target.

Avoid raw “will price go up tomorrow?” as the main label. It mixes market beta, macro noise, and stock-specific alpha.

Portfolio construction

Separate prediction from sizing.

A practical objective:

w
max
	​

μ
⊤
w−λw
⊤
Σw−c(w,w
prev
	​

)−η∥w∥
1
	​


Subject to:

Gross exposure limit.

Net exposure limit.

Max single-name weight.

Sector/industry exposure caps.

Beta exposure cap.

ADV participation cap.

Minimum liquidity.

Shorting constraints, or long-only if paper broker/data cannot realistically model shorts.

Turnover cap.

This matters because recent work on implementable ML portfolios emphasizes that transaction costs can make ML signals hard to implement and that cost-aware portfolio optimization should be integrated into the objective. 
OUP Academic

Contextual bandits and online learning

Use them narrowly.

Good uses:

Choose among validated model families.

Adjust ensemble weights.

Select rebalance frequency.

Decide whether to trade or hold.

Choose risk budget based on regime.

Bad uses:

Treat thousands of stocks as arms without robust delayed-reward handling.

Continuously update a live model every minute.

Let online learning overwrite a validated incumbent without a quarantine period.

The right hybrid:

Offline supervised alpha models + online calibration/ensemble weighting + cost-aware optimizer + strict promotion gates.

3. News, events, and filings architecture
Start with embeddings + engineered event features into tabular models

Do not start with an end-to-end sequence transformer over all news.

Use this architecture:

Event ingestion table

event_id

ticker/entity_id

source

source_published_at

vendor_received_at

system_ingested_at

tradable_available_at = max(source_published_at, vendor_received_at, system_ingested_at) + safety_lag

event_type

headline

body/hash

filing_accession_number, if SEC filing

relevance_score

Text model layer

Frozen financial sentiment model.

Embedding model.

LLM or classifier for event type: earnings, guidance, M&A, litigation, product, insider, dividend, buyback, offering, downgrade, macro, regulatory.

Novelty score versus prior articles and filings.

Aggregation layer

Per stock and time window:

count of relevant events,

sentiment mean/max/min,

negative-event flag,

novelty,

source credibility,

event recency,

event type one-hot/counts,

abnormal news volume,

filing type,

8-K item type,

filing section change score.

Tabular alpha model

Feed aggregated text/event features into GBDT or regularized model.

Compare with and without text features through feature ablation.

FinBERT-style models are relevant because financial text has specialized language; FinBERT was introduced specifically for financial sentiment analysis using finance-domain pretraining/fine-tuning. 
arXiv

Filings: retrieval features beat raw text at first

For 10-K/10-Q/8-K:

Build features like:

Filing type.

Accepted timestamp.

Filing delay.

Accession number.

Section-level sentiment.

Risk-factor section similarity versus prior filing.

MD&A similarity versus prior filing.

New terms/entities.

Accounting fact changes from XBRL:

revenue growth,

gross margin,

operating margin,

leverage,

cash flow,

accruals,

share count,

buyback/dividend signals.

“Material change” score.

SEC EDGAR access is free, but automated collection must obey fair-access rules; SEC states a current max request rate of 10 requests/second and asks scripted users to download only what they need. 
SEC

Avoiding leakage with text/events

Hard rules:

Use availability time, not article date.

Apply a safety delay for vendor/news latency.

For daily close-to-close predictions, decide whether the decision happens:

previous close,

next open,

9:45 a.m.,

noon,

close.

News after the decision time is unavailable.

SEC filing period date is not availability date; EDGAR acceptance time is the key.

Do not train embeddings or sentiment models on labels from the same evaluation period unless through proper walk-forward training.

Do not use article revisions unless the revision timestamp is modeled.

Do not aggregate “all news on date t” if some arrived after the trade decision.

Negative controls:

Randomly permute ticker-news mapping.

Shift news timestamps forward.

Use “future news” deliberately in a leak-test model; it should perform suspiciously well.

Add random fake event features; they should not survive promotion.

Compare “text features only” against “price/fundamental only” and require incremental net value.

4. Autonomous research loop: what to optimize and gate on
System loop

Run this continuously:

Collect

Pi node collects raw data.

Rate-limited, append-only, checksummed.

Store vendor payloads and timestamps.

Validate

Missingness checks.

Duplicate checks.

Vendor disagreement checks.

Corporate action checks.

Timestamp monotonicity.

Universe membership checks.

Feature build

Daily PIT features.

Minute-derived features.

Fundamental/event features.

Store with feature version.

Candidate generation

Predefined search space.

No unconstrained architecture roulette.

Every trial logged.

Training

Rolling or expanding windows.

Training/validation/test split by time.

Purged/embargoed splits for overlapping labels.

Backtest

Same portfolio construction rules as paper.

Same costs.

Same decision timestamps.

Same universe.

Gate

Compare to incumbent.

Compare to random portfolios.

Compare to simple factors.

Compare to S&P 500 total return where appropriate.

Penalize model complexity and number of trials.

Shadow

Candidate generates paper signals without replacing incumbent.

Promote or retire

Promote only if it beats incumbent under predeclared rules.

Retire if drift, instability, or cost sensitivity breaks it.

Validation design

Use walk-forward evaluation as the final judge.

Example:

Train: 36–60 months.

Validation: 6–12 months.

Test: next 1–3 months.

Roll forward.

Report distribution across windows, not just aggregate performance.

For overlapping labels, use purging and embargo. Purging removes training observations whose labels overlap with test labels; embargoing removes observations immediately following the test set because financial features can be serially correlated. 
skfolio

Recommended hierarchy:

Train/validation split for tuning.

Purged/embargoed CV inside training period for hyperparameter comparison.

Chronological walk-forward test for final reporting.

Paper/shadow period before any live consideration.

Objective functions
Model-level objectives

Use several, but promote on portfolio outcomes.

Good ML objectives:

Rank IC / Spearman correlation.

Pairwise ranking loss.

Listwise ranking loss by date.

Huber loss on residual returns.

Quantile loss for uncertainty.

Binary cross-entropy for top/bottom decile classification.

Bad primary objectives:

Plain accuracy on up/down labels.

Raw MSE on unscaled returns.

Optimizing only in-sample Sharpe.

Optimizing only top-decile return without turnover/cost.

Portfolio-level objectives

Promote on net, risk-adjusted, cost-aware portfolio performance, not just prediction metric.

Track:

Net annualized return.

Net Sharpe.

Sortino.

Max drawdown.

Calmar.

Turnover.

Cost drag.

Average holding period.

Hit rate.

Profit factor.

Exposure to market beta.

Sector exposure.

Factor exposure.

Capacity estimate.

Performance by regime.

Performance by liquidity bucket.

Rank IC by month.

Top-minus-bottom spread after costs.

For S&P comparison:

Compare long-only strategy to S&P 500 total return.

Compare market-neutral long/short to cash/T-bill plus alpha, and also report beta-adjusted alpha versus S&P.

Do not call a market-neutral strategy “beating S&P” just because it has a higher Sharpe; compare return, drawdown, beta, and capital usage.

Transaction cost model

Use pessimistic costs.

At minimum:

cost=commission+fees+spread/2+slippage+impact

Estimate:

Half-spread proxy.

Volatility-adjusted slippage.

Participation-rate penalty.

ADV cap.

Short borrow unavailable/expensive flag if shorting.

Separate open, close, and intraday execution assumptions.

Gate candidates at:

1x estimated costs.

2x estimated costs.

Delayed execution.

Worse fills.

Higher turnover cap.

A model that only works at unrealistically low friction is not an alpha model; it is a cost-model exploit.

Statistical gates

Use:

Positive mean rank IC across walk-forward windows.

Rank IC t-stat with autocorrelation-aware standard errors.

Positive net top-minus-bottom spread.

Candidate beats incumbent across most windows, not just one crisis period.

Deflated Sharpe or multiple-testing adjustment.

Probability of backtest overfitting estimate when many strategy variants were tested.

The Deflated Sharpe Ratio was proposed to correct Sharpe inflation from selection bias, backtest overfitting, and non-normal returns; PBO methods were proposed specifically for estimating overfitting probability in investment simulations. 
SSRN
+1

Suggested paper-promotion gates:

Candidate has higher net Sharpe than incumbent by at least a meaningful margin, e.g. +0.15 to +0.25, not just +0.01.

Candidate’s net return remains positive at 2x costs.

Candidate has no worse max drawdown unless return improvement compensates.

Candidate has positive rank IC in at least 60–70% of walk-forward periods.

Candidate’s performance is not concentrated in one month, one sector, or one liquidity bucket.

Candidate beats random portfolios with identical constraints and turnover.

Candidate survives feature ablation: removing one suspicious feature should not destroy the entire strategy unless that feature is economically justified.

Simpler model wins if performance is statistically indistinguishable.

Drift detection

Monitor four kinds of drift.

1. Data drift

Feature distribution shifts.

Missingness.

Vendor gaps.

Corporate action anomalies.

Universe size changes.

News volume shifts.

Tools/ideas:

PSI.

KS tests.

Population quantiles.

Missingness dashboards.

Feature z-score limits.

2. Prediction drift

Score distribution.

Rank concentration.

Average confidence.

Correlation with incumbent.

Correlation with simple factors.

Long/short exposure changes.

3. Concept/performance drift

Rolling rank IC.

Rolling net PnL.

Rolling hit rate.

Rolling top-minus-bottom spread.

Prediction calibration.

Algorithms such as Page-Hinkley and ADWIN are common streaming drift detectors; River documents Page-Hinkley for detecting mean changes and ADWIN as an adaptive-window drift detector. 
RiverML
+1

4. Regime drift

Volatility regime.

Market breadth.

Rate regime.

Credit regime.

Liquidity regime.

Correlation regime.

For macro data, use real-time vintages where possible; ALFRED exists specifically to retrieve economic data releases as they were available on historical dates. 
ALFRED

Kill-switch criteria

Even for paper/shadow, define hard stops.

Stop generating/promoting signals if:

Data is stale.

Corporate action adjustment fails.

A vendor feed has unexplained gaps.

Feature pipeline produces out-of-range values.

Universe size changes unexpectedly.

Model scores collapse to a few names.

Gross/net/sector/beta constraints are breached.

Estimated cost > expected edge.

Rolling IC is negative for a predefined window.

Paper drawdown exceeds threshold.

Candidate underperforms incumbent for N consecutive rebalance periods.

Prediction distribution differs materially from training distribution.

Strategy becomes dominated by microcaps/illiquid names.

Backtest and paper implementation disagree materially.

5. Practical 30/60/90-day roadmap
Days 0–30: build the boring system that prevents self-deception

Goal: one clean daily cross-sectional research loop.

Deliverables:

Data lake

Raw append-only Parquet or similar.

DuckDB/SQLite/Postgres metadata catalog.

Vendor request logs.

Payload hashes.

Ingestion timestamps.

External SSD backup and off-machine backup.

Point-in-time schema

symbol_master

security_id

corporate_actions

bars_daily

bars_minute

fundamentals_raw

filings_raw

news_raw

feature_values

feature_available_at

Universe

Start with liquid U.S. equities.

Avoid microcaps.

Require minimum price, ADV, and data completeness.

Be explicit that free data may not be survivorship-free. A backtest using only currently active names can overstate performance because delisted/failed stocks are missing; survivorship-bias-free backtesting requires historical inactive/delisted securities. 
Center for Research in Security Prices
+1

Baseline features

1d/5d/20d/60d/252d returns.

Short-term reversal.

Momentum.

Volatility.

Beta.

Dollar volume.

Liquidity proxy.

Sector-relative versions.

Market-cap bucket.

Basic fundamentals if point-in-time.

Baseline models

Ridge/ElasticNet.

Logistic top/bottom classifier.

LightGBM ranker/regressor.

Simple factor composite.

Backtester

Decision timestamp.

Portfolio optimizer.

Costs.

Turnover.

Exposure constraints.

Random portfolio benchmark.

S&P 500 total-return benchmark where appropriate.

Do not add deep learning yet.

Days 31–60: make it an autonomous research lab

Goal: candidate generation, walk-forward testing, and paper signals.

Deliverables:

Experiment registry

Every trial logged.

Feature version.

Label version.

Cost model version.

Hyperparameter search count.

Git commit.

Walk-forward engine

Rolling train/validation/test.

Purged/embargoed splits where needed.

Incumbent/challenger comparison.

PBO/Deflated Sharpe reporting.

Paper signal service

Daily signal file.

Paper portfolio.

Broker/paper API integration.

Signal ledger.

Fill simulation.

Minute-derived features

Realized volatility.

Intraday return decomposition.

Opening range.

Last-hour return.

Volume profile.

Intraday liquidity/slippage proxy.

Gap features.

Execution-cost estimator.

Monitoring

Data freshness.

Feature drift.

Prediction drift.

Paper PnL.

Rolling IC.

Exposure dashboard.

Alerting.

Ensemble layer

Model averaging.

Rank aggregation.

Confidence calibration.

Simple meta-labeling.

Deep learning still optional. The goal is to get a trustworthy GBDT/linear/ensemble incumbent.

Days 61–90: add text/events and controlled deep challengers

Goal: expand feature space and test whether complexity earns its keep.

Deliverables:

News/event table

Timestamp-safe event ingestion.

Entity mapping.

Relevance scoring.

Sentiment/embedding.

Event-type classification.

Per-stock aggregation.

SEC filing features

EDGAR metadata.

10-K/10-Q/8-K accepted timestamp.

XBRL fundamentals.

Filing section deltas.

Similarity to previous filing.

Risk-factor/MD&A change scores.

Ablation tests

Price-only.

Price + fundamentals.

Price + fundamentals + minute features.

Price + fundamentals + text/events.

Full model.

Controlled deep challengers

MLP/ResNet tabular model.

FT-Transformer-style tabular model.

Small TCN for minute feature extraction.

No RL except maybe contextual bandit for ensemble weighting.

Promotion rules finalized

Incumbent lock.

Challenger quarantine.

Shadow period.

Kill-switches.

Complexity penalty.

By day 90, success is not “we found a money printer.” Success is:

You have a self-running research loop.

It produces daily paper signals.

It rejects bad models automatically.

It can show whether each new data source adds incremental net value.

You know whether the current setup has a real signal or only backtest artifacts.

6. Features worth building next
Highest-value daily features

Build these first:

Momentum and reversal

1-day reversal.

5-day reversal.

20/60/120/252-day momentum.

Skip-month momentum.

Sector-neutral momentum.

Residual momentum.

Volatility and risk

Realized volatility.

Idiosyncratic volatility.

Beta.

Drawdown.

High-low volatility.

Volatility change.

Liquidity

Dollar volume.

ADV.

Amihud illiquidity.

Turnover.

Volume shock.

Spread proxy if no quote data.

Relative valuation / quality / fundamentals

Profitability.

Margins.

Leverage.

Accruals.

Revenue growth.

Earnings growth.

Free cash flow proxy.

Valuation ratios.

Share issuance/buyback.

Fundamental changes, not just levels.

Cross-sectional normalizations

By date.

By sector.

By market-cap bucket.

Winsorized.

Robust z-score.

These align with the empirical result that useful signal families often include price trends, liquidity, volatility, and valuation/fundamental ratios. 
OUP Academic

Best minute-bar features

Use minute bars mainly for risk, timing, and cost, not raw alpha at first.

Build:

Open gap.

Opening range return.

First 30-minute volatility.

Last-hour return.

Close imbalance proxy from last-minute volume/return.

Intraday realized volatility.

Intraday skew.

Volume profile deviation.

VWAP distance.

Overnight vs intraday return split.

Intraday reversal.

Intraday momentum into close.

Liquidity/slippage estimator.

Abnormal volume at time of day.

Avoid feeding thousands of raw minute sequences into a transformer before you know simple aggregated features help.

Best news/event/filing features

Build:

Event count by time window.

Abnormal event volume.

Sentiment score.

Negative-event flag.

Novelty score.

Relevance score.

Source quality.

Event category.

Earnings/guidance flags.

Analyst rating-change flags if available.

SEC filing type.

8-K item type.

Filing accepted timestamp.

MD&A similarity change.

Risk-factor similarity change.

XBRL fact changes.

“First mention in N days” feature.

Peer spillover features after major sector events.

Best macro/regime features

Use sparingly:

Market volatility regime.

S&P trend.

Market breadth.

VIX proxy if available.

Rates trend.

Yield curve.

Credit spread proxy.

Inflation/rate surprise indicators if point-in-time.

Dollar/oil regime if relevant to sectors.

Macro features are usually better for risk budget and model selection than for single-stock daily alpha.

Explicitly avoid

Avoid these until the core loop proves itself:

RL as the main alpha engine.

End-to-end transformer over all stocks and all bars.

GNN without point-in-time graph edges.

Fine-tuning LLMs on tiny financial labels.

Optimizing Sharpe across thousands of trials without multiple-testing correction.

Current-constituent-only historical universes.

Fundamentals keyed by fiscal period instead of availability date.

News keyed by article date instead of ingestion/availability timestamp.

Global normalization fit on the full dataset.

Using adjusted prices inconsistently in intraday simulation.

Ignoring delisted names while claiming historical market-beating performance.

Trading microcaps because they make backtests look good.

High-turnover minute strategies on free data.

Complex models that only beat baselines before costs.

Any model whose edge disappears at 2x estimated transaction costs.

Recommended final architecture
Raspberry Pi data node

Responsibilities:

Vendor polling.

Rate limiting.

Raw payload storage.

Heartbeats.

Basic validation.

Append-only logs.

Data replication to Mac Mini.

No model training.

No strategy logic beyond collection.

Mac Mini research node

Responsibilities:

Data cleaning.

Point-in-time feature generation.

Backtesting.

Model training.

Walk-forward testing.

Experiment registry.

Model registry.

Paper signal generation.

Drift monitoring.

Alerting.

Core services

Collector service

Raw data lake

Point-in-time feature builder

Research scheduler

Backtest engine

Model trainer

Portfolio constructor

Paper trading service

Monitoring/drift service

Incumbent/challenger registry

Final ranking
Rank	Architecture family	Practical priority	Verdict
1	Regularized linear/rank models	Immediate	Baseline, sanity check, leakage detector
2	Gradient boosted trees	Immediate	First serious alpha workhorse
3	Ensembles/meta-labeling	Early	Best way to improve robustness after clean baselines
4	Embeddings + tabular model for news/filings	Early-mid	Best text architecture for your setup
5	Temporal CNN/TCN	Mid	Useful for minute-derived feature extraction
6	Tabular deep learning / FT-Transformer	Mid-late	Worth testing, not first incumbent
7	TFT/sequence transformers	Late	Powerful but high leakage/compute/debug burden
8	GNNs	Late	Only with true point-in-time relationship data
9	Time-series foundation models	Late auxiliary	Use for volatility/regime/features, not core alpha
10	RL policy learning	Latest	Use for allocation/execution after alpha is proven

The most practical path is to become extremely good at rejecting false positives. If the system can do that, the architecture can grow safely. If it cannot, more advanced models will only make the false positives more convincing.

Sources
