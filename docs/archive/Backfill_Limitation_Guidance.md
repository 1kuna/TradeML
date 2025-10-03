Short version: don’t chase “oldest possible.” Set cutoffs by regime and use recent, high-fidelity history for training, keeping older eras for stress tests. Markets changed structurally with decimalization (2001), Reg NMS (2005–2007 rollout), LULD (2012+), the Tick Size Pilot (2016–2018), and likely half-penny quoting (effective 2025). Data from before those breakpoints won’t map cleanly to today’s microstructure and execution realities.  ￼

Here’s the pragmatic policy I’d bake into the blueprint:

How far back to collect (and how much to train on)

Dataset	Why not go too far back	Collect back to	Train on	Use older history for…
Equities EOD OHLCV	Daily returns are fairly comparable across regimes; older helps factor validation	1990+ (earlier if easy)	10–15y rolling with time-decay	Long-cycle stress tests (2000–02, 2008, 2020, 2022)
Equities minute bars	Microstructure shifts: decimalization (2001), Reg NMS (2007), LULD (2012) materially change intraday behavior.  ￼	2010+	5–10y	Sanity checks only
Consolidated equities ticks (SIP)	Massive size; regime shifts make pre-2010 less comparable	2012+ (post-LULD)	2–4y (microstructure models)	Execution tests, not model fit
IEX-only ticks	Single venue; IEX is new in modern terms (exchange since 2016)  ￼	2016+	2–3y	Feature protos/LOB research
Options chains (EOD)	Market structure matured with Weeklys (2005→); coverage improves after 2010.  ￼	2012–2015+ (as available)	7–10y	Stress (2018 volmageddon, 2020 crash)
Options NBBO (OPRA)	Enormous; vendors offer long archives but storage/cleaning heavy.  ￼	2010–2012+ (practical)	5–8y (delta-hedged/vol)	Pathological days & execution sims
Corporate actions & delistings	Needed for survivorship-free backtests	As far back as prices	N/A	Always
Index membership history	Needed for factor/risk context	As far back as obtainable	N/A	Benchmarking & risk attributions
Macro (FRED/ALFRED vintages)	Revisions matter; vintages are cheap to store	Full series	Whatever your model actually uses	Regime labeling, priors
SEC/EDGAR filings	Electronic filings go back to the 1990s; good for PIT events	1994+	Model-dependent	Event studies

Why this split works
	•	Structural breaks are real. Decimalization changed ticks (stocks to $0.01, many options to $0.05–$0.10); Reg NMS rewired routing/pricing; LULD altered intraday dynamics; Tick Size Pilot briefly widened ticks; and half-penny quoting looks slated for Nov 2025, which will shift microstructure again. Training too far back blends incompatible regimes and degrades out-of-sample behavior.  ￼
	•	Effective sample size beats raw years. For intraday/tick models, 2–4 recent years give millions of events under today’s rules; older events inflate sample size but not relevance.
	•	Options evolved. Weeklys (launched 2005) and the post-2010 expansion changed surface dynamics and execution. A 7–10 year modern window typically captures multiple vol regimes without dragging in pre-modern frictions.  ￼

Implementable blueprint knobs (drop-in)

Add to backfill.yml:

horizons:
  equities_eod:
    collect_earliest: 1990-01-01
    train_window_years: 15
  equities_minute:
    collect_earliest: 2010-01-01
    train_window_years: 10
  equities_ticks:
    collect_earliest: 2012-06-01   # post-LULD rollout
    train_window_years: 4
  options_chains:
    collect_earliest: 2012-01-01
    train_window_years: 10
  options_nbbo:
    collect_earliest: 2010-01-01
    train_window_years: 8

And in training configs:

training:
  sample_weight: time_decay(lambda_half_life_months: 18)
  regime_masks:
    - { name: pre-2012,  use_for: "stress_only" }
    - { name: 2012-2016, use_for: "fit_and_validate" }
    - { name: 2016-2024, use_for: "fit_and_validate" }
    - { name: 2025+,     use_for: "monitor_for_break" }  # half-penny regime

Bottom line
	•	Collect broadly, but train narrowly on the modern era with time-decay and CPCV; keep older eras for stress tests and priors.
	•	If storage or rate limits bite, cut first from pre-2010 intraday/tick and pre-2012 options, not from the last decade.
	•	Revisit the cutoffs again in late-2025 once half-penny quoting is in force and you’ve logged live slippage under the new regime.  ￼