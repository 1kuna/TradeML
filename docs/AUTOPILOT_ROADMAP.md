# Autopilot Roadmap

This is the north-star operating model for TradeML. `SSOT.md` remains canonical; this file is the short roadmap for execution order.

## 1. Pi Capture Loop

- Run the data node continuously on the Raspberry Pi.
- Keep the node under systemd with `Restart=always`; a clean-but-unexpected process exit should not leave collection offline.
- Keep canonical EOD integrity first: forward bars, repair tasks, and frozen-window coverage gates.
- Saturate independent quota lanes every poll cycle: Alpaca minute bars, Tiingo/Finnhub news, SEC filings/companyfacts, FRED vintages, and low-rate Massive/Twelve Data fillers.
- Burn down unused daily reserve near reset so current/free entitlements do not idle.
- Persist raw minute/news/event archives indefinitely with watermarks; do not expose them to Phase 1 features until PIT-safe curation exists.

## 2. Mac Mini Research Loop

- Consume only curated PIT-safe tables from the NAS.
- Run deterministic walk-forward experiments with costs always on.
- Compare new architectures against the accepted incumbent using IC stability, decile spread, drawdown, turnover, and cost stress.
- Save every run config, feature list, metric file, and promotion decision under `models/{model_name}/run_{timestamp}/`.
- Run the perpetual supervisor as a macOS LaunchAgent so it restarts after reboot and remains the single owner of the research loop.
- Keep the conventional label `com.trademl.research.perpetual-macmini`; use `trademl research launchd-status --program configs/research/perpetual_macmini.yml` to verify launchd ownership.

### Mac Mini LaunchAgent Runbook

Install or refresh the LaunchAgent from the Mac Mini repo checkout:

```bash
.venv/bin/trademl research \
  --data-root /Users/openclaw/atlas_mounts/dev/TradeML \
  --local-state /Users/openclaw/TradeML/control \
  --env-file .env \
  install-launchd \
  --program configs/research/perpetual_macmini.yml \
  --poll-seconds 60 \
  --python-executable /Users/openclaw/TradeML/.venv/bin/python \
  --load
```

Check ownership and health:

```bash
.venv/bin/trademl research --local-state /Users/openclaw/TradeML/control launchd-status --program configs/research/perpetual_macmini.yml
.venv/bin/trademl research --data-root /Users/openclaw/atlas_mounts/dev/TradeML --local-state /Users/openclaw/TradeML/control --env-file .env health --program-id perpetual-macmini
```

Graceful maintenance stop:

```bash
.venv/bin/trademl research --local-state /Users/openclaw/TradeML/control unload-launchd --program configs/research/perpetual_macmini.yml
.venv/bin/trademl research --data-root /Users/openclaw/atlas_mounts/dev/TradeML --local-state /Users/openclaw/TradeML/control --env-file .env stop --program-id perpetual-macmini
```

If the controller runs from another machine over SSH, do not bake a password into config. Set the configured password env var in the calling environment, or use key auth.

## 3. Drift And Health

- Alert when recent live or retrain metrics fall materially below the incumbent's accepted OOS envelope.
- Track feature-distribution drift, GREEN/AMBER/RED coverage drift, vendor 429s, entitlement failures, rows per credit, and idle quota.
- Treat drift as a research signal, not an automatic trading permission change.

## 4. Incumbent Promotion

- Phase 1 promotion remains manual.
- A challenger must pass leakage checks, replayable walk-forward evidence, placebo tests, and cost stress before replacing the incumbent.
- No model can be promoted from raw archive-only lanes until those lanes have PIT-safe curated outputs.
