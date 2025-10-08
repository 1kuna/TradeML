# TradeML — Model Architecture (Hybrid Stacked + Offline‑RL) — Legacy Carve‑Out Plan

**Scope:** Implement a clean, production‑grade modeling stack while keeping data collection unchanged. We **carve the current modeling code into `/legacy/` (frozen)** and ship the new pipeline under the **base names** (no “v2” suffixes). This avoids config‑flag spaghetti, enables instant rollback, and lets us delete `/legacy/` after a stable burn‑in.

---

## 0) Executive summary
- **Alpha core:** daily cross‑sectional **LightGBM/CatBoost** trained under **CPCV** with time‑decay + regime masks.
- **Additive sleeves (optional):**
  - **Intraday_xs:** PatchTST or Mamba (small) for minute streams; contributes only where coverage is GREEN.
  - **Options_vol:** SVI/SSVI features + delta‑hedged edges.
  - **Risk sleeve:** multi‑horizon realized‑vol / drawdown forecasts.
- **Meta‑blender:** CPCV‑consistent **stacker** over OOF predictions from sleeves.
- **Portfolio & costs:** z‑score → caps → fractional‑Kelly; fees + bid/ask + square‑root impact; borrow for shorts.
- **Execution & RL:** keep alpha supervised; use **offline RL** (CQL/IQL; DT optional) for *position sizing and schedule selection* only.
- **Governance:** CPCV + DSR + PBO + reliability curves; **Champion/Challenger** with MLflow; shadow runs before promotion.
- **Pi:** either runs daily scoring from ONNX/PKL or just orchestrates; training remains on GPU host.

---

## 1) Target repo layout (legacy carve‑out)
```
/repo
  /legacy/                   # frozen current MODELING code; no new commits
    ...                      # move old /models, /ops/pipelines, /portfolio bits here
  /data_layer                # unchanged (raw/curated/reference/qc)
  /feature_store
    equities/                # richer daily features & registries
    options/
    intraday/
  /labeling
    horizon/                 # k‑day forward returns
    triple_barrier/          # TP/SL/T labels
  /validation                # CPCV, DSR, PBO, calibration
  /models
    equities_xs/             # LightGBM/CatBoost + export → ONNX/PKL
    intraday_xs/             # PatchTST or Mamba (small)
    options_vol/             # SVI → features → learners
    meta/                    # stacker (linear/LGBM) + calibration utils
  /portfolio
    build.py                 # vol targeting, caps, fractional‑Kelly, risk scaling
  /execution
    cost_models/             # spread, impact (√‑law), borrow
    simulators/              # Almgren‑Chriss wrappers
    brokers/
      alpaca_client.py       # paper/live order client
  /backtest
    engine/                  # event‑driven backtester (extend hooks only)
  /ops
    pipelines/
      equities_xs.py         # end‑to‑end daily pipeline
      intraday_xs.py
      options_vol.py
      stacker_train.py       # trains meta‑blender on OOF predictions
      offline_rl.py          # RL env + trainers + OPE
    ssot/
      train_gate.py          # add new model names; legacy disabled
      router.py              # loads stacker model/weights; single switch
    reports/
      emitter.py             # daily JSON/MD + reliability PNGs
  /configs
    training/
      equities_xs.yml
      intraday_xs.yml
      options_vol.yml
      stacker.yml
      rl.yml
    router.yml               # enable stacker + defaults
  /scripts
    train_equities.py
    score_daily.py
    export_onnx.py
    paper_trade_alpaca.py
    build_universe.py
```
> **Rule:** only the modeling stack moves to `/legacy/`. **Data collection stays unchanged.** If needed, expose a lightweight `curated_view` selector (read‑only) for added metadata (e.g., `calendar_id`, `coverage_ratio`, options QC flags).

---

## 2) Data contracts (what each model expects)
### 2.1 Equities_xs (daily)
- **Inputs:** PIT‑safe adjusted OHLCV; aligned to exchange calendar; declared universe.
- **Features:** momentum {5,20,60,126}d; seasonality (DOW sin/cos); roll‑z; RV{20,60}; ADV{20}; price‑to‑bands; gap stats; drawdown; optional sector/size, beta.
- **Labels:** `horizon` (k‑day fwd return, default k=5) or `triple_barrier` (tp=2σ, sl=1σ, max_h=10, vol_window=20).
- **CPCV groups:** `{date, symbol}` with symbol‑aware purging + 10‑day embargo.

### 2.2 Intraday_xs (optional)
- **Inputs:** minute bars + LOB/TOB summaries for a liquid subset.
- **Features:** VWAP dislocation, OFI, rolling RV, microstructure frictions, auction flags.
- **Model:** PatchTST or Mamba (small). Train per symbol/cluster; aggregate to a **daily** score (last N minutes pre‑close).

### 2.3 Options_vol
- **Inputs:** underlier OHLCV + options chains (NBBO), IV per contract, daily SVI/SSVI fits with QC flags.
- **Targets:** delta‑hedged P&L (vol sleeve) and/or edge probability for spread templates (calendars/verticals).

### 2.4 Risk sleeve (optional)
- Multi‑horizon realized‑vol / drawdown; feeds portfolio scaling.

---

## 3) Model set
### 3.1 Equities_xs — LightGBM/CatBoost core
- **Task:** regression on forward return (start), or top/bottom‑k classification.
- **Training:** CPCV (8 folds, 10‑day embargo); time‑decay (half‑life months); regime masks.
- **HPO:** Optuna small grid. LGBM params: `num_leaves {31,63,127}`, `max_depth {−1,7,11}`, `lr {0.01,0.05,0.1}`, `min_data_in_leaf {50,200}`, `colsample {0.6,0.9}`, `subsample {0.6,0.9}`, `l1/l2 {0,10}`. CatBoost analogs.
- **Artifacts:** `model.pkl` + `model.onnx` + `feature_list.json` + `training_cfg.json` + OOF predictions.

### 3.2 Intraday_xs — PatchTST/Mamba sleeve (optional)
- Context length 256–512; patch 16–32; dropout 0.1–0.3; TorchScript/ONNX export. Output daily per‑symbol score.

### 3.3 Options_vol — SVI features → tabular learner
- Train on underliers with GREEN options coverage; targets = delta‑hedged edge or P(PnL>0). Output underlier‑day score + candidate structures with Vega caps.

### 3.4 Meta‑blender — CPCV‑consistent stacker
- Concatenate OOF predictions from enabled sleeves; align by `{date, symbol}`.
- Start ridge/LightGBM; persist `stacker.pkl` or `stacker_weights.json`. If a sleeve is missing, stacker uses available inputs; otherwise fallback to weighted average.

---

## 4) Portfolio & execution
- **Policy:** rank → z‑score → cap per name (e.g., 5%) → gross cap 1.0 → fractional‑Kelly (0.5–1.0).
- **Risk‑aware scaling:** modulate by forecast risk & sleeve confidence.
- **Costs:** fees=0 baseline; spread default 5 bps; impact via square‑root; borrow 50–200 bps annualized tiers.
- **Backtest hooks:** add borrow carry + optional auction slippage; keep engine deterministic.
- **Execution simulator:** Almgren‑Chriss frontier for urgency vs cost.

---

## 5) Offline RL for sizing & scheduling (optional)
- **Env:** `MarketEnv` wrapping backtester.
  - **State:** recent factor z‑scores, realized vol, drawdown, crowding/turnover, cost stats.
  - **Action:** weight bucket (−5%…+5%) **or** participation‑rate bucket from AC frontier.
  - **Reward:** next‑day **net** return with cost/DD penalties.
- **Dataset:** trajectories from the supervised baseline policy.
- **Algorithms:** CQL / IQL. Decision‑Transformer as challenger only.
- **OPE:** doubly‑robust; accept only if ≥ baseline + margin.
- **Integration:** RL modulates weights/schedules **after** the alpha stacker.

---

## 6) Paper trading & Raspberry Pi
- **Broker client:** `execution/brokers/alpaca_client.py` (submit/cancel/status); converts target weights → orders (market/open or TWAP via AC schedule).
- **Pi modes:** (A) local daily scoring from ONNX/PKL; (B) orchestrate only and pull `equities_YYYY‑MM‑DD.json` from S3/MinIO.
- **Tripwires:** halt on missing GREEN, extreme turnover, abnormal drift; blotter reconciliation each morning.

---

## 7) Implementation guide — file by file
### 7.1 Features & labels
- `feature_store/equities/features.py`
  - `build_features(panel: pd.DataFrame, cfg: dict) -> pd.DataFrame`
  - PIT rolling transforms; momentum/vol/liquidity/seasonality/drawdown.
- `labeling/horizon/build.py`: `make_horizon_labels(panel, k: int)`
- `labeling/triple_barrier/build.py`: `make_tb_labels(panel, tp_sigma, sl_sigma, max_h, vol_window)`

### 7.2 Models — Equities_xs
- `models/equities_xs/lgbm.py`
  - `fit_lgbm(X, y, sample_weight=None, params: dict) -> (booster, metrics)`
  - `predict_lgbm(model, X) -> np.ndarray`
  - `export_onnx(model, feature_names, path)`
- `models/equities_xs/catboost.py` (same contract)
- `models/meta/stacker.py`
  - `train_stacker(oof_df, y, cfg) -> artifact_paths`
  - `load_stacker(path) -> model`
  - `stack_scores(df_scores, model_or_weights) -> np.ndarray`

### 7.3 Pipelines
- `ops/pipelines/equities_xs.py`
  - `PipelineCfg(start,end,universe,label_type,horizon_days,tp_sigma,sl_sigma,max_h,vol_window,n_folds,embargo_days,initial_capital,spread_bps,gross_cap,max_name,kelly_fraction)`
  - Steps: Load curated → build features/labels → CPCV fit (LGBM default) → OOF → portfolio → backtest → DSR/PBO → reliability (if classification) → emit daily → persist artifacts.
- `ops/pipelines/stacker_train.py`: build OOF dataset from sleeves; train stacker; write `configs/router.yml` pointers and `models/meta/stacker.pkl`.
- `ops/ssot/router.py`: `if stacker.enabled: use stacker; else weighted average`.
- `ops/pipelines/options_vol.py`: SVI fit → features → learner for delta‑hedged edge; emit strategies JSON with Vega caps.
- `ops/pipelines/offline_rl.py`: `MarketEnv`, dataset logger, CQL/IQL trainers, OPE, export policy.

### 7.4 Portfolio, execution & backtest
- `portfolio/build.py`: `build(scores_df, risk_cfg) -> target_weights_df`
- `execution/brokers/alpaca_client.py`: `submit_orders(asof, target_weights, policy_cfg)`; supports TWAP via AC schedule.
- `backtest/engine`: extend with borrow carry & auction slippage hooks.

### 7.5 Validation & governance
- `validation/cpcv` (reuse core); ensure symbol‑aware purging + 10‑day embargo.
- `validation/pbo.py`, `validation/dsr.py` (reuse).
- `validation/calibration.py`: Brier/logloss & reliability bins; plot to `ops/reports/`.
- **Promotion workflow:** `ops/ssot/train_gate.py` logs MLflow runs with `{Challenger, Champion}` tags; `promote_if_beat_champion()` applies bars.

### 7.6 Configs
- `configs/training/equities_xs.yml` — dependencies, GREEN thresholds, CPCV, time‑decay, regime masks, HPO ranges, promotion bars.
- `configs/router.yml` — `stacker.enabled: true`, fallback weights, `stacker.pkl` path.
- `configs/rl.yml` — action buckets, reward weights, OPE thresholds.

### 7.7 CLIs & scripts
- `scripts/train_equities.py` — `--start --end --symbols ... --label horizon|tb --k 5 --embargo 10 --folds 8`
- `scripts/score_daily.py` — loads pkl/onnx; writes `ops/reports/equities_YYYY‑MM‑DD.json|md`.
- `scripts/export_onnx.py` — converts LGBM/CatBoost to ONNX.
- `scripts/paper_trade_alpaca.py` — translates target weights → orders; posts; writes blotter.

### 7.8 Tests & acceptance
- **Unit:** PIT feature builders, labelers, portfolio math, cost models, CPCV split logic, calibration bins.
- **Integration:** small‑universe E2E; assert shapes + sane metric ranges.
- **E2E acceptance cases:**
  1) Delete a week of symbols → ingest audit RED → backfill → curated GREEN → training unblocks.
  2) Vendor swap for minute data (same schema) → metrics comparable.
  3) Promotion only if DSR/PBO/Sharpe bars met; challenger > champion; optional shadow PnL ≥ 0.

---

## 8) Rollout plan (PR‑by‑PR)
**PR 0 — Legacy carve‑out**
- Move current **modeling** code to `/legacy/`.
- CI guardrail: disallow imports from `/legacy/` outside tests; optional CODEOWNERS.
- Router default remains baseline until PR 1 merges.

**PR 1 — Equities core**
- Add `models/equities_xs`, `feature_store/equities`, `labeling/*`, `ops/pipelines/equities_xs.py`, configs.

**PR 2 — Meta‑blender**
- Add `models/meta/stacker.py`, `ops/pipelines/stacker_train.py`; enable in `configs/router.yml`; update `ops/ssot/router.py`.

**PR 3 — Portfolio/Backtest polish**
- Add `portfolio/build.py` (Kelly, caps, risk scaling); borrow carry & slippage hooks.

**PR 4 — Options**
- Unify IV/SVI under `options_vol`; add minimal options backtester; reports.

**PR 5 — Offline RL (optional)**
- `ops/pipelines/offline_rl.py` (env, dataset logger, CQL/IQL trainer, OPE), policy registry.

**PR 6 — Paper trading**
- `execution/brokers/alpaca_client.py`, `scripts/paper_trade_alpaca.py`, morning blotter sync.

**PR 7 — Orchestration & promotion**
- Train‑gate routing, MLflow logging improvements, reliability PNGs in reports.

**Kill switch**
- After N stable weeks of paper/live, delete `/legacy/` in a single PR.

---

## 9) Acceptance criteria (promotion to Champion)
- **Equities sleeve:** Sharpe ≥ 1.0; MaxDD ≤ 20%; DSR > 0; PBO ≤ 5% (over small HPO); turnover within policy; realistic costs.
- **Meta‑stack:** Outperforms best single sleeve OOS without worse PBO/DSR; reliability not worse.
- **Options sleeve:** Positive delta‑hedged PnL OOS with Vega exposure within bounds; costs modeled.
- **RL policy (if used):** OPE ≥ baseline + margin; backtest uplift ≥ X%; trips clean.

---

## 10) Operability notes
- **PIT safety everywhere**; align features at entry date.
- **Symbol‑aware purging + embargo** in all CPCV jobs.
- Persist feature lists with models; runtime uses exact order.
- Stacker requires fold‑consistent OOF alignment across sleeves.
- Broker creds read‑only; paper/live opt‑in via `.env`.

---

### Appendix A — minimal function stubs
```python
# models/equities_xs/lgbm.py
@dataclass
class LGBMParams:
    num_leaves:int=63; max_depth:int=-1; learning_rate:float=0.05
    min_data_in_leaf:int=200; colsample_bytree:float=0.8
    subsample:float=0.8; lambda_l1:float=0.0; lambda_l2:float=0.0

def fit_lgbm(X:pd.DataFrame, y:pd.Series, sample_weight=None, params:LGBMParams=LGBMParams()):
    ...

def predict_lgbm(model, X:pd.DataFrame)->np.ndarray:
    ...

def export_onnx(model, feature_names:list, path:str)->None:
    ...
```
```python
# models/meta/stacker.py
def train_stacker(oof_df:pd.DataFrame, y:pd.Series, cfg:dict)->dict: ...
def load_stacker(path:str): ...
def stack_scores(df_scores:pd.DataFrame, model_or_weights)->np.ndarray: ...
```

---

**Deliverable:** After PRs 1–3 you’ll have daily ranked picks with calibrated scores, realistic costs, and strict anti‑overfitting governance. PRs 4–6 extend to options and RL and wire paper trading. `/legacy/` is a clean rollback and later a clean delete.

