# BTC/USDC Trading Strategy — Technical Specification

> **Version**: 2.0  |  **Stack**: Python 3.10+, pandas, numpy, Optuna  |  **Backtest**: 2016-02-01 → 2026-02-01

---

## Table of Contents

1. [Overview](#1-overview)
2. [System Architecture](#2-system-architecture)
3. [Production Cost Model](#3-production-cost-model)
4. [Signal Generation](#4-signal-generation)
5. [Overlay Filters](#5-overlay-filters)
6. [Risk Management](#6-risk-management)
7. [Portfolio Execution](#7-portfolio-execution)
8. [Optimization Framework](#8-optimization-framework)
9. [Walk-Forward Validation](#9-walk-forward-validation)
10. [Configuration Reference](#10-configuration-reference)

---

## 1. Overview

Single-pair BTC/USDC strategy. Continuous position sizing [0, 1] driven by a mining production-cost
fundamental signal, modulated by four overlay filters, and hard-capped by a four-mode risk FSM.
Daily rebalance with fee-aware step limiting and a dead-band guard.

**Pipeline summary:**

```
fundamental_signal
  → RSI boost
  → hash-ribbon cap
  → vol-target scaling
  → min(w_signal, mode_cap, trend_cap)   ← simultaneous hard caps
  → step-limited execution with dead-band
```

Parameter search: 3-phase sequential Optuna optimization, objective = mean OOS Sortino over 5
walk-forward folds. Hands-off test set: 2025-07-01 onward (never seen during search).

---

## 2. System Architecture

```
DataFetcher
  ├── CoinGecko REST  (daily BTC close)
  └── Hashrate source (daily TH/s)
        │
        ▼  combined_data_{N}d.csv  (cached on disk)
        │
        ├──► ProductionCostModel  →  daily_cost_usd_per_btc
        │
        └──►  BacktestEngine
                │
                ├── SignalGenerator        price_ema / cost_ema →  w_base [0,1]
                ├── RSI overlay            w_base × RSI_BOOST if RSI14 < RSI_OVERSOLD
                ├── HashRibbon overlay     w × CAP_MULT if fast_ema < slow_ema
                ├── VolScaling overlay     w × clamp(VOL_TARGET / σ_btc, 0.20, 1.00)
                ├── RiskManager  (FSM)     mode_cap ∈ {0.05, 0.45, 0.75, 1.00}
                ├── TrendFilter            trend_cap = 0.0 if price < EWM(250) else 1.0
                │
                └── w_final = min(w_signal, mode_cap, trend_cap)
                        │
                        └── Portfolio  (step ≤0.10/day, dead-band 0.04, fee 0.1%)
```

All parameters are class attributes in `config.py`. No hardcoded values elsewhere.
`optuna_optimizer.py` temporarily monkeypatches these attributes per trial via `_patched_config()`.

---

## 3. Production Cost Model

### 3.1 Formula

```
cost_raw = (energy_price_usd_kwh
            × efficiency_J_per_TH
            × hashrate_TH_per_s
            × 86_400)                          # TH/s → TH/day
          / (BLOCKS_PER_DAY × btc_subsidy)     # 144 × 3.125 BTC
          × OVERHEAD_FACTOR                    # 1.47
```

`btc_subsidy` reflects the current halving epoch (50 → 25 → 12.5 → 6.25 → 3.125).
`OVERHEAD_FACTOR = 1.47` captures CapEx amortization, cooling, and facilities on top of
pure energy cost.

### 3.2 Two-Stage EMA Smoothing

Raw daily cost is smoothed twice before the signal sees it:

| Pass | Location | Config key | Span |
|------|----------|-----------|-----:|
| 1st  | `production_cost.py` | `ProductionCostConfig.EMA_WINDOW` | 14 |
| 2nd  | `backtest.py` | `SignalConfig.COST_EMA_WINDOW` | 30 |

Effective group delay ≈ 14 + 30 = 44 trading days.

### 3.3 Electricity Price & Miner Efficiency Lookup

Year-indexed constants; `production_cost.py` picks the row matching `date.year`.

| Year | $/kWh | J/TH |
|-----:|------:|-----:|
| 2016 | 0.100 | 250  |
| 2017 | 0.100 | 150  |
| 2018 | 0.080 | 110  |
| 2019 | 0.070 |  85  |
| 2020 | 0.065 |  60  |
| 2021 | 0.065 |  50  |
| 2022 | 0.070 |  45  |
| 2023 | 0.065 |  38  |
| 2024 | 0.062 |  30  |
| 2025 | 0.060 |  24  |
| 2026 | 0.060 |  22  |

---

## 4. Signal Generation

### 4.1 Price/Cost Ratio

```python
price_ema = EMA(btc_price,        span=PRICE_EMA_WINDOW)   # 21
cost_ema  = EMA(smoothed_cost,    span=COST_EMA_WINDOW)    # 30  (2nd smoothing pass)
ratio     = price_ema / cost_ema
signal    = EMA(ratio,            span=SIGNAL_EMA_WINDOW)  # 14
```

`signal` is the input to the position table lookup.

### 4.2 Position Table

Step-wise mapping from `signal` to base weight `w_base`:

| signal range | w_base |
|:------------:|:------:|
| [0.00, 0.80) | 1.00   |
| [0.80, 0.90) | 0.85   |
| [0.90, 1.00) | 0.70   |
| [1.00, 1.10) | 0.50   |
| [1.10, 1.25) | 0.30   |
| [1.25, ∞)    | 0.00   |

---

## 5. Overlay Filters

Applied in-order in the backtest loop: RSI → Hash Ribbon → Vol Scaling.
The trend-filter cap and risk-mode cap are then resolved in a single `min()`.

### 5.1 RSI Oversold Boost

```python
rsi = RSI(close, window=RSI_WINDOW)          # 14-period Wilder RSI
if rsi < RSI_OVERSOLD:                       # threshold = 30
    w = min(w * RSI_BOOST, 1.0)             # multiplier = 1.30; clamped to [0,1]
```

### 5.2 Hash Ribbon

```python
fast = EMA(hashrate, span=HASH_RIBBON_FAST)  # 30
slow = EMA(hashrate, span=HASH_RIBBON_SLOW)  # 60

if fast < slow:                              # miner-stress / capitulation regime
    w *= HASH_RIBBON_CAP_MULT               # default 0.00; optimized best 0.10
```

`HASH_RIBBON_ENABLED = True` gates the entire block. When `fast >= slow`, `w` is unchanged.

### 5.3 Volatility Targeting

```python
_btc_returns: deque[float]                      # ring buffer, maxlen=VOL_SCALING_WINDOW (15)
_btc_returns.append(btc_daily_pct_return)       # BTC price returns, not NAV

σ   = np.std(_btc_returns, ddof=0) * sqrt(252) # population σ, annualized
k   = np.clip(VOL_TARGET / σ,                  # VOL_TARGET = 0.40
              VOL_SCALE_MIN,                    # 0.20
              VOL_SCALE_MAX)                    # 1.00  → no leverage
w  *= k
```

Scalar `k ≤ 1` always; strategy scales exposure down but never leverages up.
`ddof=0` throughout — consistent with the risk manager (§6.4).

### 5.4 Hard Caps — Trend Filter & Risk Mode (simultaneous)

```python
trend_ema = btc_price.ewm(span=TREND_FILTER_WINDOW, adjust=False).mean()  # span=250
trend_cap = TREND_BEAR_CAP if price < trend_ema else 1.0                  # 0.0 if bearish

mode_cap  = risk_manager.mode_cap()   # float ∈ {0.05, 0.45, 0.75, 1.00}

w_final_target = min(w_signal, mode_cap, trend_cap)
```

Both caps resolve in one `min()` — execution order is irrelevant; the binding cap wins.
`TREND_FILTER_WINDOW` uses exponential weighting (`ewm(span=...)`), **not** a simple rolling MA.
`TREND_BEAR_CAP = 0.0` → full cash whenever `price < trend_ema`, regardless of signal or mode.

---

## 6. Risk Management

### 6.1 State Machine

Four modes; transitions driven by evaluating three independent metrics each day.

| Mode | `mode_cap` |
|------|:----------:|
| `NORMAL`    | 1.00 |
| `CAUTION`   | 0.75 |
| `RISK_OFF`  | 0.45 |
| `EMERGENCY` | 0.05 |

### 6.2 Entry Thresholds

```python
_TRIGGERS = {
    "CAUTION":   {dd: -0.12, vol: 0.75, var: 0.04},
    "RISK_OFF":  {dd: -0.20, vol: 1.00, var: 0.06},
    "EMERGENCY": {dd: -0.35, vol: 1.40, var: 0.09},
}
```

**Escalation**: each metric independently maps the current value to the worst mode it triggers.
`entry_mode` = max severity across all three metrics. If `_SEVERITY[entry_mode] > _SEVERITY[current_mode]`:
`current_mode = entry_mode` — direct jump, no intermediate state traversal.

### 6.3 Exit / Recovery Thresholds

```python
_RECOVERY = {
    "CAUTION":   {dd: -0.09, vol: 0.65, var: 0.025},
    "RISK_OFF":  {dd: -0.16, vol: 0.85, var: 0.040},
    "EMERGENCY": {dd: -0.28, vol: 1.20, var: 0.060},
}
RECOVERY_DAYS = {"CAUTION": 2, "RISK_OFF": 3, "EMERGENCY": 5}
```

Recovery logic:
- `recovery_counter` increments only when **all three** conditions are below their thresholds simultaneously.
- Any single breach resets `recovery_counter = 0`.
- On `recovery_counter >= RECOVERY_DAYS[current_mode]`: step up **one level only**
  (EMERGENCY→RISK_OFF, RISK_OFF→CAUTION, CAUTION→NORMAL). Counter resets after transition.

### 6.4 Metric Computation

```python
# RiskManager._update() computes from trailing NAV returns (not BTC price returns)
vol = np.std(nav_returns[-VOLATILITY_WINDOW:],  ddof=0) * sqrt(252)   # 30-day, pop. σ
var = abs(np.percentile(nav_returns[-VAR_LOOKBACK:], 1.0))             # 99% VaR, 30-day
dd  = (nav[-1] - max(nav[-ROLLING_PEAK_WINDOW:])) / max(nav[-ROLLING_PEAK_WINDOW:])  # 252-day peak
```

`ddof=0` (population σ) used for both `vol` and `var`. NAV-based, distinct from the
BTC-price-return ring buffer used in the vol-targeting overlay (§5.3).

---

## 7. Portfolio Execution

### 7.1 Step Limiter

```python
delta = w_final_target - w_current
step  = np.sign(delta) * min(abs(delta), MAX_DAILY_WEIGHT_CHANGE)   # 0.10
w_new = w_current + step
```

Full repositioning from 0% → 100% requires ≥10 trading days.

### 7.2 Dead-Band Guard

```python
if abs(delta) >= MIN_REBALANCE_THRESHOLD:   # 0.04
    trade(); w_current += step; apply_fee()
# else: no-op, no fee
```

Prevents fee drag from sub-threshold signal noise.

### 7.3 Fee Model

```
fee = trade_value × TRADING_FEES_PERCENT   # 0.001 (10 bps)
```

Applied symmetrically on both legs (buy and sell).

### 7.4 Initial State

```python
initial_usdc = INITIAL_PORTFOLIO_USD     # 100_000
initial_btc  = 2.0
backtest_range = ("2016-02-01", "2026-02-01")
```

---

## 8. Optimization Framework

### 8.1 Three-Phase Sequential Search

Optuna TPE sampler. Each phase holds prior phases' best params fixed; current phase tunes
its own group. Studies persisted as SQLite (`optimization/study_phase_{a,b,c}.db`).

| Phase | Group | Params | Trials |
|-------|-------|--------|:------:|
| A | Core signal | `PRICE_EMA_WINDOW`, `COST_EMA_WINDOW`, `SIGNAL_EMA_WINDOW`, `TREND_FILTER_WINDOW`, `RSI_OVERSOLD`, `VOL_TARGET` | ~45 |
| B | Risk thresholds | `DD_THRESHOLDS` ×3, `VOL_THRESHOLDS` ×3, `VAR_THRESHOLDS` ×3 | ~35 |
| C | Overlay cap | `HASH_RIBBON_CAP_MULT` | ~20 |

Phase B samples three unconstrained floats per metric then sorts them to enforce
`CAUTION < RISK_OFF < EMERGENCY` ordering without distribution-inconsistency warnings in TPE.

### 8.2 Objective Function

```python
sortinos: list[float]   # one per fold; failed fold → -5.0

mean_s  = np.mean(sortinos)
std_s   = np.std(sortinos)
worst   = min(sortinos)

objective = mean_s
          - max(0.0, -worst) * 0.30   # penalise tail folds
          - std_s            * 0.15   # penalise cross-fold variance
```

### 8.3 Phase A Hyperparameter Bounds

| Parameter | Low | High | Step |
|-----------|----:|-----:|-----:|
| `PRICE_EMA_WINDOW` | 7 | 50 | 1 |
| `COST_EMA_WINDOW` | 10 | 60 | 1 |
| `SIGNAL_EMA_WINDOW` | 3 | 30 | 1 |
| `TREND_FILTER_WINDOW` | 100 | 400 | 10 |
| `RSI_OVERSOLD` | 20 | 40 | 1 |
| `VOL_TARGET` | 0.20 | 0.80 | 0.05 |

### 8.4 Best Params (`optimization/best_params.json`)

Mean OOS Sortino = **2.44** across 5 folds.

| Parameter | `config.py` default | Best |
|-----------|--------------------:|-----:|
| `PRICE_EMA_WINDOW` | 21 | **28** |
| `COST_EMA_WINDOW` | 30 | **46** |
| `SIGNAL_EMA_WINDOW` | 14 | **3** |
| `TREND_FILTER_WINDOW` | 250 | **360** |
| `RSI_OVERSOLD` | 30 | **28** |
| `VOL_TARGET` | 0.40 | **0.40** |
| `DD_THRESHOLDS` (C / RO / EM) | −12% / −20% / −35% | **−6% / −36% / −53%** |
| `VOL_THRESHOLDS` (C / RO / EM) | 0.75 / 1.00 / 1.40 | **1.30 / 2.20 / 2.30** |
| `VAR_THRESHOLDS` (C / RO / EM) | 4% / 6% / 9% | **4% / 6% / 9%** |
| `HASH_RIBBON_CAP_MULT` | 0.00 | **0.10** |

Notable: risk thresholds shifted substantially — CAUTION DD trigger loosened (−6 % vs −12 %)
and VOL thresholds across all modes roughly doubled, indicating default thresholds were too
sensitive (high false-positive rate in choppy regimes). VAR thresholds were not improved from
defaults by the optimizer.

---

## 9. Walk-Forward Validation

### 9.1 Fold Definitions

Expanding train window, fixed 1-year OOS per fold (except F5 which covers ~18 months):

| Fold | Train | OOS |
|------|-------|-----|
| F1 | 2017-01-01 → 2019-12-31 | 2020-01-01 → 2020-12-31 |
| F2 | 2017-01-01 → 2020-12-31 | 2021-01-01 → 2021-12-31 |
| F3 | 2017-01-01 → 2021-12-31 | 2022-01-01 → 2022-12-31 |
| F4 | 2017-01-01 → 2022-12-31 | 2023-01-01 → 2023-12-31 |
| F5 | 2017-01-01 → 2023-12-31 | 2024-01-01 → 2025-06-30 |

Defined as `Fold` dataclasses in `optimization/walk_forward.py`.

### 9.2 Indicator Warm-Up

`run_fold()` passes the engine data from `train_start` through `oos_end`. This means even
long-window indicators (EWM-360 in the optimized config) are converged before the OOS slice.
Metrics are extracted only from the OOS slice using a date mask.

### 9.3 Held-Out Test Set

```python
TEST_FOLD = Fold("test_oos", "2017-01-01", "2025-06-30", "2025-07-01", "2026-12-31")
```

Never referenced in `optuna_optimizer.py`. Reserved for final out-of-sample evaluation after
all parameter decisions are frozen.

---

## 10. Configuration Reference

`config.py` — all parameters as class attributes; no magic numbers in other modules.

### `SignalConfig`

| Attribute | Default | Notes |
|-----------|--------:|-------|
| `PRICE_EMA_WINDOW` | 21 | EMA span on BTC daily close |
| `COST_EMA_WINDOW` | 30 | 2nd-pass EMA on smoothed production cost |
| `SIGNAL_EMA_WINDOW` | 14 | EMA span on price/cost ratio |
| `POSITION_TABLE` | see §4.2 | `list[tuple[float, float, float]]` — (lo, hi, weight) |

### `PortfolioConfig`

| Attribute | Default | Notes |
|-----------|--------:|-------|
| `TREND_FILTER_WINDOW` | 250 | `ewm(span=...)` — exponential, not rolling mean |
| `TREND_BEAR_CAP` | 0.00 | Cap applied when `price < trend_ema` |
| `RSI_WINDOW` | 14 | Wilder RSI window |
| `RSI_OVERSOLD` | 30 | Trigger threshold |
| `RSI_BOOST` | 1.30 | Multiplier; result clamped to 1.0 |
| `VOL_TARGET` | 0.40 | Annualized vol target |
| `VOL_SCALING_WINDOW` | 15 | Ring-buffer length for BTC return σ |
| `VOL_SCALE_MIN` | 0.20 | Lower clip on scalar `k` |
| `VOL_SCALE_MAX` | 1.00 | Upper clip — no leverage |
| `HASH_RIBBON_ENABLED` | True | Feature flag |
| `HASH_RIBBON_FAST` | 30 | Fast hashrate EMA span |
| `HASH_RIBBON_SLOW` | 60 | Slow hashrate EMA span |
| `HASH_RIBBON_CAP_MULT` | 0.00 | Multiplier during `fast < slow` regime |
| `MAX_DAILY_WEIGHT_CHANGE` | 0.10 | Step limiter |
| `MIN_REBALANCE_THRESHOLD` | 0.04 | Dead-band |
| `TRADING_FEES_PERCENT` | 0.001 | Taker fee fraction |
| `INITIAL_PORTFOLIO_USD` | 100_000 | Starting USDC |

### `RiskManagementConfig`

| Attribute | Default | Notes |
|-----------|--------:|-------|
| `ROLLING_PEAK_WINDOW` | 252 | Lookback for drawdown high-water mark |
| `VOLATILITY_WINDOW` | 30 | NAV return window for σ |
| `VAR_LOOKBACK` | 30 | NAV return window for VaR |
| `VAR_CONFIDENCE` | 0.99 | Confidence level |
| `VAR_ZSCORE` | 2.33 | Analytical z-score (informational; empirical percentile used) |
| `MODE_CAPS` | `{NORMAL:1.00, CAUTION:0.75, RISK_OFF:0.45, EMERGENCY:0.05}` | |
| `DD_THRESHOLDS` | `{CAUTION:-0.12, RISK_OFF:-0.20, EMERGENCY:-0.35}` | |
| `VOL_THRESHOLDS` | `{CAUTION:0.75, RISK_OFF:1.00, EMERGENCY:1.40}` | |
| `VAR_THRESHOLDS` | `{CAUTION:0.04, RISK_OFF:0.06, EMERGENCY:0.09}` | |
| `RECOVERY_THRESHOLDS` | see §6.3 | All-3-conditions-AND gate |
| `RECOVERY_DAYS` | `{CAUTION:2, RISK_OFF:3, EMERGENCY:5}` | Consecutive days required |

### `ProductionCostConfig`

| Attribute | Default | Notes |
|-----------|--------:|-------|
| `OVERHEAD_FACTOR` | 1.47 | Multiplier on top of raw energy cost |
| `BLOCKS_PER_DAY` | 144 | Expected block rate |
| `EMA_WINDOW` | 14 | First-pass smoothing inside `production_cost.py` |

### `BacktestConfig`

| Attribute | Default | Notes |
|-----------|--------:|-------|
| `START_DATE` | `"2016-02-01"` | |
| `END_DATE` | `"2026-02-01"` | |
| `DAYS_TO_FETCH` | 3650 | Passed to `DataFetcher.fetch_combined_data()` |
| `USE_REAL_DATA` | `True` | `True` = load CSV cache; `False` = live API |
