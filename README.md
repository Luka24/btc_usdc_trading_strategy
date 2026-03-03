# BTC/USDC Trading Strategy

A systematic **mean-reversion trading strategy** for BTC/USDC based on the ratio of market price to Bitcoin mining production cost. The system includes a professional risk management engine, walk-forward optimised signal overlays, and an interactive Streamlit dashboard.

**[View Live Dashboard →](https://btcusdctradingstrategy-b9chdjulctmnqyappgpjtms.streamlit.app/)**

For full technical specification and parameter rationale, see [STRATEGY_SPECIFICATION.md](STRATEGY_SPECIFICATION.md).

---

## How It Works

The core signal is the **Price/Cost ratio**:

$$R_t = \frac{P_{ema,t}}{C_{ema,t}}$$

- **$P_{ema}$** — 21-day EMA of BTC price
- **$C_{ema}$** — 30-day EMA of Bitcoin production cost (from mining model)
- **$R_t < 1$** → BTC is undervalued → accumulate
- **$R_t > 1$** → BTC is overvalued → reduce exposure

The step-function output of the ratio is further smoothed with a **14-day EMA** (`SIGNAL_EMA_WINDOW`) to prevent abrupt position jumps at band boundaries.

The production cost is calculated daily from real hashrate data, year-specific electricity prices ($/kWh), year-specific network miner efficiency (J/TH), halving schedule, and an overhead factor of **1.47×** covering CAPEX, facilities, and operations (~68% electricity, 32% overhead).

---

## Position Sizing

| Ratio ($R_{ema}$) | BTC Target | USDC Target |
|---|---:|---:|
| < 0.80 | 100% | 0% |
| 0.80 – 0.90 | 85% | 15% |
| 0.90 – 1.00 | 70% | 30% |
| 1.00 – 1.10 | 50% | 50% |
| 1.10 – 1.25 | 30% | 70% |
| > 1.25 | 0% | 100% |

**Execution rules:**
- Max daily rebalance step: **10%** of portfolio
- Minimum rebalance threshold (dead band): **4%**
- Trading fee: **0.1%** per trade
- Rebalancing: once per day at 00:00 UTC

**Signal overlays** (walk-forward validated, applied in this order):
- **RSI boost**: multiply signal by 1.30× when RSI-14 < 30 (oversold accumulation boost)
- **Hash Ribbon**: force 0% BTC when 30-day hashrate SMA < 60-day SMA (miner capitulation — full exit)
- **Vol scaling**: scale to target **40%** annualised volatility (15-day lookback, scalar 20–100%)
- **Trend filter**: force 0% BTC when price < 250-day EMA (full exit in bear market — applied last)

---

## Risk Management

A 4-mode state machine with **3 independent triggers** (Drawdown, Volatility, Value-at-Risk).  
The active mode is `max(DD_mode, Vol_mode, VaR_mode)` (most severe wins).  
**Entry is instant (1 breach = immediate downgrade). Recovery requires all 3 conditions simultaneously for N consecutive days — any single breach resets the counter to 0.**

### Mode Caps

| Mode | BTC Cap | USDC Floor |
|---|---:|---:|
| NORMAL | 100% | 0% |
| CAUTION | 75% | 25% |
| RISK_OFF | 45% | 55% |
| EMERGENCY | 5% | 95% |

### Entry Thresholds

| Mode | Drawdown | Ann. Volatility | VaR 99% (1-day) |
|---|---:|---:|---:|
| CAUTION | ≤ −12% | ≥ 75% | ≥ 4% |
| RISK_OFF | ≤ −20% | ≥ 100% | ≥ 6% |
| EMERGENCY | ≤ −35% | ≥ 140% | ≥ 9% |

### Recovery Thresholds & Days

| Mode | DD Recovery | Vol Recovery | VaR Recovery | Days |
|---|---:|---:|---:|---:|
| CAUTION → NORMAL | > −9% | < 65% | < 2.5% | 2 |
| RISK_OFF → CAUTION | > −16% | < 85% | < 4% | 3 |
| EMERGENCY → RISK_OFF | > −28% | < 120% | < 6% | 5 |

Drawdown uses a **252-day rolling peak** (not all-time), which prevents permanent lockout.

---

## Project Structure

```
btc_usdc_trading_strategy/
│
├── README.md                        # This file
├── STRATEGY_SPECIFICATION.md        # Full strategy spec & parameter rationale
├── config.py                        # All parameters (single source of truth)
├── requirements.txt
├── .gitignore
│
├── Core
│   ├── backtest.py                  # Backtesting engine & signal generation
│   ├── risk_manager.py              # 4-mode risk state machine
│   ├── portfolio.py                 # Portfolio state & metrics
│   └── production_cost.py          # BTC mining cost model
│
├── Data
│   ├── data_fetcher.py              # CoinGecko + Blockchain.com API fetcher
│   └── data/                        # Cached CSV files (gitignored)
│
├── Interface
│   ├── dashboard.py                 # Streamlit entrypoint (main page)
│   ├── strategy_page.py             # Streamlit strategy methodology page
│   ├── main.py                      # CLI entrypoint
│   └── pages/
│       └── 2_Strategy_and_Methodology.py
│
├── Optimization
│   └── optimization/
│       ├── optuna_optimizer.py      # 3-phase Bayesian optimisation (Optuna)
│       ├── ml_parameter_optimizer.py
│       ├── walk_forward.py          # Walk-forward validation
│       ├── ablation_study.py        # Feature importance ablation
│       ├── phase4_validation.py     # Final OOS validation
│       └── best_params.json         # Last optimised parameters
│
└── Tests
    └── tests/
        ├── test_live_data_smoke.py
        └── test_pro_risk_modes.py
```

---

## Quick Start

### Prerequisites
- Python 3.10+

```bash
git clone https://github.com/Luka24/btc_usdc_trading_strategy.git
cd btc_usdc_trading_strategy

python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### Run Dashboard
```bash
streamlit run dashboard.py
```
Opens at `http://localhost:8501`.

### Run CLI Backtest
```bash
python main.py
```

---

## Key Parameters (`config.py`)

```python
# ── Signal ────────────────────────────────────────────────────────────────
SignalConfig.PRICE_EMA_WINDOW    = 21     # EMA window for BTC price smoothing
SignalConfig.COST_EMA_WINDOW     = 30     # EMA window for production cost smoothing
SignalConfig.SIGNAL_EMA_WINDOW   = 14     # additional EMA on step-function output

# ── Position bands ────────────────────────────────────────────────────────
# (R_low, R_high, BTC_weight)
SignalConfig.POSITION_TABLE = [
    (0.00, 0.80, 1.00),   # strong accumulation
    (0.80, 0.90, 0.85),   # aggressive buy
    (0.90, 1.00, 0.70),   # moderate buy
    (1.00, 1.10, 0.50),   # neutral
    (1.10, 1.25, 0.30),   # defensive
    (1.25, 10.0, 0.00),   # full de-risk
]

# ── Portfolio ─────────────────────────────────────────────────────────────
PortfolioConfig.MAX_DAILY_WEIGHT_CHANGE   = 0.10   # 10% max step per day
PortfolioConfig.MIN_REBALANCE_THRESHOLD   = 0.04   # 4% dead band (no trade below)
PortfolioConfig.INITIAL_PORTFOLIO_USD     = 100_000
PortfolioConfig.TRADING_FEES_PERCENT      = 0.001  # 0.1% per trade

# ── Overlays ──────────────────────────────────────────────────────────────
PortfolioConfig.TREND_FILTER_WINDOW      = 250     # bear market EMA window
PortfolioConfig.TREND_BEAR_CAP           = 0.00    # full exit below trend EMA
PortfolioConfig.RSI_WINDOW               = 14
PortfolioConfig.RSI_OVERSOLD             = 30
PortfolioConfig.RSI_BOOST                = 1.30    # +30% signal boost when oversold
PortfolioConfig.VOL_TARGET               = 0.40    # target 40% annualised vol
PortfolioConfig.VOL_SCALING_WINDOW       = 15
PortfolioConfig.VOL_SCALE_MIN            = 0.20
PortfolioConfig.VOL_SCALE_MAX            = 1.00
PortfolioConfig.HASH_RIBBON_ENABLED      = True
PortfolioConfig.HASH_RIBBON_FAST         = 30      # fast hashrate SMA window
PortfolioConfig.HASH_RIBBON_SLOW         = 60      # slow hashrate SMA window
PortfolioConfig.HASH_RIBBON_CAP_MULT     = 0.00    # 0% BTC on miner capitulation

# ── Risk ──────────────────────────────────────────────────────────────────
RiskManagementConfig.MODE_CAPS = {
    "NORMAL": 1.00, "CAUTION": 0.75, "RISK_OFF": 0.45, "EMERGENCY": 0.05
}
RiskManagementConfig.DD_THRESHOLDS  = {"CAUTION": -0.12, "RISK_OFF": -0.20, "EMERGENCY": -0.35}
RiskManagementConfig.VOL_THRESHOLDS = {"CAUTION":  0.75, "RISK_OFF":  1.00, "EMERGENCY":  1.40}
RiskManagementConfig.VAR_THRESHOLDS = {"CAUTION":  0.04, "RISK_OFF":  0.06, "EMERGENCY":  0.09}
RiskManagementConfig.RECOVERY_THRESHOLDS = {
    "CAUTION":   {"dd": -0.09, "vol": 0.65, "var": 0.025},
    "RISK_OFF":  {"dd": -0.16, "vol": 0.85, "var": 0.04},
    "EMERGENCY": {"dd": -0.28, "vol": 1.20, "var": 0.06},
}
RiskManagementConfig.RECOVERY_DAYS = {"CAUTION": 2, "RISK_OFF": 3, "EMERGENCY": 5}

# ── Backtest ─────────────────────────────────────────────────────────────
BacktestConfig.START_DATE = "2016-02-01"
BacktestConfig.END_DATE   = "2026-02-01"
```

---

## Optimisation

Parameters were tuned using a 3-phase Bayesian optimisation pipeline (`optimization/optuna_optimizer.py`):

1. **Phase A** — Signal parameters (EMA windows, position bands)
2. **Phase B** — Risk thresholds (DD / Vol / VaR entries & recovery)
3. **Phase C** — Overlays (trend filter, vol scaling, hash ribbon, RSI)

Validated with walk-forward analysis (`optimization/walk_forward.py`) and feature ablation (`optimization/ablation_study.py`). Final out-of-sample test in `optimization/phase4_validation.py`. Best parameters serialised to `optimization/best_params.json`.

**Objective:** Maximise Sharpe Ratio, subject to Max Drawdown < −30%.

---

## Production Cost Model

| Year | Electricity ($/kWh) | Avg Efficiency (J/TH) | Block Reward |
|---|---:|---:|---:|
| 2016 | 0.10 | 250 | 12.5 BTC |
| 2017 | 0.10 | 150 | 12.5 BTC |
| 2018 | 0.08 | 110 | 12.5 BTC |
| 2019 | 0.07 | 85 | 12.5 BTC |
| 2020 | 0.065 | 60 | 6.25 BTC (halving 11 May) |
| 2021 | 0.065 | 50 | 6.25 BTC |
| 2022 | 0.07 | 45 | 6.25 BTC (energy crisis) |
| 2023 | 0.065 | 38 | 6.25 BTC |
| 2024 | 0.062 | 30 | 3.125 BTC (halving 20 Apr) |
| 2025 | 0.06 | 24 | 3.125 BTC |
| 2026 | 0.06 | 22 | 3.125 BTC |

---

## Data

Historical data is fetched via:
- **BTC price** — CoinGecko API (daily close, USD)
- **Hashrate** — Blockchain.com API (EH/s, daily)

Cached CSV files are stored in `data/` with filenames encoding the period length (e.g. `combined_data_3650d.csv`). The backtest covers **2016-02-01 to 2026-02-01** (~10 years, 2 full halving cycles, multiple bull/bear regimes).

---

## Dependencies

```
pandas, numpy        — data processing
streamlit==1.53.1    — web dashboard
plotly, altair       — interactive charts
matplotlib           — static plots
requests             — API calls
yfinance             — market data fallback
optuna               — Bayesian optimisation
scikit-learn         — ML utilities
reportlab            — PDF export
```

---

## Disclaimer

This is an educational/research project. Past performance does not guarantee future results.
This is NOT financial advice.
