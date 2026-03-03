import streamlit as st
import pandas as pd


def render_strategy_page() -> None:
    st.set_page_config(
        page_title="Strategy & Methodology",
        page_icon="S",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    st.markdown(
        """
<style>
h1 { color: #2f3b2f; }
h2 { color: #3b4637; }
.stButton > button {
    background-color: #4f6f52;
    color: #ffffff;
    border: 1px solid #3f5f42;
    border-radius: 6px;
}
.stButton > button:hover {
    background-color: #3f5f42;
    color: #ffffff;
    border: 1px solid #2f4f32;
}
</style>
""",
        unsafe_allow_html=True,
    )

    if st.button("← Back to Dashboard"):
        if st.session_state.get("show_strategy_page", False):
            st.session_state["show_strategy_page"] = False
            st.rerun()
        else:
            try:
                st.switch_page("Dashboard.py")
            except Exception:
                st.session_state["show_strategy_page"] = False
                st.rerun()

    st.title("Strategy & Methodology — Full Documentation")

    st.markdown(
        "This page explains how the trading strategy works: the main idea, where the data comes from, "
        "how each parameter was chosen, and what the testing results show."
    )

    st.markdown("---")

    # ========== HOW I STARTED ==========
    st.header("0️⃣ How I Started")

    st.markdown("""
    ### First Steps

    I started by asking a basic question: can you build a rule-based system that knows when Bitcoin
    is cheap or expensive — not based on price charts alone, but on something more fundamental?

    **Research phase**
    - I read about Bitcoin mining economics: how hashrate, block rewards, and electricity costs
      connect to miner profitability.
    - Key readings: Charles Edwards's "Bitcoin Energy Value" paper (Capriole Investments),
      JPMorgan's annual Bitcoin mining cost reports, and a CESifo academic paper on mining economics.
    - I also used Cambridge's Bitcoin Electricity Consumption Index (CBECI) for energy consumption data.

    **Build phase**
    - I first built a static version with hardcoded numbers to check the core idea made sense.
    - Then I added live APIs (CoinGecko for price, Blockchain.com for hashrate).
    - Finally I built the Streamlit dashboard so the strategy runs and updates automatically.

    **What I used AI tools for**
    - Code boilerplate, debugging, and writing the Streamlit layout faster.
    - I always checked the output and corrected errors — especially the numbers, which AI often gets wrong.

    **What I learned**
    - Mining economics create real price anchors. When miners lose money, fewer coins reach the market.
      When they profit heavily, more capital enters mining and selling pressure grows.
    - Simple rule-based systems often generalise better than complex ML in cryptocurrency markets.
    """)

    st.markdown("---")

    # ========== STRATEGY OVERVIEW ==========
    st.header("1️⃣ Strategy Overview & Core Idea")

    st.markdown("""
    ### Mean-Reversion Based on Production Cost

    The strategy belongs to the **mean-reversion** family: it assumes that when Bitcoin's price moves
    far from its "fair value", it tends to come back over time. Instead of using a statistical average
    as the anchor, I use the **all-in production cost** — what it actually costs miners to produce one
    Bitcoin on the network.

    ---

    ### Economic Logic

    **Why production cost works as an anchor:**

    1. **Below cost → miner losses.** Unprofitable miners shut machines off. Less hashrate means fewer new coins produced and less selling pressure. Price tends to recover.

    2. **Far above cost → new capital enters.** High profits attract new mining farms. More coins produced increases selling pressure that pulls price back toward cost.

    3. **Market psychology.** Institutional traders use mining cost as a valuation floor and step in when price drops near it.

    Academic reference: [CESifo Working Paper — Economics of Bitcoin Mining](https://www.cesifo.org/DocDL/cesifo1_wp10145.pdf)

    ---

    ### The Core Signal: Price / Cost Ratio

    Every day the strategy calculates:

    $$R = \\frac{\\text{BTC Price (28-day EMA)}}{\\text{Production Cost (46-day EMA)}}$$

    **EMA (Exponential Moving Average):** a weighted average where recent values count more than older ones. A 28-day EMA gives roughly 7% weight to today's price and fades out older data. It follows price more closely than a simple average but filters out single-day spikes.

    The ratio $R$ tells us how expensive BTC is relative to what it costs to mine it. Both EMA windows were found by walk-forward cross-validation — see [optimization/best_params.json](https://github.com/Luka24/btc_usdc_trading_strategy/blob/main/optimization/best_params.json).

    **Walk-forward cross-validation:** a method for testing parameters on data they were never trained on. The full history is split into many windows. In each window, the model is trained on the first part and tested on the next part (which it has not seen). The window then moves forward and repeats. The final Sortino score is the average across all the test parts only — never the training parts. This prevents the strategy from being tuned specifically to past data it will never see again.

    ---

    ### Position Table — How Much BTC to Hold

    The ratio maps directly to a target BTC allocation. These bands come from
    [config.py → SignalConfig.POSITION_TABLE](https://github.com/Luka24/btc_usdc_trading_strategy/blob/main/config.py)
    and have not changed since the start of the project:

    | Ratio $R$ | Target BTC Weight | Interpretation |
    |---|---|---|
    | $R < 0.80$ | **100%** | Price well below cost — strong buy zone |
    | $0.80 \\leq R < 0.90$ | **85%** | Below cost — accumulate |
    | $0.90 \\leq R < 1.00$ | **70%** | Slightly below cost — lean long |
    | $1.00 \\leq R < 1.10$ | **50%** | Near fair value — neutral |
    | $1.10 \\leq R < 1.25$ | **30%** | Above cost — reduce exposure |
    | $R \\geq 1.25$ | **0%** | Significantly above cost — exit |

    **Why these bands?**
    - They come from Charles Edwards' "Bitcoin Energy Value" framework — [TradingView indicator by Capriole](https://www.tradingview.com/script/xDMwWgRN-Bitcoin-Energy-Value/). The five ratio thresholds (0.80, 0.90, 1.00, 1.10, 1.25) were **not** optimised by walk-forward CV. Adding five more free parameters to an already small dataset would create a high overfitting risk. The thresholds were set once from the paper and frozen.
    - The step structure means the portfolio moves gradually rather than in large jumps.

    """)

    st.markdown("---")

    # ========== DATA SOURCES ==========
    st.header("2️⃣ Data Sources")

    st.markdown("""
    ### Bitcoin Price

    | Source | Used For | Link |
    |---|---|---|
    | CoinGecko API (free tier) | Recent prices (up to 365 days) | [https://www.coingecko.com/en/api/documentation](https://www.coingecko.com/en/api/documentation) |
    | Yahoo Finance via `yfinance` | Historical data (2014 onwards) | [https://finance.yahoo.com/quote/BTC-USD/history](https://finance.yahoo.com/quote/BTC-USD/history) |

    CoinGecko gives higher-quality recent data but is rate-limited. Yahoo Finance covers
    the full 10-year backtest window for free without any API key.

    ---

    ### Network Hashrate

    **Source:** Blockchain.com Charts API  
    **Endpoint:** [https://api.blockchain.info/charts/hash-rate?timespan=all&format=json](https://api.blockchain.info/charts/hash-rate?timespan=all&format=json)

    This returns daily network hashrate in TH/s going back to 2009. Free, no authentication needed.

    **Important note:** When fetching a very long window (3,000+ days), Blockchain.com returns
    one data point every four days instead of daily. The strategy uses linear interpolation to
    fill the gaps. Fetching at least 3,000 days is intentional — it keeps data density consistent
    and prevents false Hash Ribbon signals caused by noisy data from shorter windows.

    ---

    ### Data Pipeline

    - **Code:** [data_fetcher.py](https://github.com/Luka24/btc_usdc_trading_strategy/blob/main/data_fetcher.py)
    - Fetched data is saved locally in `data/` to avoid repeated API calls during the same session.
    - The dashboard refreshes the cache every **60 minutes** (`@st.cache_data(ttl=3600)`).
    """)

    st.markdown("---")

    # ========== PRODUCTION COST MODEL ==========
    st.header("3️⃣ Production Cost Model")

    st.markdown("""
    ### The Formula (Step by Step)

    **Step 1 — Network energy consumption per day**

    $$\\text{Energy (kWh)} = \\frac{\\text{Hashrate (EH/s)} \\times 10^{18} \\times 86{,}400 \\times \\text{Efficiency (J/TH)}}{10^{12} \\times 3.6 \\times 10^{6}}$$

    *Hashrate: EH/s → H/s by ×$10^{18}$. 86,400 = seconds per day. ÷$10^{12}$ converts TH. ÷$3.6×10^6$ converts joules to kWh.*

    **Step 2 — Energy cost per day**

    $$\\text{Energy Cost} = \\text{Energy (kWh)} \\times \\text{Electricity Price (\\$/kWh)}$$

    **Step 3 — BTC produced per day**

    $$\\text{BTC/Day} = \\underbrace{144}_{\\text{blocks/day}} \\times \\text{Block Reward}$$

    One block is mined roughly every 10 minutes (enforced by the difficulty adjustment algorithm),
    giving 144 blocks per day. The block reward halves on a fixed schedule.

    **Step 4 — Energy cost per BTC**

    $$\\text{Energy Cost/BTC} = \\frac{\\text{Total Energy Cost}}{\\text{BTC/Day}}$$

    **Step 5 — All-in production cost**

    $$\\boxed{\\text{Production Cost per BTC} = \\text{Energy Cost/BTC} \\times 1.47}$$

    **Code:** [production_cost.py](https://github.com/Luka24/btc_usdc_trading_strategy/blob/main/production_cost.py)
      → `BTCProductionCostCalculator.calculate_total_cost_per_btc()`

    ---

    ### Overhead Factor = 1.47

    Energy is not a miner's only cost. The factor 1.47 means: for every 1.00 USD spent on electricity, the total all-in cost is 1.47 USD.

    **What the extra 47% covers:** hardware depreciation (ASICs last ~2–3 years), facility costs (cooling, buildings), staff and operations, and admin/pool fees (~1–2%).

    **Where 1.47 comes from:**
    CoinShares' Bitcoin Mining Network Report estimates electricity at 65–70% of total all-in mining costs, giving 1 / 0.68 ≈ 1.47.

    **This parameter is fixed — not optimised.** Changing it shifts the absolute cost level
    but not the ratio signal (both numerator and denominator scale identically).

    **Code:** [config.py](https://github.com/Luka24/btc_usdc_trading_strategy/blob/main/config.py)
      → `ProductionCostConfig.OVERHEAD_FACTOR = 1.47`

    ---

    ### Electricity Prices and Miner Efficiency (Historical)
    """)

    col1, col2 = st.columns(2)

    electricity_data = {
        'Year': [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026],
        'Price ($/kWh)': [0.10, 0.10, 0.08, 0.07, 0.065, 0.065, 0.07, 0.065, 0.062, 0.06, 0.06],
        'Notes': [
            'Small operations, high overhead',
            'Rapid growth phase',
            'More competitive energy bids',
            'Hydropower deals mature',
            'Cheap energy locked in',
            'Low-cost region expansion',
            'Energy crisis pushes up',
            'Stabilising',
            'Pre-halving, old miners still running',
            'Halving forces efficiency upgrades',
            'Current estimate',
        ]
    }

    efficiency_data = {
        'Year': [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026],
        'Efficiency (J/TH)': [250, 150, 110, 85, 60, 50, 45, 38, 30, 24, 22],
        'Dominant Hardware': [
            'Bitmain S7, S8',
            'S9 rollout',
            'S9 widespread',
            'S9 ageing out',
            'S17 series',
            'S19 series',
            'S19 growing',
            'S19 XP, early S21',
            'S19 / S21 mix',
            'S21 Pro widespread',
            'Next-gen ASICs',
        ]
    }

    with col1:
        st.subheader("Electricity Prices")
        df_electricity = pd.DataFrame(electricity_data)
        st.dataframe(df_electricity, use_container_width=True, hide_index=True)
        st.caption(
            "Network-average effective electricity rate for mid-to-large mining operations. "
            "Includes cooling and facility overhead, not just the raw generation rate. "
            "Sources: US EIA industrial rates "
            "(https://www.eia.gov/electricity/monthly/epm_table_grapher.php?t=epmt_5_6_a), "
            "CBECI (https://ccaf.io/cbnsi/cbeci), mining company disclosures."
        )

    with col2:
        st.subheader("Network-Average Miner Efficiency")
        df_efficiency = pd.DataFrame(efficiency_data)
        st.dataframe(df_efficiency, use_container_width=True, hide_index=True)
        st.caption(
            "Network average across all active machines — not just the newest generation. "
            "Old hardware stays online while it remains profitable, so the network average "
            "always lags behind the latest flagship ASIC. "
            "Sources: CoinShares Bitcoin Mining Report, CESifo WP10145, "
            "Hashrate Index (https://hashrateindex.com/machines)."
        )

    st.markdown("""
    **Code:** [config.py](https://github.com/Luka24/btc_usdc_trading_strategy/blob/main/config.py)
      → `HistoricalParameters.ELECTRICITY_PRICES_USD_PER_KWH` and
      `HistoricalParameters.NETWORK_AVERAGE_EFFICIENCY_J_PER_TH`

    ---

    ### Halving Schedule

    The block reward halves on a fixed schedule, encoded as a lookup table in the strategy:

    | Date | Block Reward |
    |---|---|
    | Before Nov 28, 2012 | 50 BTC |
    | Nov 28, 2012 | 25 BTC |
    | Jul 9, 2016 | 12.5 BTC |
    | May 11, 2020 | 6.25 BTC |
    | Apr 20, 2024 | 3.125 BTC |
    | ~Apr 2028 | 1.5625 BTC |

    Source: [https://en.bitcoin.it/wiki/Controlled_supply](https://en.bitcoin.it/wiki/Controlled_supply)

    Each halving immediately doubles the energy cost per BTC (same energy input, half the output).
    The production cost signal reflects this automatically on the halving date.

    **Code:** [config.py](https://github.com/Luka24/btc_usdc_trading_strategy/blob/main/config.py)
      → `ProductionCostConfig.HALVING_SCHEDULE`

    ---

    ### Cost Series Smoothing

    The raw daily production cost is first smoothed with a **14-day EMA** inside `ProductionCostSeries`,
    and then the ratio uses the **46-day EMA** (walk-forward optimised). Two-stage smoothing removes
    both daily hashrate noise and occasional outlier spikes.

    **Code:** [config.py](https://github.com/Luka24/btc_usdc_trading_strategy/blob/main/config.py)
      → `ProductionCostConfig.EMA_WINDOW = 14` (first-stage smoothing; fixed)
    """)

    st.markdown("---")

    # ========== PORTFOLIO MANAGEMENT ==========
    st.header("4️⃣ Portfolio Management")

    st.markdown("""
    ### Starting Position

    The backtest opens with $100,000. Up to 2 BTC are bought at the first day's price;
    any remaining capital stays in USDC.

    **Code:** [config.py](https://github.com/Luka24/btc_usdc_trading_strategy/blob/main/config.py)
      → `PortfolioConfig.INITIAL_PORTFOLIO_USD = 100_000`

    ---

    ### Daily Weight Change Limit — 10%

    At most 10 percentage points of the portfolio move toward the target each day.
    Moving from 50% BTC to 100% BTC takes five trading days.

    This limits market impact on a real exchange and prevents a single noisy signal
    from causing a large, costly trade.

    *Example: target = 100%, current = 50%.*
    *Day 1 → 60%, Day 2 → 70%, Day 3 → 80%, Day 4 → 90%, Day 5 → 100%.*

    **Fixed.** 10% is a standard daily position-limit assumption for algorithmic execution strategies.

    ---

    ### Minimum Rebalance Threshold — 4%

    No trade executes if the target weight is less than 4 percentage points from the
    current weight. This avoids paying fees for very small adjustments.

    **Fixed.** Chosen so the strategy does not churn on band-boundary noise.

    ---

    ### Trading Fees — 0.1% per trade

    Applied on every buy or sell. Current exchange rates for reference:
    - Binance spot: 0.10% standard ([https://www.binance.com/en/fee/schedule](https://www.binance.com/en/fee/schedule))
    - Coinbase Advanced Trade: 0.05–0.60% depending on volume ([https://www.coinbase.com/advanced-trade/fees](https://www.coinbase.com/advanced-trade/fees))
    - Kraken: 0.16–0.26% for most users

    0.1% is a mid-range estimate for a regular trader on a major spot exchange.
    **Fixed** — set to match Binance standard rate.

    ---

    ### Portfolio Parameters — Summary

    | Parameter | Value | Status | Source |
    |---|---|---|---|
    | Starting capital | $100,000 | **Fixed** | Normalised baseline for fair comparison |
    | Starting BTC | up to 2.0 BTC | **Fixed** | Capped so BTC value ≤ starting capital |
    | Max daily weight change | 10% | **Fixed** | Standard algorithmic execution limit |
    | Min rebalance threshold | 4% | **Fixed** | Transaction cost floor |
    | Trading fee | 0.1% | **Fixed** | [Binance standard spot rate](https://www.binance.com/en/fee/schedule) |

    **Code:** [config.py](https://github.com/Luka24/btc_usdc_trading_strategy/blob/main/config.py)
      → `PortfolioConfig`
    """)

    st.markdown("---")

    # ========== SIGNAL FILTERS & ABLATION ==========
    st.header("5️⃣ Signal Filters")

    st.markdown("""
    ### Ablation Study — What Each Component Contributes

    **Sortino ratio:** a measure of risk-adjusted return. It is the average yearly return divided by how large the losing days were (only downside volatility counts — good volatility on the upside is not penalised). Higher = better. A Sortino of 2.0 means the strategy returns twice as much per unit of downside risk.

    Each filter was switched off one at a time. Everything else stayed active.
    The table shows the drop in **out-of-sample Sortino ratio**.
    Baseline (all on) = **2.443**.

    | Component removed | Sortino change | Verdict |
    |---|---|---|
    | Signal EMA smoothing | **−0.43** | Biggest driver |
    | Hash Ribbon filter | **−0.34** | Second biggest |
    | Trend filter (360-day EMA) | **−0.26** | Significant |
    | RSI oversold boost | **−0.07** | Small but consistent; kept |
    | 4-mode risk engine | **−0.02** | Low Sortino impact, strong tail protection |
    | Volatility scaling | **−0.02** | Low Sortino impact, keeps vol near 40% target |

    The risk engine and volatility scaler rarely activate, so their Sortino improvement
    looks small. But in 2018 and 2022 they cut exposure for weeks during the worst drops —
    which Sortino alone does not fully show.

    ---

    ### Trend Filter

    **Setting:** `TREND_FILTER_WINDOW = 360 days` — **walk-forward optimised**
    ([best_params.json](https://github.com/Luka24/btc_usdc_trading_strategy/blob/main/optimization/best_params.json))

    When BTC price is below its 360-day EMA, the strategy caps BTC exposure at **0%**
    regardless of the ratio signal. This prevents holding BTC through a prolonged bear market.

    Shorter windows (e.g. 200 days) generate too many false bear signals during normal
    corrections. 360 days was found optimal by walk-forward cross-validation.

    ---

    ### Hash Ribbon Filter

    **Settings:** Fast SMA = **30 days**, Slow SMA = **60 days** — **fixed from literature**;
    `HASH_RIBBON_CAP_MULT = 0.1` — **walk-forward optimised**
    ([best_params.json](https://github.com/Luka24/btc_usdc_trading_strategy/blob/main/optimization/best_params.json))

    **What the Hash Ribbon measures:** Network hashrate is the total computing power solving Bitcoin blocks. When mining is profitable, hashrate grows as operators add machines. When miners lose money, they shut machines off and hashrate falls.

    The filter tracks two simple moving averages (SMAs) of daily hashrate:
    - **Fast SMA (30 days):** reacts quickly to recent changes in miner activity.
    - **Slow SMA (60 days):** represents the longer-term hashrate trend.

    When the fast SMA crosses below the slow SMA, it means hashrate has recently started falling — miners are actively turning off machines. This is called **miner capitulation**. Historically it precedes large price drops, because distressed miners sell BTC to cover running costs before shutting down. The fast/slow SMA pair catches the crossover early: the fast line reacts to the drop first, then the slow line confirms the trend.

    - Fast SMA < Slow SMA (capitulation active): BTC exposure capped to `0.1 × current mode cap`
    - Fast SMA > Slow SMA (recovery): normal trading resumes

    The 30/60-day windows come from Charles Edwards' original Hash Ribbon research:
    [Capriole — Hash Ribbons & Bitcoin Bottoms](https://capriole.com/hash-ribbons-bitcoin-bottoms/).
    The cap multiplier of 0.1 was tuned by walk-forward optimisation — instead of a hard 0% shutdown, a small residual position slightly improves out-of-sample Sortino.

    **Code:** [config.py](https://github.com/Luka24/btc_usdc_trading_strategy/blob/main/config.py)
      → `PortfolioConfig.HASH_RIBBON_FAST = 30`, `HASH_RIBBON_SLOW = 60`, `HASH_RIBBON_CAP_MULT = 0.1`

    ---

    ### Volatility Scaling

    **Settings:** Vol target = **40% annualised** (optimised), lookback = **15 days** (fixed),
    min scale = **0.20** (fixed), max scale = **1.00** (fixed)

    $$\\text{Scale} = \\text{clip}\\!\\left(\\frac{0.40}{\\text{Realised Vol (15d)}},\\ 0.20,\\ 1.00\\right)$$

    When realised vol exceeds 40% annualised, the position shrinks proportionally.
    At low vol, scale stays at 1.00 — no leverage.

    - **VOL_TARGET = 0.40** — optimised. Raw BTC runs at 60–80% annualised vol.
    - **VOL_SCALING_WINDOW = 15 days** — fixed. Standard two-week lookback.
    - **VOL_SCALE_MIN = 0.20** — keeps at least 20% BTC exposure during recoveries.
    - **VOL_SCALE_MAX = 1.00** — no leverage.

    ---

    ### RSI Oversold Boost

    **Settings:** Window = **14 days** (fixed), threshold = **28** (optimised), multiplier = **1.30** (fixed)

    When the 14-day RSI drops below 28, the target BTC weight multiplies by ×1.30.
    This adds a small extra buy push during very oversold conditions (e.g. March 2020).

    | Parameter | Value | Status | Source |
    |---|---|---|---|
    | RSI period | 14 days | **Fixed** | [Wilder (1978) — RSI standard](https://en.wikipedia.org/wiki/Relative_strength_index) |
    | RSI threshold | 28 | **Optimised** | [best_params.json](https://github.com/Luka24/btc_usdc_trading_strategy/blob/main/optimization/best_params.json) |
    | Multiplier | 1.30 | **Fixed** | Calibrated; consistent with momentum literature |

    ---

    ### Sentiment Signals Tested and Dropped

    **Fear & Greed Index** — removed. Native data starts January 2018; pre-2018 values are
    back-calculated from other indicators, not observed in real time. Using them inflates
    backtest results artificially.
    Source: [alternative.me/crypto/fear-and-greed-index](https://alternative.me/crypto/fear-and-greed-index/)

    **SOPR (Spent Output Profit Ratio)** — removed. Reliable daily data starts around 2020.
    Too short a history to validate properly without overfitting.
    Source: [glassnode.com/metrics/sopr](https://glassnode.com/metrics/sopr)
    """)

    st.markdown("---")

    # ========== RISK MANAGEMENT ==========
    st.header("6️⃣ Risk Management Engine")

    st.markdown("""
    ### Overview

    The risk engine runs every day and checks **three signals**: drawdown, annualised volatility,
    and Value-at-Risk (99%, 1-day). If any one of them crosses its threshold, the engine moves
    up to a more defensive mode and caps the maximum BTC allocation.

    Recovery is slow on purpose: the engine returns to a less defensive mode only when **all three
    signals have healed** and a minimum number of consecutive calm days has passed.
    This stops the strategy from rushing back to full exposure after a short bounce.

    ---

    ### Indirect vs. Direct Risk Controls

    Most of the strategy's downside protection is **indirect** — it works by reducing position size, not by forcing a hard exit:

    | Indirect control | How it protects |
    |---|---|
    | **Production cost ratio** | Reduces BTC allocation when price is well above cost |
    | **360-day trend filter** | Forces 0% BTC in structural bear markets |
    | **Hash Ribbon filter** | Cuts exposure to near-zero during miner capitulation |
    | **4-mode risk engine** | Caps max BTC at 75/45/5% when drawdown, vol, or VaR exceeds threshold |
    | **Volatility scaling** | Shrinks position proportionally when realised vol exceeds 40% |
    | **Max daily weight change** | Limits how fast the portfolio moves, reducing impact of false signals |

    I also tested a **direct** approach: stop-loss at −15%, take-profit at +25%, and trailing stop at −10%. All three forced hard binary exits. They consistently triggered on normal pullbacks and caused the strategy to miss the recovery. All three hurt Sortino and were dropped. The indirect, gradual approach works better for an asset as volatile as Bitcoin.

    ---

    ### The 4 Modes — Optimised Thresholds

    Drawdown thresholds, vol thresholds, and VaR thresholds are all **walk-forward optimised**
    (loaded from [best_params.json](https://github.com/Luka24/btc_usdc_trading_strategy/blob/main/optimization/best_params.json)).
    Max BTC caps are **fixed** — see note below.

    | Mode | DD trigger | Ann. vol trigger | VaR (99%, 1d) trigger | Max BTC |
    |---|---|---|---|---|
    | **NORMAL** | — | — | — | **100%** |
    | **CAUTION** | ≤ −6% | ≥ 130% | ≥ 4% | **75%** |
    | **RISK_OFF** | ≤ −36% | ≥ 220% | ≥ 6% | **45%** |
    | **EMERGENCY** | ≤ −53% | ≥ 230% | ≥ 9% | **5%** |

    The RISK_OFF and EMERGENCY drawdown thresholds look large because they apply to
    **portfolio drawdown** — already dampened by position sizing, trend filter, and Hash Ribbon.
    Raw BTC drawdown in bear markets often exceeds 70–80%.

    Max BTC caps (100 → 75 → 45 → 5%) are **fixed**. The ablation impact was −0.02 Sortino,
    meaning changing them barely helps on average. The geometric step-down reflects standard
    institutional risk-escalation logic and was not changed after it was set.

    **Code:** [config.py](https://github.com/Luka24/btc_usdc_trading_strategy/blob/main/config.py)
      → `RiskManagementConfig.MODE_CAPS`

    ---

    ### Sticky Recovery

    To recover to a less defensive mode, **all three** signals must drop below their
    recovery thresholds, and the required number of calm days must have passed:

    | Current mode | Recovery conditions | Calm days needed |
    |---|---|---|
    | CAUTION → NORMAL | DD > −9%, Vol < 65%, VaR < 2.5% | **2 days** |
    | RISK_OFF → CAUTION | DD > −16%, Vol < 85%, VaR < 4% | **3 days** |
    | EMERGENCY → RISK_OFF | DD > −28%, Vol < 120%, VaR < 6% | **5 days** |

    Recovery days and recovery thresholds are **fixed** — not walk-forward optimised.
    They set a minimum stabilisation period to rule out dead-cat-bounce recoveries.

    **Code:** [config.py](https://github.com/Luka24/btc_usdc_trading_strategy/blob/main/config.py)
      → `RiskManagementConfig.RECOVERY_DAYS` and `RECOVERY_THRESHOLDS`  
    **Code:** [risk_manager.py](https://github.com/Luka24/btc_usdc_trading_strategy/blob/main/risk_manager.py)

    ---

    ### Risk Metrics — Formulas

    **Drawdown (rolling 252-day peak)**

    $$\\text{DD} = \\frac{\\text{Portfolio Value} - \\text{Peak}_{252\\text{d}}}{\\text{Peak}_{252\\text{d}}}$$

    A rolling 252-day peak (one trading year) prevents the engine from staying locked in EMERGENCY
    mode for years because the 2021 all-time high is still above current levels.
    Source for 252-day standard: hedge fund industry annual calculation horizon.

    **Annualised Volatility (30-day lookback, fixed)**

    $$\\text{Vol} = \\text{std}(r_{1..30}) \\times \\sqrt{252}$$

    **Value-at-Risk (99% confidence, 30-day lookback, fixed)**

    $$\\text{VaR}_{99} = \\mu - 2.33 \\times \\sigma$$

    where $\\mu$ and $\\sigma$ are the mean and standard deviation of the last 30 daily returns.
    Z-score = 2.33 is the standard 99th-percentile value for a normal distribution.

    **Sharpe Ratio**

    $$\\text{Sharpe} = \\frac{\\bar{r}}{\\sigma_r} \\times \\sqrt{252}$$

    Return per unit of **total** volatility — both upside and downside days are penalised equally.

    **Sortino Ratio**

    $$\\text{Sortino} = \\frac{\\bar{r}}{\\sigma_{\\text{downside}}} \\times \\sqrt{252}$$

    Return per unit of **downside** volatility only. Days where the portfolio gains are not counted
    as risk.

    ---

    **Why Sortino was chosen as the optimisation target instead of Sharpe:**

    Bitcoin has a strongly asymmetric return distribution. A good year (2020: +300%, 2023: +154%)
    produces very large positive daily returns. Sharpe treats these large positive moves as just
    as bad as large negative moves — both increase $\\sigma_r$. This means Sharpe *penalises the
    strategy for bull-market performance*, which is the opposite of what an investor wants.

    Sortino only measures risk where it actually hurts — on losing days. A strategy that rides
    large Bitcoin rallies fully and cuts exposure during crashes will score higher on Sortino than
    on Sharpe, and that is the correct reward for exactly the behaviour this strategy targets.

    Concretely: in walk-forward CV on 3,297 days of data, the baseline strategy achieved
    Sortino = **2.443** vs. Sharpe ≈ **1.38**. The gap exists because Bull-run months have
    30–50% monthly returns — Sharpe counts those as "volatile" and docks the score. Sortino
    does not.

    ---

    **Code:** [risk_manager.py](https://github.com/Luka24/btc_usdc_trading_strategy/blob/main/risk_manager.py)

    ---

    ### Potential Improvement — Regime-Aware VaR

    The current VaR calculation assumes returns follow a normal distribution (z = 2.33 for 99%).
    Bitcoin returns are not normal — they have fat tails, meaning extreme losses happen more
    often than the model expects. A future improvement would be to use a **GARCH model** or
    **historical simulation** instead of parametric VaR. GARCH adapts to changing volatility
    regimes; historical simulation uses the actual tail of observed losses without any
    distribution assumption.
    Reference: [Engle (2001) — GARCH 101](https://pubs.aeaweb.org/doi/10.1257/jep.15.4.157)
    """)

    st.markdown("---")

    # ========== PARAMETER TABLE ==========
    st.header("7️⃣ Complete Parameter Reference")

    st.markdown("""
    ### Optimised vs. Fixed — Summary Table

    | Parameter | Value | Status | Source |
    |---|---|---|---|
    | Price EMA window | 28 days | **Optimised** | [best_params.json](https://github.com/Luka24/btc_usdc_trading_strategy/blob/main/optimization/best_params.json) |
    | Cost EMA window | 46 days | **Optimised** | [best_params.json](https://github.com/Luka24/btc_usdc_trading_strategy/blob/main/optimization/best_params.json) |
    | Signal EMA window | 3 days | **Optimised** | [best_params.json](https://github.com/Luka24/btc_usdc_trading_strategy/blob/main/optimization/best_params.json) |
    | Trend filter window | 360 days | **Optimised** | [best_params.json](https://github.com/Luka24/btc_usdc_trading_strategy/blob/main/optimization/best_params.json) |
    | RSI oversold threshold | 28 | **Optimised** | [best_params.json](https://github.com/Luka24/btc_usdc_trading_strategy/blob/main/optimization/best_params.json) |
    | Volatility target | 40% ann. | **Optimised** | [best_params.json](https://github.com/Luka24/btc_usdc_trading_strategy/blob/main/optimization/best_params.json) |
    | CAUTION DD threshold | −6% | **Optimised** | [best_params.json](https://github.com/Luka24/btc_usdc_trading_strategy/blob/main/optimization/best_params.json) |
    | RISK_OFF DD threshold | −36% | **Optimised** | [best_params.json](https://github.com/Luka24/btc_usdc_trading_strategy/blob/main/optimization/best_params.json) |
    | EMERGENCY DD threshold | −53% | **Optimised** | [best_params.json](https://github.com/Luka24/btc_usdc_trading_strategy/blob/main/optimization/best_params.json) |
    | Overhead factor | 1.47 | **Fixed** | [CoinShares Bitcoin Mining Report](https://coinshares.com/research/bitcoin-mining-network) |
    | Blocks per day | 144 | **Fixed** | Bitcoin protocol — 10-minute block target |
    | Hash Ribbon fast SMA | 30 days | **Fixed** | [Capriole Hash Ribbon research](https://capriole.com/hash-ribbons-bitcoin-bottoms/) |
    | Hash Ribbon slow SMA | 60 days | **Fixed** | [Capriole Hash Ribbon research](https://capriole.com/hash-ribbons-bitcoin-bottoms/) |
    | Hash Ribbon cap multiplier | 0.10 | **Optimised** | [best_params.json](https://github.com/Luka24/btc_usdc_trading_strategy/blob/main/optimization/best_params.json) |
    | RSI window | 14 days | **Fixed** | Wilder (1978) — technical analysis standard |
    | RSI multiplier | 1.30 | **Fixed** | Momentum literature calibration |
    | Vol scaling lookback | 15 days | **Fixed** | Standard two-week window |
    | Vol scale min | 0.20 | **Fixed** | Keep minimum market participation |
    | Vol scale max | 1.00 | **Fixed** | No leverage |
    | Rolling peak window | 252 days | **Fixed** | One trading year — hedge fund industry standard |
    | VaR lookback | 30 days | **Fixed** | Standard 1-month lookback |
    | VaR confidence z-score | 2.33 | **Fixed** | 99th percentile normal distribution |
    | Recovery days — CAUTION | 2 days | **Fixed** | Minimum stabilisation window |
    | Recovery days — RISK_OFF | 3 days | **Fixed** | Minimum stabilisation window |
    | Recovery days — EMERGENCY | 5 days | **Fixed** | Minimum stabilisation window |
    | Max daily weight change | 10% | **Fixed** | Market impact assumption |
    | Min rebalance threshold | 4% | **Fixed** | Transaction cost floor |
    | Trading fee | 0.1% | **Fixed** | [Binance standard spot rate](https://www.binance.com/en/fee/schedule) |
    | Starting capital | $100,000 | **Fixed** | Normalised for fair comparison across periods |

    ---

    ### ML Optimisation Attempt (Ridge Regression)

    As a separate experiment, I trained a Ridge regression model to predict 7-day, 14-day, and
    30-day forward returns using features built from the ratio, EMAs, momentum, and volatility.

    **Result:** Model R² ranged from 0.02 to 0.08 — very low predictive power.
    The thresholds it suggested (BUY at 0.87, SELL at 1.13) produced only a 2–3% Sortino improvement
    over the walk-forward baseline, which was not worth the added complexity and overfitting risk.

    **Takeaway:** In a market with frequent regime changes (bull/bear/halving cycles), the simpler
    rule-based system with interpretable parameters generalises better than a learned model.

    Academic reference: Liu, Tsyvinski & Wu (2019), "The Cross-Section of Expected Crypto Returns":
    [https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3244646](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3244646)

    **Code:** [optimization/ml_parameter_optimizer.py](https://github.com/Luka24/btc_usdc_trading_strategy/blob/main/optimization/ml_parameter_optimizer.py)
    """)

    st.markdown("---")

    # ========== LIMITATIONS ==========
    st.header("8️⃣ Assumptions & Known Limitations")

    st.markdown("""
    ### Model Simplifications

    **1. Fixed overhead factor**
    The 1.47 factor is held constant across all years. In reality it varies: CAPEX per TH has
    fallen significantly since 2016 as ASIC manufacturing has scaled and become more competitive.

    **2. Year-step efficiency table**
    The efficiency table advances in yearly steps. The real network average shifts continuously
    as new machines are added and old ones are retired month by month.

    **3. No slippage model**
    The backtest assumes trades execute at the closing price with no market impact.
    Large orders on a real exchange would move the price slightly, especially in thin markets.

    **4. Hashrate interpolation**
    For windows longer than ~2,000 days, Blockchain.com returns one data point every four days.
    Linear interpolation fills the gaps. Short-term hashrate spikes may be smoothed over.

    **5. Backtest period and forward caution**
    The strategy was designed and optimised on 2016–2026 data. It has not been run in live trading.
    Past backtest performance does not guarantee future returns.

    ---

    ### Potential Future Improvements

    - **Adaptive overhead factor:** Model CAPEX per TH over time using public ASIC price data from
      Hashrate Index ([https://hashrateindex.com/machines](https://hashrateindex.com/machines)).
    - **On-chain signals (post-2020 data):** SOPR and miner wallet flows become usable when a full
      training window is available.
    - **Live paper trading:** Running in paper-trade mode on a real exchange API to validate signal
      quality against live market conditions.
    - **Mining cost benchmarking:** The model was already compared against MacroMicro's public cost
      chart (scraped from the public page since the raw API is paid).
      Code: `compare_real_costs.py`.
    """)

    st.markdown("---")

    # ========== SOURCE INDEX ==========
    st.header("Complete Source Index")

    st.markdown("""
    ### Code Repository
    **GitHub:** [https://github.com/Luka24/btc_usdc_trading_strategy](https://github.com/Luka24/btc_usdc_trading_strategy)

    ### Key Files
    | File | Purpose |
    |---|---|
    | [config.py](https://github.com/Luka24/btc_usdc_trading_strategy/blob/main/config.py) | All strategy parameters |
    | [production_cost.py](https://github.com/Luka24/btc_usdc_trading_strategy/blob/main/production_cost.py) | Mining cost calculator |
    | [backtest.py](https://github.com/Luka24/btc_usdc_trading_strategy/blob/main/backtest.py) | Backtest engine |
    | [portfolio.py](https://github.com/Luka24/btc_usdc_trading_strategy/blob/main/portfolio.py) | Position sizing and rebalancing |
    | [risk_manager.py](https://github.com/Luka24/btc_usdc_trading_strategy/blob/main/risk_manager.py) | 4-mode risk engine |
    | [data_fetcher.py](https://github.com/Luka24/btc_usdc_trading_strategy/blob/main/data_fetcher.py) | API data pipeline |
    | [optimization/best_params.json](https://github.com/Luka24/btc_usdc_trading_strategy/blob/main/optimization/best_params.json) | Walk-forward optimised parameters |

    ---

    ### Research & Data Sources

    **Strategy & Mining Economics**
    1. Capriole Investments — Bitcoin Energy Value (TradingView indicator): [https://www.tradingview.com/script/xDMwWgRN-Bitcoin-Energy-Value/](https://www.tradingview.com/script/xDMwWgRN-Bitcoin-Energy-Value/)
    2. Capriole — Hash Ribbons & Bitcoin Bottoms: [https://capriole.com/hash-ribbons-bitcoin-bottoms/](https://capriole.com/hash-ribbons-bitcoin-bottoms/)
    3. CoinShares — Bitcoin Mining Network Report: [https://coinshares.com/research/bitcoin-mining-network](https://coinshares.com/research/bitcoin-mining-network)
    4. CESifo WP10145 — Economics of Bitcoin Mining: [https://www.cesifo.org/DocDL/cesifo1_wp10145.pdf](https://www.cesifo.org/DocDL/cesifo1_wp10145.pdf)

    **Energy & Hardware Data**
    5. Cambridge CBECI: [https://ccaf.io/cbnsi/cbeci](https://ccaf.io/cbnsi/cbeci)
    6. US EIA — Industrial Electricity Prices: [https://www.eia.gov/electricity/monthly/epm_table_grapher.php?t=epmt_5_6_a](https://www.eia.gov/electricity/monthly/epm_table_grapher.php?t=epmt_5_6_a)
    7. MacroMicro — Bitcoin Production Cost: [https://en.macromicro.me/charts/27916/bitcoin-production-cost](https://en.macromicro.me/charts/27916/bitcoin-production-cost)
    8. Hashrate Index (Luxor) — ASIC Hardware: [https://hashrateindex.com/machines](https://hashrateindex.com/machines)

    **Sentiment & On-Chain**
    9. Alternative.me — Fear & Greed Index: [https://alternative.me/crypto/fear-and-greed-index/](https://alternative.me/crypto/fear-and-greed-index/)
    10. Glassnode — SOPR: [https://glassnode.com/metrics/sopr](https://glassnode.com/metrics/sopr)

    **Data APIs**
    11. CoinGecko API: [https://www.coingecko.com/en/api/documentation](https://www.coingecko.com/en/api/documentation)
    12. Blockchain.com Hashrate: [https://api.blockchain.info/charts/hash-rate?timespan=all&format=json](https://api.blockchain.info/charts/hash-rate?timespan=all&format=json)

    **Exchange Fees**
    13. Binance Fee Schedule: [https://www.binance.com/en/fee/schedule](https://www.binance.com/en/fee/schedule)
    14. Coinbase Advanced Trade Fees: [https://www.coinbase.com/advanced-trade/fees](https://www.coinbase.com/advanced-trade/fees)

    **Academic**
    15. Liu, Tsyvinski & Wu (2019) — Cross-Section of Expected Crypto Returns: [https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3244646](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3244646)
    16. Bitcoin Halving Schedule: [https://en.bitcoin.it/wiki/Controlled_supply](https://en.bitcoin.it/wiki/Controlled_supply)
    """)

    st.markdown("---")
    st.caption(
        "All parameters reflect the codebase as of February 2026. "
        "Walk-forward optimised values are loaded from optimization/best_params.json at runtime "
        "and override the defaults in config.py. "
        "Fixed parameters are set directly in config.py and do not change between runs."
    )