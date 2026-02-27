import streamlit as st


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

    st.title("Strategy & Methodology - Complete Documentation")

    st.markdown('<p style="font-size:18px;color:#555;">Most of this page is already up to date. Full update in progress — will be completed by end of February 27, 2026.</p>', unsafe_allow_html=True)

    st.markdown(
        """
This page explains how the trading strategy works. You can read about the main ideas, where I got the data, 
and why I made certain choices when building this system.
"""
    )

    st.markdown("---")

    # ========== HOW I STARTED ==========
    st.header("0️⃣ How I Started the Project")

    st.markdown("""
    ### How I Started the Task
    
    At the start, this task looked very interesting to me. I always liked to analyze numbers.
    
    **First step: research**
    - I asked: what is the main point and what is the problem?
    - I already knew the basic idea of Bitcoin.
    - I read some pages on Google and watched a few YouTube videos.
    - I also asked Gemini for help.
    
    **Second step: build a simple base**
    - I first made a static report with all the main parts.
    - I used the theory I read before.
    - I used AI tools to help write code because the project is big and time was short.
    - I always checked the AI output and fixed it if it did not make sense.
    
    **Third step: add more features**
    - I saw I could make it more complex.
    - I added live data with APIs.
    - I built a dynamic Streamlit dashboard.
    
    **What I learned**
    - I learned a lot about trading theory.
    - I also learned a lot about programming.
    
    **Ideas for the future**
    - I have many ideas to improve the project, as I said in the video I included.
    """)

    st.markdown("---")

    # ========== STRATEGY OVERVIEW ==========
    st.header("1️⃣ Strategy Overview & Theoretical Foundation")
    
    st.markdown("""
    ### Core Concept: Mean-Reversion Based on Production Cost
    
    This strategy uses Bitcoin's **production cost** (how much it costs to mine) as a reference point. 
    
    **Main idea:** When Bitcoin's price goes too far away from the mining cost, it usually comes back. We can use this to know when to buy or sell.
    
    ---
    
    ### Where I Got My Ideas
    
    I read many sources before building this. Here are the main ones:
    
    **1. Charles Edwards - Capriole Investments: "Bitcoin Energy Value"**
    - **Link:** https://capriole.io/
    - **What I learned:** Production cost works like a floor when prices are falling and a ceiling when prices get too high
    - **How I used it:** This helped me decide the buy and sell thresholds
    
    **2. JPMorgan Bitcoin Mining Analysis (2021-2024)**
    - **Link:** https://www.jpmorgan.com/insights/research
    - **What I learned:** Mining costs and Bitcoin price are strongly connected
    - **How I used it:** This made me more confident that production cost is a good reference point
    
    **3. CESifo Working Paper: "The Economics of Bitcoin Mining"**
    - **Link:** https://www.cesifo.org/DocDL/cesifo1_wp10145.pdf
    - **What I learned:** Academic research shows mining costs create natural price limits
    - **How I used it:** Gave me the economic reason for using mean reversion
    
    **4. Cambridge Centre for Alternative Finance**
    - **Link:** https://ccaf.io/cbnsi/cbeci
    - **What I used:** Their data on how much electricity Bitcoin mining uses
    
    **5. Visual Capitalist Mining Cost Research**
    - **Link:** https://www.visualcapitalist.com/
    - **What I used:** Historical electricity prices and how efficient miners were over time
    
    ---
    
    ### Why This Works
    
    **Economic reasons:**
    1. **When price is too low:** Miners lose money and stop mining → less Bitcoin sold → price goes up
    2. **When price is too high:** More people start mining → more Bitcoin sold → price goes down
    3. **Money flow:** When mining is profitable, more money comes in. When it's not, money leaves.
    
    **Psychology:**
    - People think of production cost as the "fair price"
    - When the price moves too far away from this, traders start buying or selling to bring it back
    
    ---
    
    ### Trading Rules
    
    **Price / Production Cost Ratio:**
    
    $$\\text{Ratio} = \\frac{\\text{BTC Market Price}}{\\text{Production Cost (EMA Smoothed)}}$$
    
    **When to trade:**
    - **BUY:** Ratio < 0.90 → Bitcoin is cheap (price is below 90% of mining cost)
    - **SELL:** Ratio > 1.10 → Bitcoin is expensive (price is above 110% of mining cost)
    - **HOLD:** 0.90 ≤ Ratio ≤ 1.10 → Price is normal
    
    **Settings:**
    - `RATIO_BUY_THRESHOLD = 0.90`
    - `RATIO_SELL_THRESHOLD = 1.10`
    - Code location: [config.py → SignalConfig](https://github.com/Luka24/btc_usdc_trading_strategy/blob/main/config.py)
    
    **Why did I choose ±10%?**
    - **Testing:** I tested the strategy from 2016 to 2024 and ±10% worked best
    - **Too wide (±20%):** We miss good chances to buy or sell
    - **Too narrow (±5%):** We trade too much and pay too many fees. Also get false signals.
    - **Others use it too:** Similar to what Capriole uses in their model
    - **Filters noise:** ±10% helps ignore small daily price movements
    """)
    
    st.markdown("---")
    
    # ========== DATA SOURCES ==========
    st.header("2️⃣ Where I Get My Data")
    st.markdown("""
    ### Bitcoin Price
    
    **Main source:** CoinGecko API (free, but only gives 365 days)
    - Docs: https://www.coingecko.com/en/api/documentation
    - Use for: Recent prices (less than 1 year)
    
    **Backup source:** Yahoo Finance (using yfinance library)
    - Data: https://finance.yahoo.com/quote/BTC-USD/history
    - Use for: Old data (more than 1 year)
    - Why: It's free, no limits, and has data going back to 2014
    
    ### Network Hashrate
    
    **Source:** Blockchain.com Charts API
    - API: https://api.blockchain.info/charts/hash-rate?timespan=all&format=json
    - Data: Daily hashrate (mining power) since 2010
    - Why: Free, easy to use, has all the history I need
    
    ### Technical Details
    
    - Code: [data_fetcher.py](https://github.com/Luka24/btc_usdc_trading_strategy/blob/main/data_fetcher.py)
    - Saves data: I save data locally to avoid asking for it too many times
    - Updates: Dashboard refreshes data every hour
    """)
    
    st.markdown("---")
    
    # ========== PRODUCTION COST MODEL ==========
    st.header("3️⃣ How I Calculate Mining Cost")
    
    st.markdown("""
    ### The Math
    
    **Step 1: How much energy miners use per day**
    
    $$\\text{Total Hashes/Day} = \\text{Hashrate (H/s)} \\times 86400 \\text{ seconds}$$
    
    $$\\text{Energy (Joules)} = \\text{Total Hashes} \\times \\frac{\\text{Efficiency (J/TH)}}{10^{12}}$$
    
    $$\\text{Energy (kWh)} = \\frac{\\text{Energy (J)}}{3.6 \\times 10^6}$$
    
    **Step 2: Cost of that energy**
    
    $$\\text{Energy Cost} = \\text{Energy (kWh)} \\times \\text{Electricity Price (\\$/kWh)}$$
    
    **Step 3: How much Bitcoin gets mined per day**
    
    $$\\text{BTC/Day} = \\text{Blocks/Day} \\times \\text{Block Reward}$$
    
    - Blocks/Day = 144 (one block every 10 minutes)
    - Block Reward = Changes with halvings (3.125 BTC after April 2024)
    
    **Step 4: Energy cost for one Bitcoin**
    
    $$\\text{Energy Cost per BTC} = \\frac{\\text{Total Energy Cost}}{\\text{BTC/Day}}$$
    
    **Step 5: Total cost (including everything)**
    
    $$\\text{Total Cost per BTC} = \\text{Energy Cost per BTC} \\times \\text{OVERHEAD\\_FACTOR}$$
    
    ---
    
    ### About the Overhead Factor
    
    **OVERHEAD_FACTOR = 1.47**
    
    **What does this mean?** Energy is 68% of mining costs. The other 32% are other expenses.
    
    **What else do miners pay for:**
    - **Hardware:** Mining machines get old and break (last 2-3 years)
    - **Buildings:** Cooling, electricity setup, rent
    - **People:** Workers who fix and manage everything
    - **Other stuff:** Insurance, internet, mining pool fees
    
    **Where I got 1.47:**
    - **JPMorgan report (2022):** They say energy is 65-70% of all costs
    - **MacroMicro:** https://en.macromicro.me/ (Mining cost details from Taiwan)
    - **Mining companies:** Reports from Compass Mining and Luxor Mining Pool
    
    **Why not just use energy costs?**
    - That would be wrong → bad trading signals
    - Real miners need to pay for all these things, not just electricity
    
    ---
    
    ### Electricity Prices Over Time

    
    **Interactive Tables:**
    """)
    
    import pandas as pd
    
    col1, col2 = st.columns(2)
    
    electricity_data = {
        'Year': [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026],
        'Price ($/kWh)': [0.10, 0.10, 0.08, 0.07, 0.065, 0.065, 0.07, 0.065, 0.062, 0.06, 0.06],
    }
    
    efficiency_data = {
        'Year': [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026],
        'Efficiency (J/TH)': [250, 150, 110, 85, 60, 50, 45, 38, 30, 24, 22],
    }
    
    with col1:
        st.subheader("Electricity Prices")
        df_electricity = pd.DataFrame(electricity_data)
        st.dataframe(df_electricity, use_container_width=True, hide_index=True)
        st.caption("Network average electricity costs for miners")
    
    with col2:
        st.subheader("Network Efficiency")
        df_efficiency = pd.DataFrame(efficiency_data)
        st.dataframe(df_efficiency, use_container_width=True, hide_index=True)
        st.caption("Network average miner efficiency (all miners, not just newest)")
    
    st.markdown("""
    
    **Where I got this data:**
    - **US EIA:** https://www.eia.gov/ (US industrial electricity prices)
    - **Cambridge:** https://ccaf.io/cbnsi/cbeci (Bitcoin electricity use index)
    - **Mining companies:** Compass Mining, Luxor Mining Pool price data
    
    **Note:** These prices include everything - electricity, cooling, building costs. Not just the basic electricity price.
    
    ---
    
    ### Mining Machine Efficiency (J/TH)
    
    **Why "Network Average"?**
    
    The network has old and new machines. So the average is worse than the newest machines.
    
    **Values I used:**
    
    | Year | J/TH | Main Miners | Source |
    |------|------|-------------|--------|
    | 2016 | 250 | S7, S8 (old machines) | MacroMicro, CESifo |
    | 2017 | 150 | Moving to S9 | Mining pool data |
    | 2018 | 110 | S9 everywhere | Company specs |
    | 2019 | 85 | S9 getting old | Network guesses |
    | 2020 | 60 | S17 starting | JPMorgan report |
    | 2021 | 50 | S19 series | Pool data |
    | 2022 | 45 | More S19 | Industry reports |
    | 2023 | 38 | S19 XP, first S21 | Cambridge data |
    | 2024 | 30 | Mix of S19/S21 | After halving |
    | 2025 | 24 | S21 Pro common | My estimate |
    | 2026 | 22 | Modern machines | Current guess |
    
    **Where I got this:**
    - **MacroMicro:** https://en.macromicro.me/ (Network efficiency guesses)
    - **JPMorgan:** Reports on network efficiency
    - **CESifo Paper:** Academic estimates
    - **Luxor:** https://hashrateindex.com/
    
    **In the code:** [config.py → HistoricalParameters](https://github.com/Luka24/btc_usdc_trading_strategy/blob/main/config.py)
    
    ---
    
    ### Smoothing the Data
    
    **Setting:** `EMA_WINDOW = 14 days`
    
    **Why I do this:** To remove daily noise from the cost calculation
    
    **Why 14 days?**
    - Good balance between quick and stable
    - I tried 7, 14, 21, and 30 days
    - 14 days gave the best signals (less fake signals)
    - People often use 14 days in trading (2-week cycle)
    
    **In the code:** [config.py → ProductionCostConfig](https://github.com/Luka24/btc_usdc_trading_strategy/blob/main/config.py)
    """)
    
    st.markdown("---")
    
    # ========== PORTFOLIO & POSITION SIZING ==========
    st.header("4️⃣ Portfolio Management & Position Sizing")
    
    st.markdown("""
    ### Position Table (How Much Bitcoin to Hold)
    
    I change how much Bitcoin I hold based on the Price/Cost ratio:
    
    | Ratio Range | BTC Weight | Why |
    |-------------|------------|-----|
    | < 0.85 | 100% | **Super cheap** → Hold max Bitcoin |
    | 0.85 - 0.95 | 70% | **Pretty cheap** → Hold lots of Bitcoin |
    | 0.95 - 1.05 | 50% | **Normal price** → Hold 50/50 |
    | 1.05 - 1.20 | 30% | **Getting expensive** → Sell some Bitcoin |
    | > 1.20 | 5% | **Super expensive** → Hold almost no Bitcoin |
    
    **Why I designed it this way:**
    
    - **Slow changes:** Don't go all-in or all-out at once → less risk
    - **Always some Bitcoin:** Keep at least 5% in case price suddenly goes up
    - **50/50 middle ground:** Less trading when price is normal
    - **No borrowing:** Maximum 100%, I don't use leverage
    
    **Idea from:** Risk parity strategies
    - Research: "Risk Parity" (Bridgewater Associates)
    - I adapted it for mean-reversion
    
    **In the code:** [config.py → PortfolioConfig.POSITION_TABLE](https://github.com/Luka24/btc_usdc_trading_strategy/blob/main/config.py)
    
    ---
    
    ### Trading Limits
    
    **Setting:** `MAX_DAILY_WEIGHT_CHANGE = 0.25` (25%)
    
    **Why:** Stop big sudden changes in the portfolio
    
    **Why 25%?**
    - Stops bad signals from making big mistakes
    - I tried 10%, 25%, 50%
    - 25% was good - fast enough but not too much
    - Saves on trading fees
    
    **Example:**
    - I have 50% Bitcoin now
    - Signal says go to 100%
    - I only go to 75% today (can only add +25%)
    - Tomorrow I can add another +25% if signal still says so
    
    **Common in trading:** People use this in professional trading
    
    ---
    
    ### Starting Values
    
    **Starting money:** $100,000
    
    **Starting Bitcoin:** Up to 2.0 BTC (depends on price)
    
    **How it works:**
    - If 2 BTC × First_Price ≤ $100k → Start with 2 BTC + rest in USDC
    - If 2 BTC × First_Price > $100k → Buy what $100k can buy
    
    **Why like this?**
    - Same starting value no matter what dates I test
    - No borrowing money (safe)
    - Good for medium-size trader
    
    **In the code:** [backtest.py](https://github.com/Luka24/btc_usdc_trading_strategy/blob/main/backtest.py), [portfolio.py](https://github.com/Luka24/btc_usdc_trading_strategy/blob/main/portfolio.py)
    
    ---
    
    ### Trading Costs
    
    **Trading Fee:** `0.1%` for each trade (0.001 as decimal)
    
    **Where I got this:**
    - **Coinbase Pro:** 0.05% - 0.50% (depends on volume)
    - **Kraken:** 0.16% - 0.26%
    - **Binance:** 0.10% (average)
    - **I chose:** 0.1% as a safe middle value
    
    **Slippage:** `0.2%` (I defined it but don't use it yet)
    
    **In the code:** [config.py → PortfolioConfig](https://github.com/Luka24/btc_usdc_trading_strategy/blob/main/config.py)
    """)
    
    st.markdown("---")
    
    # ========== ML OPTIMIZATION ATTEMPTS ==========
    st.header("5️⃣ I Tried Using Machine Learning")
    
    st.markdown("""
    ### What I Tried
    
    **Goal:** Make the strategy better by using machine learning to find the best settings.
    
    ---
    
    ### Try 1: Ridge Regression
    
    **What I did:**
    - Used 10 years of data (2014-2024)
    - Made features: Ratio, moving averages, volatility, momentum
    - Tried to predict: 7-day, 14-day, 30-day future returns
    - Model: Ridge regression with cross-validation
    
    **What happened:**
    - **Model score:** 0.02 - 0.08 (very bad at predicting)
    - **New thresholds:** BUY at 0.87, SELL at 1.13 (vs my 0.90/1.10)
    - **Performance:** Only 2-3% better
    - **My decision:** Not worth it, too complicated for small gain
    
    **Why it didn't work well:**
    - Bitcoin is very random and always changing
    - My simple ratio already works pretty good
    - Risk of overfitting to old data
    
    **What I learned:**
    - In crypto where everything changes, simple strategies often beat complex ML
    - Production cost is a strong enough signal by itself
    
    **Research backs this up:**
    - "The Cross-Section of Expected Crypto Returns" (Liu, Tsyvinski, Wu, 2019)
    - Finding: Simple strategies beat complex ML in crypto
    
    **The code:** [optimization/ml_parameter_optimizer.py](https://github.com/Luka24/btc_usdc_trading_strategy/blob/main/optimization/ml_parameter_optimizer.py)
    """)
    
    st.markdown("---")
    
    # ========== RISK METRICS ==========
    st.header("6️⃣ Risk Management & Protection Systems")

    st.markdown("""
    ### How the Risk Engine Works

    The strategy uses a **4-mode professional risk engine** that is always active.
    It looks at three numbers every day — drawdown, volatility, and VaR — and decides which mode to be in.
    Each mode puts a hard cap on how much BTC the portfolio can hold.

    ---

    ### The 4 Modes

    | Mode | Max BTC Exposure | When It Activates |
    |---|---|---|
    | **NORMAL** | 100% | Everything looks fine |
    | **CAUTION** | 75% | Drawdown < −12%, or vol > 75%, or VaR > 4% |
    | **RISK_OFF** | 45% | Drawdown < −20%, or vol > 100%, or VaR > 6% |
    | **EMERGENCY** | 5% | Drawdown < −35%, or vol > 140%, or VaR > 9% |

    **Example:** If Bitcoin drops 22% from its recent peak, the engine switches to RISK_OFF and the portfolio can hold at most 45% BTC. This limits losses automatically.

    ---

    ### Sticky Recovery

    The engine does **not jump back to NORMAL immediately** when things improve.
    It waits until drawdown, volatility, *and* VaR all return to safe levels, **and** it counts several calm days first:

    - CAUTION → needs 2 calm days before upgrading
    - RISK_OFF → needs 3 calm days
    - EMERGENCY → needs 5 calm days

    **Why?** Markets often bounce briefly after a crash. Sticky recovery stops the strategy from going back to full exposure too early.

    ---

    ### Volatility Scaling (always on)

    Even inside NORMAL mode, position size is scaled by volatility:  
    $$\\text{Scale} = \\frac{\\text{Vol Target (40\\% ann.)}}{\\text{Realised Vol (15-day)}}$$

    The scale is capped between 0.20 and 1.00, so the strategy never levers up and never goes below 20% exposure.

    **Why?** This keeps risk roughly constant over time. In calm markets we hold more; in wild markets we hold less.

    ---

    ### Hash Ribbon Filter (miner capitulation)

    The Hash Ribbon compares a **30-day** and **60-day** moving average of the Bitcoin network hashrate.  
    When the fast SMA drops below the slow SMA, miners are shutting down — this is a danger sign.

    - Fast SMA < Slow SMA → strategy caps exposure to 0% (full exit)
    - Once miners recover (fast > slow), normal trading resumes

    **Why?** Miner capitulation often precedes large price drops (e.g. 2018, 2022). This filter was validated in walk-forward testing (+0.27 OOS Sortino improvement).

    ---

    ### Trend Filter (Bull / Bear)

    The strategy checks if the current BTC price is above its 250-day EMA:
    - Above EMA → normal trading
    - Below EMA → exposure capped at 0% (bear market protection)

    This stops the strategy from buying into prolonged bear markets.

    ---

    ### RSI Oversold Boost

    When RSI (14-day) drops below 30, the target BTC weight is multiplied by **×1.30**.
    This adds a small extra buy when Bitcoin is deeply oversold — a classic mean-reversion signal.

    ---

    ### Risk Metrics Tracked

    **Volatility (30-day)**
    - Formula: $\\text{Vol} = \\text{std}(\\text{daily returns}) \\times \\sqrt{252}$
    - Measures how much risk the portfolio carries right now

    **Value-at-Risk (99% confidence)**
    - Formula: $\\text{VaR}_{99} = \\mu - 2.33 \\times \\sigma$
    - Worst daily loss expected 99 out of 100 days

    **Max Drawdown**
    - Formula: $\\text{DD} = \\frac{\\text{Value} - \\text{Peak}}{\\text{Peak}}$
    - Biggest fall from any previous high point

    **Sharpe / Sortino Ratios**
    - Sharpe: $\\frac{\\mu}{\\sigma} \\times \\sqrt{252}$ — return per unit of total risk
    - Sortino: same but only counts downside volatility (more relevant for crypto)

    **Code:** [risk_manager.py](https://github.com/Luka24/btc_usdc_trading_strategy/blob/main/risk_manager.py)
    """)
    
    st.markdown("---")
    
    # ========== LIMITATIONS ==========
    st.header("7️⃣ What's Missing & What I Assume")
    
    st.markdown("""
    ### Model Problems
    
    **1. Mining Cost Simplifications**
    - Uses average efficiency, not real miner mix
    - Overhead factor is fixed but it changes over time
    
    **2. Trading Assumptions**
    - Assumes instant execution at market price
    - Doesn't model slippage or partial fills
    
    **3. Transaction Costs**
    - Fixed 0.1% fee regardless of trade size
    
    **4. Risk Management Limitations**
    - Simple percentages (could be adaptive)
    - No protection against flash crashes
    
    **5. Data Quality**
    - Free public data may have gaps
    - Electricity prices are estimates
    
    ### Potential Improvements (Future Work)
    
    - **ML price prediction:** Try ML to predict fair price and find best parameters (ratio thresholds, stop-loss, take-profit)
    - **Sentiment analysis:** Add social media sentiment as extra signal
    - **Hard part:** Mining cost prediction is very hard, model is noisy and uncertain
    - **Benchmarking:** I compared my model with MacroMicro data (paid API) so I scraped the public chart and tuned parameters
    - **Code reference:** I compare real costs in [compare_real_costs.py](compare_real_costs.py)
    
    """)
    
    st.markdown("---")
    
    # ========== FULL SOURCE INDEX ==========
    st.header("Complete Source Index")
    
    st.markdown("""
    ### Code
    - **GitHub:** https://github.com/Luka24/btc_usdc_trading_strategy
    
    ### Main Files
    - [config.py](https://github.com/Luka24/btc_usdc_trading_strategy/blob/main/config.py) - All settings
    - [backtest.py](https://github.com/Luka24/btc_usdc_trading_strategy/blob/main/backtest.py) - Testing the strategy
    - [risk_manager.py](https://github.com/Luka24/btc_usdc_trading_strategy/blob/main/risk_manager.py) - Risk tracking
    
    ### Key Research Sources
    
    **Strategy & Mining Cost:**
    1. Capriole Investments - Bitcoin Energy Value: https://capriole.io/
    2. JPMorgan Bitcoin Research: https://www.jpmorgan.com/insights/research
    3. Cambridge CCAF - Bitcoin Electricity: https://ccaf.io/cbnsi/cbeci
    
    **Risk Management:**
    4. Binance Academy - Risk Management: https://academy.binance.com/
    5. Investopedia - Stop-Loss: https://www.investopedia.com/terms/s/stop-lossorder.asp
    6. Investopedia - Risk/Reward: https://www.investopedia.com/terms/r/riskrewardratio.asp
    7. Schwab - Trailing Stops: https://www.schwab.com/
    
    **Data Sources:**
    8. CoinGecko API: https://www.coingecko.com/en/api/documentation
    9. Blockchain.com Charts: https://api.blockchain.info/charts/hash-rate
    10. Bitcoin Volatility Index: https://www.buybitcoinworldwide.com/volatility-index/
    """)
    
    st.markdown("---")
    st.info("**Note:** All values and sources current as of February 2026. I built this by reading research and testing different ideas.")
