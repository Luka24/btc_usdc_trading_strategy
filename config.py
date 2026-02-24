"""
Configuration for BTC/USDC Trading Strategy
============================================
Defines strategy parameters that can be easily modified for different scenarios.
"""

# ============ HISTORICAL PARAMETERS (Dynamic) ============
class HistoricalParameters:
    """
    Historical electricity prices and miner efficiency.
    Data sources:
    - Electricity: US EIA (Energy Information Administration) + industry reports
    - Miner efficiency: Network average (all miners, not just newest), from MacroMicro, JPMorgan reports
    
    IMPORTANT: This uses NETWORK AVERAGE efficiency, not the best miners.
    Network average includes old and new miners, hence higher values than specs.
    """
    
    # Historical US electricity prices ($/kWh) - realistic for medium-large miners
    # Includes: cooling losses, facility overhead (not pure generation cost)
    ELECTRICITY_PRICES_USD_PER_KWH = {
        2016: 0.10,  # Much higher in early days - smaller operations, less efficient
        2017: 0.10,  # Still high - rapid growth, inefficient setups
        2018: 0.08,
        2019: 0.07,
        2020: 0.065,
        2021: 0.065,
        2022: 0.07,  # Energy crisis
        2023: 0.065, # Stabilizing
        2024: 0.062, # Slightly higher pre-halving (old miners still running)
        2025: 0.06,
        2026: 0.06,
    }
    
    # NETWORK AVERAGE miner efficiency (J/TH)
    # NOT the best machines, but average of all machines operating on network
    # This is why the values are higher than individual spec sheets
    # Sources: MacroMicro, JPMorgan, CESifo reports
    NETWORK_AVERAGE_EFFICIENCY_J_PER_TH = {
        2016: 250,   # Lot of old S7, S8 still running
        2017: 150,   # Transition to S9
        2018: 110,
        2019: 85,
        2020: 60,    # S9 aging out, S17 coming in
        2021: 50,    # S9 mostly gone, S17/S19 dominant
        2022: 45,    # S19 series growing
        2023: 38,    # S19 XP, early S21
        2024: 30,    # Pre-halving: mix of old S19 and new S21, post-halving: forced upgrades
        2025: 24,    # S21 Pro widespread, old miners shut down
        2026: 22,    # Modern efficient fleet dominant
    }
    
    # Legacy parameter - keep for backwards compatibility
    MINER_EFFICIENCY_J_PER_TH = NETWORK_AVERAGE_EFFICIENCY_J_PER_TH
    
    @staticmethod
    def get_electricity_price(date) -> float:
        """
        Get electricity price for a given date (step function, no interpolation).
        Uses realistic prices for medium-large miners (including cooling, facility overhead).
        
        Args:
            date: datetime.date or pandas.Timestamp
            
        Returns:
            float: Electricity price in $/kWh
        """
        import pandas as pd
        if hasattr(date, 'year'):
            year = date.year
        else:
            year = pd.Timestamp(date).year
        
        prices = HistoricalParameters.ELECTRICITY_PRICES_USD_PER_KWH
        
        if year in prices:
            return prices[year]
        
        years = sorted(prices.keys())
        if year < years[0]:
            return prices[years[0]]
        if year > years[-1]:
            return prices[years[-1]]
        
        # Step function (use latest known year)
        return prices[max([y for y in years if y <= year])]
    
    @staticmethod
    def get_miner_efficiency(date) -> float:
        """
        Get network average miner efficiency for a given date.
        Uses network average (all miners), not just newest miners.
        
        Args:
            date: datetime.date or pandas.Timestamp
            
        Returns:
            float: Network average miner efficiency in J/TH
        """
        import pandas as pd
        
        if hasattr(date, 'year'):
            year = date.year
        else:
            year = pd.Timestamp(date).year
        
        efficiency = HistoricalParameters.NETWORK_AVERAGE_EFFICIENCY_J_PER_TH
        
        if year in efficiency:
            return efficiency[year]
        
        years = sorted(efficiency.keys())
        if year < years[0]:
            return efficiency[years[0]]
        if year > years[-1]:
            return efficiency[years[-1]]
        
        # Step function (use latest known year)
        return efficiency[max([y for y in years if y <= year])]


# ============ PRODUCTION COST PARAMETERS ============
class ProductionCostConfig:
    """
    Parameters for calculating BTC production cost.
    
    Uses simplified model based on research reports (MacroMicro, JPMorgan, CESifo, Visual Capitalist):
    - Energy cost = Network_Average_Efficiency * Hashrate * Electricity_Price
    - Total cost = Energy_Cost * OVERHEAD_FACTOR (covers CAPEX, maintenance, employees)
    """
    
    # Network efficiency and electricity price are now dynamic (see HistoricalParameters)
    
    # OVERHEAD_FACTOR: multiplier for all-in costs
    # If = 1.0: only electricity
    # If = 1.47: electricity is 68% of costs, other 32% are CAPEX + facilities + payroll
    # Accounts for hardware costs, cooling, facilities, personnel
    # Calibrated to match historical production cost estimates
    OVERHEAD_FACTOR = 1.47
    
    # Network constants
    BLOCKS_PER_DAY = 144  # ~144 blocks/day (10 min/block average)
    BLOCK_REWARD = 3.125  # BTC/block (after 2024 halving, April 20)
    # Note: Previous halving was April 2024 (6.25 → 3.125)

    # Halving schedule (date -> reward after halving)
    PRE_HALVING_REWARD = 50.0
    HALVING_SCHEDULE = [
        ("2012-11-28", 25.0),      # 50 → 25
        ("2016-07-09", 12.5),      # 25 → 12.5
        ("2020-05-11", 6.25),      # 12.5 → 6.25
        ("2024-04-20", 3.125),     # 6.25 → 3.125
        ("2028-04-20", 1.5625),    # 3.125 → 1.5625 (predicted)
    ]
    
    # Smoothing
    EMA_WINDOW = 14  # 14-day EMA for cost
    
    # Legacy parameters (kept for backwards compatibility but not used in new model)
    OPEX_PERCENTAGE = 0.00  # Deprecated - use OVERHEAD_FACTOR instead
    HARDWARE_DEPRECIATION = 0.00  # Deprecated - use OVERHEAD_FACTOR instead
    FEES_AND_EXCHANGE_BUFFER = 0.00  # Deprecated - use OVERHEAD_FACTOR instead


# ============ SIGNAL PARAMETERS ============
class SignalConfig:
    """Parameters for signal generation and decision rules"""
    
    # Price/Cost ratio bands for signal target weight (R_ema based)
    # Format: (min_ratio, max_ratio, btc_weight)
    POSITION_TABLE = [
        (0.00, 0.80, 1.00),
        (0.80, 0.90, 0.85),
        (0.90, 1.00, 0.70),
        (1.00, 1.10, 0.50),
        (1.10, 1.25, 0.30),
        (1.25, 10.0, 0.00),  # full USDC when severely overpriced
    ]

    # Legacy thresholds kept for compatibility in parts of existing code
    RATIO_BUY_THRESHOLD = 0.90
    RATIO_NEUTRAL_LOW = 0.90
    RATIO_NEUTRAL_HIGH = 1.10
    RATIO_SELL_THRESHOLD = 1.10

    # Smoothing per PRO specification
    PRICE_EMA_WINDOW = 21    # was 7  – 3-week EMA suppresses intraweek noise
    COST_EMA_WINDOW = 30     # was 14 – smoother production-cost baseline
    SIGNAL_EMA_WINDOW = 14   # blends hard band steps into gradual ramps


# ============ PORTFOLIO PARAMETERS ============
class PortfolioConfig:
    """Parameters for portfolio management and positioning"""
    
    # Position table (delegated to SignalConfig for single source of truth)
    POSITION_TABLE = SignalConfig.POSITION_TABLE
    
    # Rebalancing constraints
    MAX_DAILY_WEIGHT_CHANGE = 0.10  # max 10%/day – fast enough to catch dip recoveries
    MIN_REBALANCE_THRESHOLD = 0.04  # only rebalance if drift > 4% – filters noise trades

    # Volatility targeting – scale w_signal so portfolio vol ~= VOL_TARGET (annualized)
    # Formula: scalar = VOL_TARGET / realized_btc_vol, clipped to [VOL_SCALE_MIN, 1.0]
    # Set to 0.0 to disable.
    VOL_TARGET = 0.48            # target annualized portfolio volatility (48%) – grid-search optimum
    VOL_SCALING_WINDOW = 15      # days of BTC returns used for realized-vol estimate
    VOL_SCALE_MIN = 0.20         # floor: never cut position below 20% of signal weight
    VOL_SCALE_MAX = 1.00         # cap: never lever above signal weight

    # 200-day trend filter (institutional standard for regime detection)
    # When price < long EMA: bearish regime → hard cap on max allocation
    TREND_FILTER_WINDOW = 250    # grid-search optimal (250 > 200 for BTC cycles)
    TREND_BEAR_CAP = 0.00        # full exit in bearish regime – best for Sharpe & maxDD

    # RSI overlay – oversold boost only; suppress disabled (found to hurt Sharpe)
    RSI_WINDOW = 14
    RSI_OVERSOLD = 30            # RSI below this → multiply w_signal by RSI_BOOST
    RSI_OVERBOUGHT = 100         # set above max → effectively disabled
    RSI_BOOST = 1.30             # +30% position size when deeply oversold
    RSI_SUPPRESS = 1.00          # disabled (suppress hurt Sharpe in all sweep combos)

    # ──────────────────────────────────────────────────────────────────
    # MVRV Z-score overlay (on-chain valuation cycle signal)
    # Source: CoinMetrics CapMVRVCur (Community API – no API key needed)
    # Z-score = rolling Z-normalisation of MVRV ratio over MVRV_Z_WINDOW days
    #
    # Historical cycle reference (2017–2024 BTC data):
    #   MVRV Z < 0   → extreme undervaluation (perfect buy zone)
    #   MVRV Z 0–2  → accumulation / fair-value
    #   MVRV Z 2–4  → caution / distribution begins
    #   MVRV Z 4–6  → late-cycle / reduce exposure
    #   MVRV Z > 6  → cycle top (2017 peak: ~9, 2021 peak: ~8)
    # ──────────────────────────────────────────────────────────────────
    MVRV_ENABLED         = False   # Tested 2022-2026: no Sharpe improvement (risk mgr already handles cycle signals)
    MVRV_Z_WINDOW        = 730    # 2-year rolling Z-score window
    # Signal thresholds (grid-searched on 2022-2026 BTC data)
    MVRV_OVERSOLD_Z      = -0.5   # below → undervaluation boost (+Sharpe)
    MVRV_CAUTION_Z       =  2.0   # above → caution zone (key: max MVRV Z in data ~3.7)
    MVRV_RISK_OFF_Z      =  4.5   # above → risk-off zone (rarely triggers in 2022-2026)
    MVRV_EXTREME_Z       =  6.5   # above → cycle-top zone (2017 top: ~9, 2021 top: ~8)
    # Multipliers applied to w_signal (after RSI overlay, before risk caps)
    MVRV_BOOST           = 1.15   # +15% when in deep undervaluation (MVRV Z < -0.5)
    MVRV_CAUTION_FACTOR  = 0.70   # -30% when MVRV Z > 2.0 (key suppression in bull mkt)
    MVRV_RISK_OFF_FACTOR = 0.55   # -45% in risk-off zone (rarely triggers)
    MVRV_EXTREME_FACTOR  = 0.25   # -75% at cycle tops (history: 2017, 2021 peaks)

    # ── Monthly Seasonality overlay ────────────────────────────────────────
    # Full history validation (yfinance 2014-2026, n=11):
    #   Aug: 18% win rate (9/11 years negative), avg +1.0% (mean pulled by
    #        2017 +73% and 2021 +18% bull runs). OOS walk-forward +0.036 Sharpe.
    #   CAUTION: 4-year sample 2022-2025 was 4/4 negative → biased window.
    #        Broader history shows bull-run Augusts exist → use mild multiplier.
    #   aug=0.50 is too aggressive given 2017/2021 tail risk.
    #   aug=0.70 is robust: reduces exposure for typical bear August (-5...-18%)
    #        while preserving ~70% upside in rare bull August regime.
    #   Jun: 55% win rate over 11 years, avg +1.6%. Signal NOT statistically
    #        significant. Removed.
    # Multipliers applied to w_signal AFTER RSI, BEFORE trend/risk caps.
    # ──────────────────────────────────────────────────────────────────────
    SEASONAL_ENABLED = True
    SEASONAL_MULTIPLIERS = {
        # month → multiplier (1.0 = neutral; only include statistically robust months)
        # June REMOVED: 55% win rate, avg +1.6% over 11 years → no signal
        8:  0.70,   # August: 18% win rate (9/11 yrs neg). Conservative vs aug=0.50
                    # to avoid over-cutting in rare bull Augusts (2017 +73%, 2021 +18%)
    }
    
    # Costs
    TRADING_FEES_PERCENT = 0.001  # 0.1% fee (maker/taker average)
    SLIPPAGE_PERCENT = 0.002     # 0.2% slippage
    
    # Initial portfolio
    INITIAL_PORTFOLIO_USD = 100_000  # Starting capital


# ============ RISK MANAGEMENT PARAMETERS ============
class RiskManagementConfig:
    """Parameters for risk monitoring and control"""

    # Core windows (industry-standard in spec)
    ROLLING_PEAK_WINDOW = 252
    VOLATILITY_WINDOW = 30
    VAR_LOOKBACK = 30
    VAR_CONFIDENCE = 0.99
    VAR_ZSCORE = 2.33

    # Mode caps
    # CAUTION/RISK_OFF raised to allow meaningful exposure when market is merely elevated not collapsing
    MODE_CAPS = {
        "NORMAL": 1.00,
        "CAUTION": 0.75,   # was 0.60
        "RISK_OFF": 0.45,  # was 0.30
        "EMERGENCY": 0.05,
    }

    # Entry thresholds (instant downgrade) – calibrated for BTC (long-run vol ~65–80%)
    DD_THRESHOLDS = {
        "CAUTION": -0.12,   # was -0.08  (BTC can drop 8% in a single bad week)
        "RISK_OFF": -0.20,  # was -0.15
        "EMERGENCY": -0.35, # was -0.25
    }
    VOL_THRESHOLDS = {
        "CAUTION": 0.75,    # was 0.55  (BTC baseline ~65%, so 75% = genuinely elevated)
        "RISK_OFF": 1.00,   # was 0.75
        "EMERGENCY": 1.40,  # was 1.10
    }
    VAR_THRESHOLDS = {
        "CAUTION": 0.04,    # was 0.03
        "RISK_OFF": 0.06,   # was 0.05
        "EMERGENCY": 0.09,  # was 0.07
    }

    # Recovery thresholds – proportionally loosened so recovery is reachable for BTC
    RECOVERY_THRESHOLDS = {
        "CAUTION": {"dd": -0.09, "vol": 0.65, "var": 0.025},  # was -0.05 / 0.45 / 0.01
        "RISK_OFF": {"dd": -0.16, "vol": 0.85, "var": 0.04},  # was -0.12 / 0.65 / 0.03
        "EMERGENCY": {"dd": -0.28, "vol": 1.20, "var": 0.06}, # was -0.22 / 1.00 / 0.05
    }

    # Required consecutive days for upward recovery – faster to avoid permanent lock-in
    RECOVERY_DAYS = {
        "CAUTION": 2,   # was 3
        "RISK_OFF": 3,  # was 5
        "EMERGENCY": 5, # was 7
    }

    # Legacy compatibility constants (deprecated by mode engine)
    MAX_DRAWDOWN_THRESHOLD = 0.20
    RISK_OFF_MODE_BTC_MAX = 0.30
    VOLATILITY_HIGH_THRESHOLD = 0.80
    VAR_LIMIT_PERCENT = 0.04


# ============ BACKTEST PARAMETERS ============
class BacktestConfig:
    """Parameters for backtesting"""
    
    # Data source
    USE_REAL_DATA = True  # Must remain True for production/backtest runs
    STRICT_REAL_DATA = True  # If True, synthetic fallback paths raise errors
    DAYS_TO_FETCH = 3650  # Number of days to fetch from API (10 years)
    
    # Period
    START_DATE = "2016-02-01"  # 10 years of data
    END_DATE = "2026-02-01"
    
    # Rebalancing frequency
    REBALANCE_FREQUENCY = "D"  # "D" = daily, "W" = weekly
    REBALANCE_TIME = "00:00"   # UTC rebalancing time
    
    # Metrics to compute
    METRICS_TO_COMPUTE = [
        "total_return",
        "cagr",
        "sharpe_ratio",
        "sortino_ratio",
        "max_drawdown",
        "win_rate",
        "trade_count",
        "turnover",
    ]
