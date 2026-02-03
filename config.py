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
    
    # Price to Cost ratio thresholds
    RATIO_BUY_THRESHOLD = 0.90  # Buy when price < 90% of cost
    RATIO_NEUTRAL_LOW = 0.90    # Lower bound of neutral zone
    RATIO_NEUTRAL_HIGH = 1.10   # Upper bound of neutral zone
    RATIO_SELL_THRESHOLD = 1.10  # Sell when price > 110% of cost
    
    # Signal smoothing
    SIGNAL_EMA_WINDOW = 7  # 7-day EMA to reduce noise
    
    # Filtering
    PRICE_EMA_WINDOW = 14  # Price smoothing for stability
    COST_EMA_WINDOW = 14   # Cost smoothing


# ============ PORTFOLIO PARAMETERS ============
class PortfolioConfig:
    """Parameters for portfolio management and positioning"""
    
    # Position table based on ratio
    # Format: (min_ratio, max_ratio, btc_weight)
    POSITION_TABLE = [
        (0.0,   0.85, 1.00),  # Ratio < 0.85  → 100% BTC
        (0.85,  0.95, 0.70),  # 0.85–0.95    → 70% BTC
        (0.95,  1.05, 0.50),  # 0.95–1.05    → 50% BTC
        (1.05,  1.20, 0.30),  # 1.05–1.20    → 30% BTC
        (1.20,  10.0, 0.05),  # > 1.20       → 5% BTC (only reserve)
    ]
    
    # Rebalancing constraints
    MAX_DAILY_WEIGHT_CHANGE = 0.25  # Max 25% change per day
    MIN_REBALANCE_THRESHOLD = 0.02  # Only rebalance if change > 2%
    
    # Costs
    TRADING_FEES_PERCENT = 0.001  # 0.1% fee (maker/taker average)
    SLIPPAGE_PERCENT = 0.002     # 0.2% slippage
    
    # Initial portfolio
    INITIAL_PORTFOLIO_USD = 100_000  # Starting capital


# ============ RISK MANAGEMENT PARAMETERS ============
class RiskManagementConfig:
    """Parameters for risk monitoring and control"""
    
    # Drawdown control
    MAX_DRAWDOWN_THRESHOLD = 0.20  # Max 20% drawdown from peak
    RISK_OFF_MODE_BTC_MAX = 0.30   # In risk-off: max 30% BTC
    RECOVERY_THRESHOLD = 0.95      # Return to normal at 95% of peak
    
    # Volatility
    VOLATILITY_WINDOW = 30         # 30-day volatility
    VOLATILITY_HIGH_THRESHOLD = 0.80  # 80% annualized volatility threshold
    VOLATILITY_REDUCTION = 0.30    # Reduce BTC by 30% when exceeded
    VOLATILITY_NORMAL_THRESHOLD = 0.60  # Normalization below 60%
    
    # Value-at-Risk (VaR)
    VAR_CONFIDENCE = 0.99          # 99% confidence level
    VAR_LOOKBACK = 30              # 30-day lookback
    VAR_LIMIT_PERCENT = 0.04       # VaR limit 4% of portfolio
    
    # Regime filter
    SMA_WINDOW_REGIME = 200        # 200-day SMA for regime detection
    BEAR_MARKET_REDUCTION = 0.30   # Reduce BTC by 30% in bear trend
    
    # Liquidity
    MIN_VOLUME_24H_USD = 300_000_000  # Min 300M USDC volume
    MAX_SPREAD_BPS = 20            # Max 20 bps spread
    
    # Stop-loss and take-profit
    SOFT_STOP_LOSS_PERCENT = -0.12  # -12% from entry → reduce 50%
    SOFT_TAKE_PROFIT_PERCENT = 0.20  # +20% from entry → realize 50%


# ============ BACKTEST PARAMETERS ============
class BacktestConfig:
    """Parameters for backtesting"""
    
    # Data source
    USE_REAL_DATA = True  # True = API data, False = synthetic data
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
