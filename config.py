"""Configuration for BTC/USDC Trading Strategy"""

import pandas as pd


def _year_lookup(table: dict, date) -> float:
    """Step-function lookup: return value for the year of `date`."""
    year = date.year if hasattr(date, 'year') else pd.Timestamp(date).year
    if year in table:
        return table[year]
    years = sorted(table)
    if year < years[0]:
        return table[years[0]]
    if year > years[-1]:
        return table[years[-1]]
    return table[max(y for y in years if y <= year)]


class HistoricalParameters:
    # US electricity prices ($/kWh) for medium-to-large mining operations,
    # including cooling and facility overhead (not just generation cost).
    # Sources: US EIA, industry reports.
    
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
    
    # Network-average miner efficiency (J/TH) — all machines on the network,
    # not just the latest generation. Sources: MacroMicro, JPMorgan, CESifo.
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
        2026: 22,
    }

    @staticmethod
    def get_electricity_price(date) -> float:
        return _year_lookup(HistoricalParameters.ELECTRICITY_PRICES_USD_PER_KWH, date)

    @staticmethod
    def get_miner_efficiency(date) -> float:
        return _year_lookup(HistoricalParameters.NETWORK_AVERAGE_EFFICIENCY_J_PER_TH, date)


class ProductionCostConfig:
    """BTC production cost model. Energy cost × OVERHEAD_FACTOR gives all-in cost."""

    # Electricity is ~68% of total cost; OVERHEAD_FACTOR covers the rest (CAPEX, staff, etc.)
    OVERHEAD_FACTOR = 1.47

    BLOCKS_PER_DAY = 144
    BLOCK_REWARD = 3.125   # after 2024 halving

    PRE_HALVING_REWARD = 50.0
    HALVING_SCHEDULE = [
        ("2012-11-28", 25.0),
        ("2016-07-09", 12.5),
        ("2020-05-11", 6.25),
        ("2024-04-20", 3.125),
        ("2028-04-20", 1.5625),
    ]

    EMA_WINDOW = 14


class SignalConfig:
    """Signal generation parameters."""

    # Price/cost ratio → target BTC weight
    POSITION_TABLE = [
        (0.00, 0.80, 1.00),
        (0.80, 0.90, 0.85),
        (0.90, 1.00, 0.70),
        (1.00, 1.10, 0.50),
        (1.10, 1.25, 0.30),
        (1.25, 10.0, 0.00),
    ]

    # Used by ml_parameter_optimizer
    RATIO_BUY_THRESHOLD = 0.90
    RATIO_SELL_THRESHOLD = 1.10

    PRICE_EMA_WINDOW = 21
    COST_EMA_WINDOW = 30
    SIGNAL_EMA_WINDOW = 14


class PortfolioConfig:
    """Portfolio management parameters."""

    POSITION_TABLE = SignalConfig.POSITION_TABLE

    MAX_DAILY_WEIGHT_CHANGE = 0.10
    MIN_REBALANCE_THRESHOLD = 0.04

    # Volatility targeting: scale position so annualised port vol ≈ VOL_TARGET
    VOL_TARGET = 0.48
    VOL_SCALING_WINDOW = 15
    VOL_SCALE_MIN = 0.20
    VOL_SCALE_MAX = 1.00

    # Trend filter: full exit when price < long-term EMA
    TREND_FILTER_WINDOW = 250
    TREND_BEAR_CAP = 0.00

    # RSI overlay: boost on oversold, suppress disabled (hurts Sharpe)
    RSI_WINDOW = 14
    RSI_OVERSOLD = 30
    RSI_OVERBOUGHT = 100   # effectively disabled
    RSI_BOOST = 1.30
    RSI_SUPPRESS = 1.00

    # MVRV Z-score overlay (on-chain cycle signal)
    # Disabled: no Sharpe improvement vs baseline in 2022-2026 testing.
    MVRV_ENABLED = False
    MVRV_Z_WINDOW = 730
    MVRV_OVERSOLD_Z = -0.5
    MVRV_CAUTION_Z = 2.0
    MVRV_RISK_OFF_Z = 4.5
    MVRV_EXTREME_Z = 6.5
    MVRV_BOOST = 1.15
    MVRV_CAUTION_FACTOR = 0.70
    MVRV_RISK_OFF_FACTOR = 0.55
    MVRV_EXTREME_FACTOR = 0.25

    # Seasonal overlay (only statistically robust months; June removed: no edge)
    SEASONAL_ENABLED = True
    SEASONAL_MULTIPLIERS = {
        8: 0.70,   # August: 18% win rate over 11 years (9/11 negative)
    }

    # Halving cycle overlay: reduce exposure ~18 months post-halving (bear phase)
    # Grid-searched: 0.10 gives best OOS Sharpe (+0.118 vs baseline)
    HALVING_ENABLED = True
    HALVING_BEAR_MULT = 0.10

    # Hash Ribbon: miner capitulation (fast SMA < slow SMA → exit)
    # Walk-forward validated: OOS Sharpe +0.266
    HASH_RIBBON_ENABLED = True
    HASH_RIBBON_FAST = 30
    HASH_RIBBON_SLOW = 60
    HASH_RIBBON_CAP_MULT = 0.00

    TRADING_FEES_PERCENT = 0.001
    SLIPPAGE_PERCENT = 0.002

    INITIAL_PORTFOLIO_USD = 100_000


class RiskManagementConfig:
    """Risk monitoring and mode-switching parameters."""

    ROLLING_PEAK_WINDOW = 252
    VOLATILITY_WINDOW = 30
    VAR_LOOKBACK = 30
    VAR_CONFIDENCE = 0.99
    VAR_ZSCORE = 2.33

    MODE_CAPS = {
        "NORMAL": 1.00,
        "CAUTION": 0.75,
        "RISK_OFF": 0.45,
        "EMERGENCY": 0.05,
    }

    DD_THRESHOLDS = {
        "CAUTION": -0.12,
        "RISK_OFF": -0.20,
        "EMERGENCY": -0.35,
    }
    VOL_THRESHOLDS = {
        "CAUTION": 0.75,
        "RISK_OFF": 1.00,
        "EMERGENCY": 1.40,
    }
    VAR_THRESHOLDS = {
        "CAUTION": 0.04,
        "RISK_OFF": 0.06,
        "EMERGENCY": 0.09,
    }

    RECOVERY_THRESHOLDS = {
        "CAUTION":   {"dd": -0.09, "vol": 0.65, "var": 0.025},
        "RISK_OFF":  {"dd": -0.16, "vol": 0.85, "var": 0.04},
        "EMERGENCY": {"dd": -0.28, "vol": 1.20, "var": 0.06},
    }

    RECOVERY_DAYS = {
        "CAUTION": 2,
        "RISK_OFF": 3,
        "EMERGENCY": 5,
    }


class BacktestConfig:
    """Backtest run parameters."""

    USE_REAL_DATA = True
    STRICT_REAL_DATA = True
    DAYS_TO_FETCH = 3650

    START_DATE = "2016-02-01"
    END_DATE = "2026-02-01"
