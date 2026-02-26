"""
Bitcoin production cost calculator.
Uses hashrate, miner efficiency, and electricity prices to estimate the all-in cost per BTC.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from config import ProductionCostConfig as Config, HistoricalParameters


# Parse once at import time — called for every row in a 3650-day backtest
_HALVING_SCHEDULE = [
    (datetime.strptime(date_str, '%Y-%m-%d'), reward)
    for date_str, reward in Config.HALVING_SCHEDULE
]


def get_block_reward_for_date(date_input) -> float:
    """Return the block reward for a given date, accounting for halvings."""
    if isinstance(date_input, str):
        date = datetime.strptime(date_input, '%Y-%m-%d')
    elif isinstance(date_input, datetime):
        date = date_input
    else:
        date = pd.Timestamp(date_input).to_pydatetime()

    reward = Config.PRE_HALVING_REWARD
    for halving_date, halving_reward in _HALVING_SCHEDULE:
        if date >= halving_date:
            reward = halving_reward
        else:
            break
    return reward


class BTCProductionCostCalculator:
    """Calculates the cost to mine 1 BTC given hashrate, electricity price, and miner efficiency."""
    
    def __init__(self, date=None, energy_price_per_kwh: float = None,
                 miner_efficiency_j_per_th: float = None,
                 block_reward: float = None):
        """Pass date to use historical parameters, or override them manually."""
        self.date = date if date else pd.Timestamp.now()
        self.energy_price = energy_price_per_kwh if energy_price_per_kwh is not None else HistoricalParameters.get_electricity_price(self.date)
        self.efficiency = miner_efficiency_j_per_th if miner_efficiency_j_per_th is not None else HistoricalParameters.get_miner_efficiency(self.date)
        self.block_reward = block_reward if block_reward is not None else get_block_reward_for_date(self.date)
        
    def calculate_energy_cost_per_btc(self, 
                                      hashrate_eh_per_s: float,
                                      blocks_per_day: int = Config.BLOCKS_PER_DAY,
                                      block_reward: float = None) -> float:
        """Energy cost (USD) per 1 BTC at the given hashrate."""
        reward = block_reward if block_reward is not None else self.block_reward
        btc_per_day = blocks_per_day * reward
        # EH/s → H/s, multiply by 86400s, divide by 1e12 (J/TH), divide by 3.6e6 (J→kWh)
        energy_kwh = (hashrate_eh_per_s * 1e18 * 86400 * self.efficiency) / 1e12 / 3.6e6
        return (energy_kwh * self.energy_price) / btc_per_day
    
    def calculate_total_cost_per_btc(self,
                                     hashrate_eh_per_s: float,
                                     overhead_factor: float = None) -> float:
        """All-in cost per BTC: energy × overhead factor (CAPEX, facilities, staff)."""
        if overhead_factor is None:
            overhead_factor = Config.OVERHEAD_FACTOR
        return self.calculate_energy_cost_per_btc(hashrate_eh_per_s) * overhead_factor


class ProductionCostSeries:
    """Tracks per-day production costs and exposes an EMA-smoothed series."""
    
    def __init__(self, ema_window: int = Config.EMA_WINDOW):
        self.ema_window = ema_window
        self.data = pd.DataFrame()
        
    def add_daily_data(self, date: str, hashrate_eh_per_s: float) -> dict:
        """Calculate and store costs for one day. Returns the cost dict."""
        calculator = BTCProductionCostCalculator(date=date)
        energy_cost = calculator.calculate_energy_cost_per_btc(hashrate_eh_per_s)
        total_cost = calculator.calculate_total_cost_per_btc(hashrate_eh_per_s)
        new_row = pd.DataFrame({
            'date': [pd.to_datetime(date)],
            'hashrate_eh_per_s': [hashrate_eh_per_s],
            'energy_cost_usd': [energy_cost],
            'total_cost_usd': [total_cost],
            'electricity_price': [calculator.energy_price],
            'miner_efficiency': [calculator.efficiency],
        })
        
        self.data = pd.concat([self.data, new_row], ignore_index=True)
        
        return {
            'date': date,
            'energy_cost': energy_cost,
            'total_cost': total_cost,
            'hashrate': hashrate_eh_per_s,
            'electricity_price': calculator.energy_price,
            'miner_efficiency': calculator.efficiency,
        }
    
    def add_from_dataframe(self, df: pd.DataFrame) -> None:
        """Batch-load a DataFrame with 'date' and 'hashrate_eh_per_s' columns."""
        results = []
        for _, row in df.iterrows():
            date = pd.to_datetime(row['date'])
            hashrate = row['hashrate_eh_per_s']
            
            calculator = BTCProductionCostCalculator(date=date)
            
            energy_cost = calculator.calculate_energy_cost_per_btc(hashrate)
            total_cost = calculator.calculate_total_cost_per_btc(hashrate)
            
            results.append({
                'date': date,
                'hashrate_eh_per_s': hashrate,
                'energy_cost_usd': energy_cost,
                'total_cost_usd': total_cost,
                'electricity_price': calculator.energy_price,
                'miner_efficiency': calculator.efficiency,
            })
        
        self.data = pd.DataFrame(results)
        self.data.set_index('date', inplace=True)
        self.data.sort_index(inplace=True)
    
    def smooth_with_ema(self, column: str = 'total_cost_usd') -> pd.Series:
        """Return an EMA-smoothed version of the given column."""
        return self.data[column].ewm(span=self.ema_window, adjust=False).mean()
    
    def get_latest_cost(self, smoothed: bool = True) -> float:
        """Return the most recent cost (smoothed by default)."""
        if self.data.empty:
            return 0.0
        if smoothed:
            return self.smooth_with_ema().iloc[-1]
        return self.data['total_cost_usd'].iloc[-1]
    
    def summary(self) -> dict:
        """Return a quick-look dict with latest, average, min, max costs."""
        if self.data.empty:
            return {}
        
        return {
            'latest_cost': self.data['total_cost_usd'].iloc[-1],
            'latest_cost_smoothed': self.smooth_with_ema().iloc[-1],
            'average_cost': self.data['total_cost_usd'].mean(),
            'min_cost': self.data['total_cost_usd'].min(),
            'max_cost': self.data['total_cost_usd'].max(),
            'latest_hashrate': self.data['hashrate_eh_per_s'].iloc[-1],
        }


# ============ EXAMPLE USAGE ============
if __name__ == "__main__":
    # Example 1: Single calculation
    print("=" * 60)
    print("EXAMPLE 1: Single calculation of production cost")
    print("=" * 60)
    
    calc = BTCProductionCostCalculator(
        energy_price_per_kwh=0.05,
        miner_efficiency_j_per_th=25
    )
    
    # Current BTC network hashrate (approximate)
    current_hashrate = 650  # EH/s (example)
    
    energy_cost = calc.calculate_energy_cost_per_btc(current_hashrate)
    total_cost = calc.calculate_total_cost_per_btc(current_hashrate)
    
    print(f"Hashrate: {current_hashrate} EH/s")
    print(f"Energy cost: ${energy_cost:.2f}")
    print(f"Total cost (with OPEX + depreciation): ${total_cost:.2f}")
    print(f"Breakeven price: ${total_cost:.2f}")
    
    # Example 2: Time series
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Time series of costs (daily)")
    print("=" * 60)
    
    # Simulated data (rising hashrate trend)
    dates = pd.date_range(start="2024-01-01", periods=60, freq="D")
    hashrates = np.linspace(600, 700, 60) + np.random.normal(0, 20, 60)
    
    df = pd.DataFrame({
        'date': dates,
        'hashrate_eh_per_s': hashrates,
    })
    
    cost_series = ProductionCostSeries(ema_window=14)
    cost_series.add_from_dataframe(df)
    
    # Show first and last data
    print(cost_series.data.head())
    print("\n... [middle rows omitted] ...\n")
    print(cost_series.data.tail())
    
    # Summary
    print("\n" + "=" * 60)
    print("COST SUMMARY")
    print("=" * 60)
    summary = cost_series.summary()
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key}: ${value:.2f}" if 'cost' in key or 'hashrate' not in key else f"{key}: {value:.2f}")
