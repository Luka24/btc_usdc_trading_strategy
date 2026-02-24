"""
Module for calculating Bitcoin production cost per 1 BTC
========================================================
Based on mining model: energy costs + operational costs.
Uses dynamic electricity prices and miner efficiency from HistoricalParameters.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from config import ProductionCostConfig as Config, HistoricalParameters


def get_block_reward_for_date(date_input) -> float:
    """
    Get the correct BTC block reward for a specific date.
    Accounts for bitcoin halvings.
    
    Args:
        date_input: Date in YYYY-MM-DD format (str, date, or Timestamp)
        
    Returns:
        float: BTC block reward for that date
    """
    halving_schedule = [
        (datetime.strptime(date_str, '%Y-%m-%d'), reward)
        for date_str, reward in Config.HALVING_SCHEDULE
    ]
    
    # Handle different input types
    if isinstance(date_input, str):
        date = datetime.strptime(date_input, '%Y-%m-%d')
    elif isinstance(date_input, datetime):
        date = date_input
    else:
        # Assume it's a Timestamp or other datetime-like
        date = pd.Timestamp(date_input).to_pydatetime()
    
    # Find the applicable reward for this date
    for halving_date, reward in halving_schedule:
        if date < halving_date:
            # Return reward from the previous halving
            idx = halving_schedule.index((halving_date, reward))
            if idx == 0:
                return Config.PRE_HALVING_REWARD  # Before first halving
            else:
                return halving_schedule[idx - 1][1]
    
    # Date is after all known halvings, return latest
    return halving_schedule[-1][1]


class BTCProductionCostCalculator:
    """
    Calculates the production cost of 1 BTC based on:
    - Network hashrate
    - Difficulty
    - Electricity price (dynamic from HistoricalParameters)
    - Miner efficiency (dynamic from HistoricalParameters)
    - Block reward (supports halvings)
    """
    
    def __init__(self, date=None, energy_price_per_kwh: float = None,
                 miner_efficiency_j_per_th: float = None,
                 block_reward: float = None):
        """
        Initialize calculator.
        
        Args:
            date: datetime.date or pandas.Timestamp for dynamic parameter lookup
            energy_price_per_kwh (float): Override electricity price ($/kWh)
            miner_efficiency_j_per_th (float): Override miner efficiency (J/TH)
            block_reward (float): BTC per block (None = use halving schedule for date)
        """
        self.date = date if date else pd.Timestamp.now()
        
        # Use provided values or get from HistoricalParameters
        if energy_price_per_kwh is not None:
            self.energy_price = energy_price_per_kwh
        else:
            self.energy_price = HistoricalParameters.get_electricity_price(self.date)
        
        if miner_efficiency_j_per_th is not None:
            self.efficiency = miner_efficiency_j_per_th
        else:
            self.efficiency = HistoricalParameters.get_miner_efficiency(self.date)
        
        # Use dynamic block reward based on date (accounts for halvings)
        if block_reward is not None:
            self.block_reward = block_reward
        else:
            self.block_reward = get_block_reward_for_date(self.date)
        
    def set_block_reward(self, block_reward: float) -> None:
        """Update block reward (for handling halvings)"""
        self.block_reward = block_reward
        
    def calculate_energy_cost_per_btc(self, 
                                      hashrate_eh_per_s: float,
                                      blocks_per_day: int = Config.BLOCKS_PER_DAY,
                                      block_reward: float = None) -> float:
        """
        Calculate energy cost per 1 BTC.
        
        Formula:
            Energy (kWh) = (Hashrate [H/s] * J/H) / (3.6M J/kWh)
            Cost = Energy * Price/kWh
        
        Args:
            hashrate_eh_per_s (float): Network hashrate in EH/s (exahashes/sec)
            blocks_per_day (int): Number of blocks per day
            block_reward (float): BTC per block (None = use instance value)
            
        Returns:
            float: Energy cost per 1 BTC (USD)
        """
        
        # Use provided or instance block reward
        reward = block_reward if block_reward is not None else self.block_reward
        
        # Convert EH/s to H/s
        hashrate_h_per_s = hashrate_eh_per_s * 1e18
        
        # BTC per day
        btc_per_day = blocks_per_day * reward
        
        # Total energy for all hashes in one day (J)
        # 1 day = 86400 seconds
        seconds_per_day = 86400
        total_hashes_per_day = hashrate_h_per_s * seconds_per_day
        
        # Energy (J) = hashes * J/hash
        # 1 J/TH = 1 J per 1 trillion hashes
        energy_joules = (total_hashes_per_day * self.efficiency) / 1e12
        
        # Convert J  kWh (1 kWh = 3.6M J)
        energy_kwh = energy_joules / 3.6e6
        
        # Energy cost for all BTC
        total_energy_cost = energy_kwh * self.energy_price
        
        # Cost per 1 BTC
        energy_cost_per_btc = total_energy_cost / btc_per_day
        
        return energy_cost_per_btc
    
    def calculate_total_cost_per_btc(self,
                                     hashrate_eh_per_s: float,
                                     overhead_factor: float = None) -> float:
        """
        Calculate total cost per 1 BTC (energy + overhead).
        
        Formula (based on research from MacroMicro, JPMorgan, CESifo):
            Energy cost per BTC = (Daily_Hashes * Efficiency_J_TH) / 1e12 / 3.6e6 * Price
            Total cost per BTC = Energy_cost * OVERHEAD_FACTOR
        
        OVERHEAD_FACTOR (default 1.40) covers:
            - CAPEX (hardware depreciation)
            - Facilities (electricity delivery, cooling)
            - Personnel (staff, management)
        
        If OVERHEAD_FACTOR = 1.40, then:
            - Energy is ~71% of costs
            - Other costs (CAPEX + facilities + staff) are ~29%
        
        Args:
            hashrate_eh_per_s (float): Hashrate in EH/s
            overhead_factor (float): Multiplier for all-in costs (None = use config)
            
        Returns:
            float: Total cost per 1 BTC (USD)
        """
        
        if overhead_factor is None:
            overhead_factor = Config.OVERHEAD_FACTOR
        
        # Energy cost per BTC
        energy_cost = self.calculate_energy_cost_per_btc(hashrate_eh_per_s)
        
        # Total cost = Energy * Overhead factor
        total_cost = energy_cost * overhead_factor
        
        return total_cost
    
    def calculate_breakeven_price(self,
                                  hashrate_eh_per_s: float) -> float:
        """
        Calculate breakeven price for BTC (minimum price to cover costs).
        
        Args:
            hashrate_eh_per_s (float): Hashrate in EH/s
            
        Returns:
            float: Breakeven price for BTC (USD)
        """
        return self.calculate_total_cost_per_btc(hashrate_eh_per_s)


class ProductionCostSeries:
    """
    Manages time series of production costs with dynamic parameters.
    - Electricity prices change by year (from HistoricalParameters)
    - Miner efficiency improves over time (from HistoricalParameters)
    - Enables daily updates and smoothing (EMA).
    """
    
    def __init__(self, ema_window: int = Config.EMA_WINDOW):
        """
        Args:
            ema_window (int): Window for exponential moving average
        """
        self.ema_window = ema_window
        self.data = pd.DataFrame()
        
    def add_daily_data(self, date: str, hashrate_eh_per_s: float) -> dict:
        """
        Add daily data and calculate costs with dynamic parameters.
        
        Args:
            date (str): Date (YYYY-MM-DD format)
            hashrate_eh_per_s (float): Hashrate in EH/s
            
        Returns:
            dict: Dictionary with calculated costs
        """
        
        # Get dynamic parameters for this date
        calculator = BTCProductionCostCalculator(date=date)
        
        # Calculate costs
        energy_cost = calculator.calculate_energy_cost_per_btc(hashrate_eh_per_s)
        total_cost = calculator.calculate_total_cost_per_btc(hashrate_eh_per_s)
        
        # Add to DataFrame
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
        """
        Add data from DataFrame (date, hashrate_eh_per_s).
        Each row uses dynamic parameters based on its date.
        
        Args:
            df (pd.DataFrame): DataFrame with 'date' and 'hashrate_eh_per_s' columns
        """
        results = []
        for _, row in df.iterrows():
            date = pd.to_datetime(row['date'])
            hashrate = row['hashrate_eh_per_s']
            
            # Get dynamic parameters for this specific date
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
        """
        Smooth costs with exponential moving average.
        
        Args:
            column (str): Column to smooth
            
        Returns:
            pd.Series: Smoothed costs
        """
        return self.data[column].ewm(span=self.ema_window, adjust=False).mean()
    
    def get_latest_cost(self, smoothed: bool = True) -> float:
        """
        Return the latest calculated cost.
        
        Args:
            smoothed (bool): Return smoothed value
            
        Returns:
            float: Latest cost (USD)
        """
        if self.data.empty:
            return 0.0
        
        if smoothed:
            return self.smooth_with_ema().iloc[-1]
        else:
            return self.data['total_cost_usd'].iloc[-1]
    
    def summary(self) -> dict:
        """
        Return summary of current data.
        
        Returns:
            dict: Summary (latest value, average, min, max)
        """
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
    dates = pd.date_range(start="2024-0-1-0-1", periods=60, freq="D")
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
