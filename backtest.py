"""
Module for BTC/USDC Trading Strategy Backtesting
=================================================
Simulates trading strategy on historical data.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from enum import Enum
from production_cost import ProductionCostSeries
from portfolio import PortfolioManager
from config import BacktestConfig as Config
from config import ProductionCostConfig as CostConfig
from config import SignalConfig


# Signal enumeration
class Signal(Enum):
    """Trading signals"""
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"


def get_block_reward_for_date(date_input) -> float:
    """
    Get the correct BTC block reward for a specific date.
    Accounts for bitcoin halvings.
    
    Args:
        date_input: Date in YYYY-MM-DD format (str or Timestamp)
        
    Returns:
        float: BTC block reward for that date
    """
    halving_schedule = [
        (datetime.strptime(date_str, '%Y-%m-%d'), reward)
        for date_str, reward in CostConfig.HALVING_SCHEDULE
    ]
    
    # Handle both string and Timestamp inputs
    if isinstance(date_input, str):
        date = datetime.strptime(date_input, '%Y-%m-%d')
    else:
        # Assume it's a Timestamp or datetime
        date = pd.Timestamp(date_input).to_pydatetime()
    
    # Find the applicable reward for this date
    for halving_date, reward in halving_schedule:
        if date < halving_date:
            # Return reward from the previous halving
            idx = halving_schedule.index((halving_date, reward))
            if idx == 0:
                return CostConfig.PRE_HALVING_REWARD  # Before first halving
            else:
                return halving_schedule[idx - 1][1]
    
    # Date is after all known halvings, return latest
    return halving_schedule[-1][1]


class BacktestEngine:
    """
    Backtest engine for BTC/USDC strategy.
    Combines: cost, signal, portfolio management.
    """
    
    def __init__(self, initial_capital: float = 100_000):
        """
        Initialize backtest engine.
        
        Args:
            initial_capital (float): Starting capital
        """
        self.initial_capital = initial_capital
        self.cost_series = ProductionCostSeries()
        self.portfolio_manager = PortfolioManager(initial_capital=initial_capital)
        
        # Signal thresholds
        self.buy_threshold = SignalConfig.RATIO_BUY_THRESHOLD
        self.sell_threshold = SignalConfig.RATIO_SELL_THRESHOLD
        
        self.backtest_data = pd.DataFrame()
        self.results = {}
    
    def add_daily_data(self, date: str, btc_price: float, hashrate_eh_per_s: float) -> None:
        """
        Add daily data point with dynamic halving and dynamic parameters support.
        Uses historical electricity prices and miner efficiency based on date.
        
        Args:
            date (str): Date (YYYY-MM-DD)
            btc_price (float): BTC market price
            hashrate_eh_per_s (float): Network hashrate
        """
        
        # Get correct block reward for this date (accounts for halvings)
        block_reward = get_block_reward_for_date(date)
        
        # Add daily data (automatically uses dynamic parameters for this date)
        cost_data = self.cost_series.add_daily_data(date, hashrate_eh_per_s)
        
        # Calculate signal
        cost_smoothed = self.cost_series.smooth_with_ema().iloc[-1]
        ratio = self._calculate_ratio(btc_price, cost_smoothed)
        signal = self._generate_signal(ratio)
        signal_strength = self._get_signal_strength(ratio)
        
        # Save combined
        row_dict = {
            'date': pd.to_datetime(date),
            'btc_price': btc_price,
            'hashrate': hashrate_eh_per_s,
            'production_cost': cost_data['total_cost'],
            'production_cost_smoothed': cost_smoothed,
            'signal_ratio': ratio,
            'signal': signal.value,
            'signal_strength': signal_strength,
            'block_reward': block_reward,
            'electricity_price': cost_data['electricity_price'],
            'miner_efficiency': cost_data['miner_efficiency'],
        }
        
        new_row = pd.DataFrame([row_dict], index=[len(self.backtest_data)])
        self.backtest_data = pd.concat([self.backtest_data, new_row], ignore_index=True)
    
    def _calculate_ratio(self, btc_price: float, production_cost: float) -> float:
        """Calculate ratio: Price / Cost"""
        if production_cost == 0:
            return 0.0
        return btc_price / production_cost
    
    def _generate_signal(self, ratio: float) -> Signal:
        """Generate signal based on ratio."""
        if ratio < self.buy_threshold:
            return Signal.BUY
        elif ratio > self.sell_threshold:
            return Signal.SELL
        else:
            return Signal.HOLD
    
    def _get_signal_strength(self, ratio: float) -> float:
        """Return signal strength (0-1)."""
        if ratio < self.buy_threshold:
            strength = 1.0 - (ratio / self.buy_threshold)
            return max(0.5, strength)
        elif ratio > self.sell_threshold:
            excess = ratio - self.sell_threshold
            strength = min(excess / (self.sell_threshold * 0.5), 0.5)
            return 0.5 - strength
        else:
            return 0.5
    
    def add_from_dataframe(self, df: pd.DataFrame) -> None:
        """
        Add data from DataFrame.
        Expects columns: date, btc_price, hashrate_eh_per_s
        
        Args:
            df (pd.DataFrame): Data
        """
        
        for _, row in df.iterrows():
            self.add_daily_data(
                date=row['date'],
                btc_price=row['btc_price'],
                hashrate_eh_per_s=row['hashrate_eh_per_s']
            )
    
    def run_backtest(self, initial_btc_quantity: float = 2.0) -> pd.DataFrame:
        """
        Run backtest with rebalancing simulation.
        
        Args:
            initial_btc_quantity (float): Initial BTC quantity
            
        Returns:
            pd.DataFrame: Backtest results
        """
        
        if self.backtest_data.empty:
            return self.portfolio_manager.get_portfolio_dataframe()

        first_price = self.backtest_data.iloc[0]['btc_price']
        self.portfolio_manager.initialize_holdings(first_price, initial_btc_quantity)
        
        for _, row in self.backtest_data.iterrows():
            date = row['date']
            price = row['btc_price']
            ratio = row['signal_ratio']
            signal = row['signal']
            
            # Rebalance
            rebalance = self.portfolio_manager.rebalance(ratio, enforce_limit=True)
            self.portfolio_manager.execute_rebalance(price, self.portfolio_manager.btc_quantity, rebalance['actual_weight'])
            
            # Add to history
            self.portfolio_manager.add_to_history(
                date.strftime('%Y-%m-%d'),
                price,
                self.portfolio_manager.btc_quantity,
                ratio,
                signal
            )
        
        return self.portfolio_manager.get_portfolio_dataframe()
    
    def calculate_backtest_metrics(self) -> dict:
        """
        Calculate backtest metrics.
        
        Returns:
            dict: Performance metrics
        """
        
        portfolio_df = self.portfolio_manager.get_portfolio_dataframe()
        metrics = self.portfolio_manager.calculate_metrics(portfolio_df)
        
        # Additional metrics - count signals from backtest data
        buy_count = (self.backtest_data['signal'] == 'BUY').sum()
        sell_count = (self.backtest_data['signal'] == 'SELL').sum()
        hold_count = (self.backtest_data['signal'] == 'HOLD').sum()
        
        # Win rate (days when portfolio grew)
        portfolio_df_copy = portfolio_df.copy()
        portfolio_df_copy['daily_return'] = portfolio_df_copy['total_value'].pct_change()
        win_days = (portfolio_df_copy['daily_return'] > 0).sum()
        total_trading_days = len(portfolio_df_copy) - 1
        win_rate = (win_days / total_trading_days * 100) if total_trading_days > 0 else 0
        
        return {
            **metrics,
            'buy_signals': buy_count,
            'sell_signals': sell_count,
            'hold_signals': hold_count,
            'win_rate_pct': win_rate,
            'data_points': len(self.backtest_data),
        }
    
    def generate_report(self) -> str:
        """
        Generate backtest report.
        
        Returns:
            str: Text report
        """
        
        metrics = self.calculate_backtest_metrics()
        portfolio_df = self.portfolio_manager.get_portfolio_dataframe()
        
        report = f"""
{'='*70}
BACKTEST REPORT: BTC/USDC Trading Strategy
{'='*70}

TIME PERIOD
{'-'*70}
Start:              {portfolio_df.index[0].strftime('%Y-%m-%d')}
End:                {portfolio_df.index[-1].strftime('%Y-%m-%d')}
Duration:             {len(portfolio_df)} days

CAPITAL
{'-'*70}
Initial capital:      ${self.initial_capital:>15,.2f}
Final capital:       ${metrics['final_value']:>15,.2f}
Total return:         {metrics['total_return_pct']:>15.2f}%

RETURNS AND VOLATILITY
{'-'*70}
Average daily return: {metrics['avg_daily_return_pct']:>12.2f}%
Daily volatility:     {metrics['daily_volatility_pct']:>12.2f}%
Sharpe ratio:           {metrics['sharpe_ratio']:>12.2f}

RISK
{'-'*70}
Max drawdown:           {metrics['max_drawdown_pct']:>12.2f}%

SIGNALS AND TRADES
{'-'*70}
BUY signals:            {metrics['buy_signals']:>15}
SELL signals:           {metrics['sell_signals']:>15}
HOLD signals:           {metrics['hold_signals']:>15}
Total trades:        {metrics['num_trades']:>15}
Win rate:               {metrics['win_rate_pct']:>15.2f}%

FINAL ALLOCATION
{'-'*70}
BTC weight:               {metrics['final_btc_weight']:>15.1%}
USDC weight:              {1 - metrics['final_btc_weight']:>15.1%}

{'='*70}
"""
        
        return report
    
    def export_results(self, filename: str = None) -> None:
        """
        Export results to CSV.
        
        Args:
            filename (str): Filename (default: results/backtest_results.csv)
        """
        
        import os
        os.makedirs('results', exist_ok=True)
        
        if filename is None:
            filename = f"results/backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        # Export backtest_data instead of portfolio to include production_cost
        self.backtest_data.to_csv(filename, index=False)
        print(f"Results exported to: {filename}")


# ============ SYNTHETIC DATA GENERATION ============
def create_synthetic_data(days: int = 365) -> pd.DataFrame:
    """
    Create synthetic data for testing.
    
    Args:
        days (int): Number of days
        
    Returns:
        pd.DataFrame: Synthetic data
    """
    
    dates = pd.date_range(start="2023-01-01", periods=days, freq="D")
    
    # BTC price with realistic cycles (bull/bear/sideways)
    base_price = 42000
    
    # Create price with multiple cycles
    cycle1 = np.sin(np.linspace(0, 4*np.pi, days)) * 5000  # Oscillation
    trend = np.linspace(0, 8000, days)  # Uptrend
    volatility = np.random.normal(0, 2500, days)
    
    btc_prices = base_price + cycle1 + trend + volatility
    btc_prices = np.maximum(btc_prices, 10000)  # Min price
    
    # Hashrate matches price movement (production cost follows price)
    base_hashrate = 600
    hashrate_cycle = np.sin(np.linspace(0, 4*np.pi, days)) * 40
    hashrate_trend = np.linspace(0, 60, days)
    hashrate_noise = np.random.normal(0, 10, days)
    
    hashrates = base_hashrate + hashrate_cycle + hashrate_trend + hashrate_noise
    hashrates = np.maximum(hashrates, 100)
    
    df = pd.DataFrame({
        'date': [d.strftime('%Y-%m-%d') for d in dates],
        'btc_price': btc_prices,
        'hashrate_eh_per_s': hashrates,
    })
    
    return df


# ============ USAGE EXAMPLE ============
if __name__ == "__main__":
    print("=" * 70)
    print("BACKTEST BTC/USDC STRATEGY")
    print("=" * 70)
    
    # Create synthetic data
    print("\nCreating synthetic data...")
    df = create_synthetic_data(days=365)
    
    # Run backtest
    print("Running backtest...")
    engine = BacktestEngine(initial_capital=100_000)
    engine.add_from_dataframe(df)
    
    portfolio_df = engine.run_backtest(initial_btc_quantity=2.0)
    
    # Print report
    print("\n" + engine.generate_report())
    
    # Show last rows
    print("\nLast 10 days:")
    print("-" * 70)
    print(portfolio_df.tail(10).to_string())
