import pandas as pd
import numpy as np
from datetime import datetime
from enum import Enum
from production_cost import ProductionCostSeries
from portfolio import PortfolioManager
from risk_manager import RiskManager
from config import BacktestConfig as Config
from config import ProductionCostConfig as CostConfig
from config import SignalConfig


class Signal(Enum):
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"

def get_block_reward(date_input):
    """Return block reward for halving schedule."""
    halving_schedule = [
        (datetime.strptime(d, '%Y-%m-%d'), r)
        for d, r in CostConfig.HALVING_SCHEDULE
    ]
    
    if isinstance(date_input, str):
        date = datetime.strptime(date_input, '%Y-%m-%d')
    else:
        date = pd.Timestamp(date_input).to_pydatetime()
    
    for halving_date, reward in halving_schedule:
        if date < halving_date:
            idx = halving_schedule.index((halving_date, reward))
            if idx == 0:
                return CostConfig.PRE_HALVING_REWARD
            return halving_schedule[idx - 1][1]
    
    return halving_schedule[-1][1]


class BacktestEngine:
    """Backtest engine for BTC/USDC strategy."""
    
    def __init__(self, initial_capital=100_000, enable_risk_management=True):
        self.initial_capital = initial_capital
        self.cost_series = ProductionCostSeries()
        self.portfolio_manager = PortfolioManager(initial_capital=initial_capital)
        self.risk_manager = RiskManager()
        
        self.enable_risk_management = enable_risk_management
        self.buy_threshold = SignalConfig.RATIO_BUY_THRESHOLD
        self.sell_threshold = SignalConfig.RATIO_SELL_THRESHOLD
        
        # Risk limits
        self.volatility_threshold = 0.60
        self.max_drawdown_limit = -0.30
        self.position_reduction_on_high_vol = 0.50
        self.stop_loss_pct = -0.15
        self.take_profit_pct = 0.25
        self.trailing_stop_pct = -0.10
        self.max_consecutive_losses = 5
        self.consecutive_loss_reduction = 0.30
        
        self.btc_entry_price = None
        self.btc_peak_price = None
        self.consecutive_loss_count = 0
        
        self.backtest_data = pd.DataFrame()
        self.results = {}

    @classmethod
    def init(cls, **kwargs):
        return cls(**kwargs)
    
    def add_daily_data(self, date, btc_price, hashrate_eh_per_s):
        """Add daily data point."""
        block_reward = get_block_reward(date)
        cost_data = self.cost_series.add_daily_data(date, hashrate_eh_per_s)
        
        cost_smoothed = self.cost_series.smooth_with_ema().iloc[-1]
        ratio = self._calc_ratio(btc_price, cost_smoothed)
        signal = self._generate_signal(ratio)
        signal_strength = self._signal_strength(ratio)
        
        row = {
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
            'risk_action': 'PENDING',
            'position_scale': 1.0,
            'current_drawdown_pct': 0.0,
            'current_volatility_pct': 0.0,
            'consecutive_losses': 0,
            'stop_loss_active': False,
            'take_profit_active': False,
        }
        
        new_row = pd.DataFrame([row], index=[len(self.backtest_data)])
        self.backtest_data = pd.concat([self.backtest_data, new_row], ignore_index=True)
    
    def _calc_ratio(self, price, cost):
        if cost == 0:
            return 0.0
        return price / cost
    
    def _generate_signal(self, ratio):
        if ratio < self.buy_threshold:
            return Signal.BUY
        elif ratio > self.sell_threshold:
            return Signal.SELL
        return Signal.HOLD
    
    def _signal_strength(self, ratio):
        if ratio < self.buy_threshold:
            strength = 1.0 - (ratio / self.buy_threshold)
            return max(0.5, strength)
        elif ratio > self.sell_threshold:
            excess = ratio - self.sell_threshold
            strength = min(excess / (self.sell_threshold * 0.5), 0.5)
            return 0.5 - strength
        return 0.5
    
    def add_from_dataframe(self, df):
        """Add data from DataFrame."""
        for _, row in df.iterrows():
            self.add_daily_data(
                date=row['date'],
                btc_price=row['btc_price'],
                hashrate_eh_per_s=row['hashrate_eh_per_s']
            )
    
    def get_portfolio_value(self, btc_price):
        portfolio = self.portfolio_manager.calculate_portfolio_value(
            btc_price,
            self.portfolio_manager.btc_quantity
        )
        return portfolio['total_value']
    
    def run_backtest(self, initial_btc_quantity=2.0):
        """Run backtest with risk management."""
        if self.backtest_data.empty:
            return self.portfolio_manager.get_portfolio_dataframe()

        first_price = self.backtest_data.iloc[0]['btc_price']
        self.portfolio_manager.initialize_holdings(first_price, initial_btc_quantity)
        
        daily_returns = []
        
        for idx, row in self.backtest_data.iterrows():
            date = row['date']
            price = row['btc_price']
            ratio = row['signal_ratio']
            signal = row['signal']
            date_str = date.strftime('%Y-%m-%d')
            
            current_value = self.get_portfolio_value(price)
            
            risk_action = "NORMAL"
            position_scale = 1.0
            
            if self.enable_risk_management:
                # Check drawdown
                self.risk_manager.update_peak(current_value, date_str)
                drawdown_hit, dd = self.risk_manager.check_drawdown_limit(
                    current_value, limit=self.max_drawdown_limit
                )
                
                if drawdown_hit:
                    signal = Signal.SELL
                    risk_action = f"DRAWDOWN_BRAKE (DD={dd:.1%})"
                
                # Stop loss / take profit
                btc_wt = self.portfolio_manager.btc_quantity * price / current_value if current_value > 0 else 0
                
                if btc_wt > 0.40 and self.btc_entry_price is None:
                    self.btc_entry_price = price
                    self.btc_peak_price = price
                
                if btc_wt > 0.40 and self.btc_peak_price is not None:
                    self.btc_peak_price = max(self.btc_peak_price, price)
                
                if self.btc_entry_price is not None and btc_wt > 0.20:
                    pct_change = (price - self.btc_entry_price) / self.btc_entry_price
                    
                    if pct_change < self.stop_loss_pct:
                        signal = Signal.SELL
                        risk_action = f"STOP_LOSS ({pct_change:.1%})"
                        self.btc_entry_price = None
                        self.btc_peak_price = None
                    elif pct_change > self.take_profit_pct:
                        signal = Signal.SELL
                        risk_action = f"TAKE_PROFIT ({pct_change:.1%})"
                        self.btc_entry_price = None
                        self.btc_peak_price = None
                    elif self.btc_peak_price is not None:
                        peak_dd = (price - self.btc_peak_price) / self.btc_peak_price
                        if peak_dd < self.trailing_stop_pct:
                            signal = Signal.SELL
                            risk_action = f"TRAILING_STOP ({peak_dd:.1%})"
                            self.btc_entry_price = None
                            self.btc_peak_price = None
                
                if btc_wt < 0.20:
                    self.btc_entry_price = None
                    self.btc_peak_price = None
                
                # Consecutive losses
                if daily_returns and daily_returns[-1] < 0:
                    self.consecutive_loss_count += 1
                elif daily_returns and daily_returns[-1] > 0:
                    self.consecutive_loss_count = 0
                
                loss_trigger = False
                if self.consecutive_loss_count >= self.max_consecutive_losses:
                    loss_trigger = True
                    if risk_action == "NORMAL":
                        risk_action = f"CONSECUTIVE_LOSSES ({self.consecutive_loss_count}d)"
                
                # Volatility-based sizing
                if len(daily_returns) >= 10:
                    vol = np.std(daily_returns[-30:]) * np.sqrt(252)
                else:
                    vol = 0.0
                
                daily_returns.append(0.0)
                
                if vol > self.volatility_threshold:
                    position_scale = self.position_reduction_on_high_vol
                    if risk_action == "NORMAL":
                        risk_action = f"HIGH_VOL ({position_scale:.0%})"
                
                if loss_trigger:
                    position_scale = min(position_scale, self.consecutive_loss_reduction)
            else:
                daily_returns.append(0.0)
                vol = 0.0
                dd = 0.0
            
            # Rebalance
            rebalance = self.portfolio_manager.rebalance(ratio, enforce_limit=True)
            
            if self.enable_risk_management:
                adj_weight = rebalance['actual_weight'] * position_scale
                adj_weight = np.clip(adj_weight, 0.0, 1.0)
            else:
                adj_weight = rebalance['actual_weight']
            
            self.portfolio_manager.execute_rebalance(
                price, 
                self.portfolio_manager.btc_quantity, 
                adj_weight
            )
            
            new_value = self.get_portfolio_value(price)
            if len(daily_returns) > 0:
                ret = (new_value - current_value) / current_value if current_value > 0 else 0
                daily_returns[-1] = ret
                self.risk_manager.update_returns(ret)
            
            self.portfolio_manager.add_to_history(
                date_str, price,
                self.portfolio_manager.btc_quantity,
                ratio, signal
            )
            
            self.backtest_data.at[idx, 'risk_action'] = risk_action
            self.backtest_data.at[idx, 'position_scale'] = position_scale
            self.backtest_data.at[idx, 'current_drawdown_pct'] = dd * 100 if self.enable_risk_management else 0
            self.backtest_data.at[idx, 'current_volatility_pct'] = vol * 100
            self.backtest_data.at[idx, 'consecutive_losses'] = self.consecutive_loss_count
            self.backtest_data.at[idx, 'stop_loss_active'] = 'STOP_LOSS' in risk_action
            self.backtest_data.at[idx, 'take_profit_active'] = 'TAKE_PROFIT' in risk_action or 'TRAILING' in risk_action
        
        return self.portfolio_manager.get_portfolio_dataframe()
    
    def calculate_metrics(self):
        """Calculate backtest performance metrics."""
        portfolio_df = self.portfolio_manager.get_portfolio_dataframe()
        metrics = self.portfolio_manager.calculate_metrics(portfolio_df)
        
        buy = (self.backtest_data['signal'] == 'BUY').sum()
        sell = (self.backtest_data['signal'] == 'SELL').sum()
        hold = (self.backtest_data['signal'] == 'HOLD').sum()
        
        pdf = portfolio_df.copy()
        pdf['daily_return'] = pdf['total_value'].pct_change()
        wins = (pdf['daily_return'] > 0).sum()
        total_days = len(pdf) - 1
        win_rate = (wins / total_days * 100) if total_days > 0 else 0
        
        return {
            **metrics,
            'buy_signals': buy,
            'sell_signals': sell,
            'hold_signals': hold,
            'win_rate_pct': win_rate,
            'data_points': len(self.backtest_data),
        }
    
    def generate_report(self):
        """Generate backtest report."""
        metrics = self.calculate_metrics()
        pdf = self.portfolio_manager.get_portfolio_dataframe()
        
        return f"""
{'='*70}
BACKTEST REPORT: BTC/USDC Trading Strategy
{'='*70}

PERIOD
Start:              {pdf.index[0].strftime('%Y-%m-%d')}
End:                {pdf.index[-1].strftime('%Y-%m-%d')}
Days:                 {len(pdf)}

CAPITAL
Initial:             ${self.initial_capital:>15,.2f}
Final:               ${metrics['final_value']:>15,.2f}
Return:              {metrics['total_return_pct']:>15.2f}%

PERFORMANCE
Avg daily return:     {metrics['avg_daily_return_pct']:>12.2f}%
Volatility:           {metrics['daily_volatility_pct']:>12.2f}%
Sharpe:               {metrics['sharpe_ratio']:>12.2f}

RISK
Max drawdown:         {metrics['max_drawdown_pct']:>12.2f}%

SIGNALS
Buy:                  {metrics['buy_signals']:>15}
Sell:                 {metrics['sell_signals']:>15}
Hold:                 {metrics['hold_signals']:>15}
Total trades:         {metrics['num_trades']:>15}
Win rate:             {metrics['win_rate_pct']:>15.2f}%

FINAL ALLOCATION
BTC:                  {metrics['final_btc_weight']:>15.1%}
USDC:                 {1 - metrics['final_btc_weight']:>15.1%}

{'='*70}
"""
    
    def export_results(self, filename=None):
        """Export results to CSV."""
        import os
        os.makedirs('results', exist_ok=True)
        
        if not filename:
            filename = f"results/backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        self.backtest_data.to_csv(filename, index=False)
        print(f"Results exported to: {filename}")


def create_synthetic_data(days=365):
    """Create synthetic data for testing."""
    dates = pd.date_range(start="2023-01-01", periods=days, freq="D")
    
    base_price = 42000
    
    cycle = np.sin(np.linspace(0, 4*np.pi, days)) * 5000
    trend = np.linspace(0, 8000, days)
    vol = np.random.normal(0, 2500, days)
    
    prices = base_price + cycle + trend + vol
    prices = np.maximum(prices, 10000)
    
    base_hr = 600
    hr_cycle = np.sin(np.linspace(0, 4*np.pi, days)) * 40
    hr_trend = np.linspace(0, 60, days)
    hr_noise = np.random.normal(0, 10, days)
    
    hrs = base_hr + hr_cycle + hr_trend + hr_noise
    hrs = np.maximum(hrs, 100)
    
    return pd.DataFrame({
        'date': [d.strftime('%Y-%m-%d') for d in dates],
        'btc_price': prices,
        'hashrate_eh_per_s': hrs,
    })


if __name__ == "__main__":
    print("=" * 70)
    print("BACKTEST BTC/USDC STRATEGY")
    print("=" * 70)
    
    print("\nGenerating synthetic data...")
    df = create_synthetic_data(days=365)
    
    print("Running backtest...")
    engine = BacktestEngine(initial_capital=100_000)
    engine.add_from_dataframe(df)
    
    pdf = engine.run_backtest(initial_btc_quantity=2.0)
    
    print("\n" + engine.generate_report())
    
    print("\nLast 10 days:")
    print("-" * 70)
    print(pdf.tail(10).to_string())
