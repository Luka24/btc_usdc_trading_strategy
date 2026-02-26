"""Portfolio manager for BTC/USDC positions."""

import pandas as pd
import numpy as np
from config import PortfolioConfig as Config


class PortfolioManager:
    """Manages BTC/USDC allocation and rebalancing."""

    def __init__(self,
                 initial_capital: float = Config.INITIAL_PORTFOLIO_USD,
                 position_table: list = Config.POSITION_TABLE,
                 max_daily_change: float = Config.MAX_DAILY_WEIGHT_CHANGE):
        self.initial_capital = initial_capital
        self.position_table = position_table
        self.max_daily_change = max_daily_change
        
        self.portfolio_history = []
        self.current_btc_weight = 0.50  # Start with 50% BTC
        self.previous_btc_weight = 0.50
        self.btc_quantity = 0.0
        self.usdc_quantity = self.initial_capital

    def initialize_holdings(self, btc_price: float, initial_btc_quantity: float) -> None:
        """Set initial BTC and USDC holdings."""
        btc_value = initial_btc_quantity * btc_price
        if btc_value > self.initial_capital:
            initial_btc_quantity = self.initial_capital / btc_price
            btc_value = initial_btc_quantity * btc_price

        self.btc_quantity = initial_btc_quantity
        self.usdc_quantity = self.initial_capital - btc_value
    
    def determine_target_weight(self, signal_ratio: float) -> float:
        """Return target BTC weight for the given price/cost ratio."""
        for min_ratio, max_ratio, weight in self.position_table:
            if min_ratio <= signal_ratio < max_ratio:
                return weight
        
        # Fallback (if ratio outside table)
        if signal_ratio > self.position_table[-1][1]:
            return self.position_table[-1][2]
        else:
            return self.position_table[0][2]

    def apply_weight_change_limit(self, target_weight: float) -> float:
        """Clamp the weight change to MAX_DAILY_WEIGHT_CHANGE."""
        weight_change = abs(target_weight - self.current_btc_weight)
        if weight_change > self.max_daily_change:
            # Limit to maximum change
            if target_weight > self.current_btc_weight:
                allowed_weight = self.current_btc_weight + self.max_daily_change
            else:
                allowed_weight = self.current_btc_weight - self.max_daily_change
            
            return max(0.0, min(1.0, allowed_weight))
        
        return target_weight
    
    def rebalance(self, signal_ratio: float, enforce_limit: bool = True) -> dict:
        """Compute new BTC weight given signal ratio and daily-change constraint."""
        target_weight = self.determine_target_weight(signal_ratio)
        if enforce_limit:
            actual_weight = self.apply_weight_change_limit(target_weight)
        else:
            actual_weight = target_weight
        self.previous_btc_weight = self.current_btc_weight
        self.current_btc_weight = actual_weight
        return {
            'target_weight': target_weight,
            'actual_weight': actual_weight,
            'weight_limited': target_weight != actual_weight,
            'btc_weight': actual_weight,
            'usdc_weight': 1.0 - actual_weight,
        }
    
    def calculate_portfolio_value(self, btc_price: float, btc_quantity: float) -> dict:
        """Return current portfolio value breakdown."""
        btc_value = btc_quantity * btc_price
        usdc_value = self.usdc_quantity
        total_value = btc_value + usdc_value
        
        return {
            'btc_quantity': btc_quantity,
            'btc_price': btc_price,
            'btc_value': btc_value,
            'usdc_value': usdc_value,
            'total_value': total_value,
            'btc_weight': btc_value / total_value if total_value > 0 else 0,
            'usdc_weight': usdc_value / total_value if total_value > 0 else 0,
        }
    
    def execute_rebalance(self, btc_price: float, btc_quantity: float,
                          target_weight: float, fees: float = Config.TRADING_FEES_PERCENT) -> dict:
        """Execute a rebalance trade and update internal holdings."""
        current_portfolio = self.calculate_portfolio_value(btc_price, self.btc_quantity)
        current_btc_value = current_portfolio['btc_value']
        current_usdc_value = current_portfolio['usdc_value']
        total_value = current_btc_value + current_usdc_value
        
        # Target value BTC
        target_btc_value = total_value * target_weight
        btc_value_delta = target_btc_value - current_btc_value
        if btc_value_delta > 0:
            # Buy BTC (limit by available USDC)
            max_affordable = self.usdc_quantity / (1 + fees) if self.usdc_quantity > 0 else 0
            btc_value_delta = min(btc_value_delta, max_affordable)
            btc_quantity_delta = btc_value_delta / btc_price if btc_price > 0 else 0
            trade_cost = btc_value_delta * fees
            self.usdc_quantity -= btc_value_delta * (1 + fees)
            self.btc_quantity += btc_quantity_delta
            transaction = "BUY"
        elif btc_value_delta < 0:
            # Sell BTC (limit by available BTC)
            max_sell_value = self.btc_quantity * btc_price
            btc_value_delta = max(btc_value_delta, -max_sell_value)
            btc_quantity_delta = btc_value_delta / btc_price if btc_price > 0 else 0
            trade_cost = abs(btc_value_delta) * fees
            self.btc_quantity += btc_quantity_delta
            self.usdc_quantity += abs(btc_value_delta) * (1 - fees)
            transaction = "SELL"
        else:
            btc_quantity_delta = 0
            trade_cost = 0
            transaction = "NONE"
        
        new_btc_quantity = self.btc_quantity
        new_portfolio = self.calculate_portfolio_value(btc_price, new_btc_quantity)
        
        return {
            'transaction': transaction,
            'btc_quantity_change': btc_quantity_delta,
            'btc_value_change': btc_value_delta,
            'trade_cost': trade_cost,
            'old_portfolio': current_portfolio,
            'new_portfolio': new_portfolio,
            'new_btc_quantity': new_btc_quantity,
        }
    
    def add_to_history(self, date: str, btc_price: float, btc_quantity: float,
                       signal_ratio: float, signal: str) -> dict:
        """Append a portfolio snapshot to history."""
        portfolio_value = self.calculate_portfolio_value(btc_price, btc_quantity)
        
        snapshot = {
            'date': pd.to_datetime(date),
            'btc_price': btc_price,
            'btc_quantity': btc_quantity,
            'btc_weight': portfolio_value['btc_weight'],
            'usdc_weight': portfolio_value['usdc_weight'],
            'btc_value': portfolio_value['btc_value'],
            'usdc_value': portfolio_value['usdc_value'],
            'total_value': portfolio_value['total_value'],
            'signal_ratio': signal_ratio,
            'signal': signal,
            'btc_price_change_pct': None,
        }
        self.portfolio_history.append(snapshot)
        return snapshot
    
    def get_portfolio_dataframe(self) -> pd.DataFrame:
        """Return portfolio history as a DataFrame indexed by date."""
        df = pd.DataFrame(self.portfolio_history)
        df.set_index('date', inplace=True)
        df['price_change_pct'] = df['btc_price'].pct_change() * 100
        df['portfolio_value_change_pct'] = df['total_value'].pct_change() * 100
        return df
    
    def calculate_metrics(self, df: pd.DataFrame) -> dict:
        """Calculate key performance metrics from portfolio history DataFrame."""
        total_return = (df['total_value'].iloc[-1] / df['total_value'].iloc[0] - 1) * 100
        daily_returns = df['total_value'].pct_change().dropna()
        sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if daily_returns.std() > 0 else 0

        # Sortino: penalises only downside volatility
        downside_returns = daily_returns[daily_returns < 0]
        downside_std = downside_returns.std()
        sortino_ratio = daily_returns.mean() / downside_std * np.sqrt(252) if downside_std > 0 else 0

        running_max = df['total_value'].cummax()
        drawdown = (df['total_value'] - running_max) / running_max
        max_drawdown = drawdown.min() * 100

        # Calmar: CAGR / |MaxDrawdown|
        n_years = len(daily_returns) / 252
        cagr = ((df['total_value'].iloc[-1] / df['total_value'].iloc[0]) ** (1 / n_years) - 1) * 100 if n_years > 0 else 0
        calmar_ratio = cagr / abs(max_drawdown) if max_drawdown != 0 else 0

        btc_trades = (df['btc_weight'].diff().abs() > 1e-6).sum()
        
        return {
            'total_return_pct': total_return,
            'avg_daily_return_pct': daily_returns.mean() * 100,
            'daily_volatility_pct': daily_returns.std() * 100,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'cagr_pct': cagr,
            'max_drawdown_pct': max_drawdown,
            'num_trades': btc_trades,
            'final_value': df['total_value'].iloc[-1],
            'final_btc_weight': df['btc_weight'].iloc[-1],
        }

