"""
Module for portfolio management
==============================
Manages positioning (BTC/USDC), rebalancing and portfolio weight calculations.
"""

import pandas as pd
import numpy as np
from config import PortfolioConfig as Config


class PortfolioManager:
    """
    Portfolio manager for BTC/USDC.
    Determines BTC weight based on signal and maintains constraints.
    """
    
    def __init__(self,
                 initial_capital: float = Config.INITIAL_PORTFOLIO_USD,
                 position_table: list = Config.POSITION_TABLE,
                 max_daily_change: float = Config.MAX_DAILY_WEIGHT_CHANGE):
        """
        Initialize.
        
        Args:
            initial_capital (float): Initial capital (USD)
            position_table (list): Table (min_ratio, max_ratio, btc_weight)
            max_daily_change (float): Maximum daily weight change
        """
        self.initial_capital = initial_capital
        self.position_table = position_table
        self.max_daily_change = max_daily_change
        
        self.portfolio_history = []
        self.current_btc_weight = 0.50  # Start with 50% BTC
        self.previous_btc_weight = 0.50
        self.btc_quantity = 0.0
        self.usdc_quantity = self.initial_capital

    def initialize_holdings(self, btc_price: float, initial_btc_quantity: float) -> None:
        """
        Initialize BTC and USDC holdings based on initial capital and price.
        """
        btc_value = initial_btc_quantity * btc_price
        if btc_value > self.initial_capital:
            initial_btc_quantity = self.initial_capital / btc_price
            btc_value = initial_btc_quantity * btc_price

        self.btc_quantity = initial_btc_quantity
        self.usdc_quantity = self.initial_capital - btc_value
    
    def determine_target_weight(self, signal_ratio: float) -> float:
        """
        Determine target BTC weight glede na ratio price/cost.
        
        Args:
            signal_ratio (float): Price/cost ratio
            
        Returns:
            float: Target BTC weight (0-1)
        """
        
        for min_ratio, max_ratio, weight in self.position_table:
            if min_ratio <= signal_ratio <= max_ratio:
                return weight
        
        # Fallback (if ratio outside table)
        if signal_ratio > self.position_table[-1][1]:
            return self.position_table[-1][2]
        else:
            return self.position_table[0][2]
    
    def apply_weight_change_limit(self, target_weight: float) -> float:
        """
        Limit daily change teze.
        
        Rule: weight cannot cannot change by more than max_daily_change per day.
        
        Args:
            target_weight (float): Desired weight
            
        Returns:
            float: Actual allowed weight
        """
        
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
        """
        Rebalance portfolio based on signala.
        
        Args:
            signal_ratio (float): Price/cost ratio
            enforce_limit (bool): Ali upostevati dnevno limit
            
        Returns:
            dict: Result rebalancinga
        """
        
        # Doloci ciljno tezo
        target_weight = self.determine_target_weight(signal_ratio)
        
        # Limititev to daily change
        if enforce_limit:
            actual_weight = self.apply_weight_change_limit(target_weight)
        else:
            actual_weight = target_weight
        
        # Save preteklo tezo
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
        """
        Calculate value portfelja.
        
        Args:
            btc_price (float): Price 1 BTC (USD)
            btc_quantity (float): Kolicina BTC
            
        Returns:
            dict: Value komponent in skupna vrednost
        """
        
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
        """
        Execute rebalancing (nakup/prodajo BTC).
        
        Args:
            btc_price (float): Current cena BTC
            btc_quantity (float): Current kolicina BTC
            target_weight (float): Target weight BTC
            fees (float): Fees than delez (npr. 0.00-1 = 0.1%)
            
        Returns:
            dict: Result izvedbe
        """
        
        # Current value portfelja
        current_portfolio = self.calculate_portfolio_value(btc_price, self.btc_quantity)
        current_btc_value = current_portfolio['btc_value']
        current_usdc_value = current_portfolio['usdc_value']
        total_value = current_btc_value + current_usdc_value
        
        # Target value BTC
        target_btc_value = total_value * target_weight
        
        # Difference
        btc_value_delta = target_btc_value - current_btc_value
        
        # Quantity changecine
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
        
        # Nova kolicina
        new_btc_quantity = self.btc_quantity

        # New value
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
        """
        Add portfolio snapshot v zgodovino.
        
        Args:
            date (str): Datum
            btc_price (float): Price BTC
            btc_quantity (float): Kolicina BTC
            signal_ratio (float): Ratio signala
            signal (str): Tip signala (BUY, HOLD, SELL)
            
        Returns:
            dict: Snapshot portfelja
        """
        
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
            'btc_price_change_pct': None,  # Calculateamo kasneje
        }
        
        self.portfolio_history.append(snapshot)
        
        return snapshot
    
    def get_portfolio_dataframe(self) -> pd.DataFrame:
        """
        Return history portfelja than DataFrame.
        
        Returns:
            pd.DataFrame: History portfelja
        """
        df = pd.DataFrame(self.portfolio_history)
        df.set_index('date', inplace=True)
        
        # Calculateaj dnevne changes
        df['price_change_pct'] = df['btc_price'].pct_change() * 100
        df['portfolio_value_change_pct'] = df['total_value'].pct_change() * 100
        
        return df
    
    def calculate_metrics(self, df: pd.DataFrame) -> dict:
        """
        Calculate key metrics portfelja.
        
        Args:
            df (pd.DataFrame): DataFrame s zgodovino portfelja
            
        Returns:
            dict: Metrics
        """
        
        total_return = (df['total_value'].iloc[-1] / df['total_value'].iloc[0] - 1) * 100
        
        daily_returns = df['total_value'].pct_change().dropna()
        sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if daily_returns.std() > 0 else 0
        
        running_max = df['total_value'].cummax()
        drawdown = (df['total_value'] - running_max) / running_max
        max_drawdown = drawdown.min() * 100
        
        btc_trades = (df['btc_weight'].diff().abs() > 0.0-1).sum()
        
        return {
            'total_return_pct': total_return,
            'avg_daily_return_pct': daily_returns.mean() * 100,
            'daily_volatility_pct': daily_returns.std() * 100,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown_pct': max_drawdown,
            'num_trades': btc_trades,
            'final_value': df['total_value'].iloc[-1],
            'final_btc_weight': df['btc_weight'].iloc[-1],
        }


# ============ PRIMER UPORABE ============
if __name__ == "__main__":
    print("=" * 60)
    print("EXAMPLE: Portfolio management")
    print("=" * 60)
    
    manager = PortfolioManager(initial_capital=100_000)
    
    # Simulatesni podatki
    dates = pd.date_range(start="2024-0-1-0-1", periods=60, freq="D")
    btc_prices = np.linspace(40000, 50000, 60) + np.random.normal(0, 1000, 60)
    ratios = np.linspace(0.95, 1.05, 60) + np.random.normal(0, 0.05, 60)
    signals = ['BUY' if r < 0.90 else 'SELL' if r > 1.10 else 'HOLD' for r in ratios]
    
    btc_quantity = 2.0  # Start z 2 BTC
    
    print("\nSimulacija 60 dni:")
    print("-" * 60)
    
    for i, (date, price, ratio, signal) in enumerate(zip(dates, btc_prices, ratios, signals)):
        # Rebalance
        rebalance_result = manager.rebalance(ratio)
        
        # Addj v zgodovino
        manager.add_to_history(date.strftime('%Y-%m-%d'), price, btc_quantity, ratio, signal)
        
        if i < 5 or i >= len(dates) - 5:
            print(f"{date.strftime('%Y-%m-%d')} | BTC: ${price:7.0f} | "
                  f"Ratio: {ratio:.3f} | Signal: {signal:4} | "
                  f"BTC Weight: {rebalance_result['btc_weight']:.1%}")
        elif i == 5:
            print("... [middle rows omitted] ...")
    
    # Metrics
    print("\n" + "=" * 60)
    print("PORTFOLIO METRICS")
    print("=" * 60)
    
    df = manager.get_portfolio_dataframe()
    metrics = manager.calculate_metrics(df)
    
    for key, value in metrics.items():
        if 'pct' in key or 'ratio' in key or 'volatility' in key:
            print(f"{key}: {value:.2f}%")
        else:
            print(f"{key}: {value:.2f}")
