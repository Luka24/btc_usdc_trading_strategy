"""
RISK MANAGEMENT MODUL
=====================
"""

import pandas as pd
import numpy as np
from enum import Enum
from typing import Dict, Tuple, Optional


class RiskMode(Enum):
    """
Risk modes"""
    NORMAL = "NORMAL"
    RISK_OFF = "RISK_OFF"
    KILL_SWITCH = "KILL_SWITCH"


class RiskManager:
    """
    Comprehensive system za upravljanje tveganja.
    Combinesh razlicne kontrole tveganja.
    """
    
    def __init__(self):
        """
Initialize risk managerra"""
        
        self.peak_value = 100_000
        self.current_value = 100_000
        self.peak_date = None
        
        # Volatility
        self.recent_returns = []
        self.vol_lookback = 30
        
        # Mode
        self.current_mode = RiskMode.NORMAL
        self.mode_history = []
        
        # Logiranje
        self.risk_log = []
    
    # ============ 1. DRAWDOWN KONTROLA ============
    
    def update_peak(self, current_value: float, date: str) -> None:
        """Update peak value."""
        if current_value > self.peak_value:
            self.peak_value = current_value
            self.peak_date = date
    
    def calculate_drawdown(self, current_value: float) -> float:
        """Calculate current drawdown."""
        if self.peak_value == 0:
            return 0.0
        return (current_value - self.peak_value) / self.peak_value
    
    def check_drawdown_limit(self, current_value: float, limit: float = -0.20) -> Tuple[bool, float]:
        """Check if drawdown exceeded limit."""
        drawdown = self.calculate_drawdown(current_value)
        exceeded = drawdown < limit
        return exceeded, drawdown
    
    # ============ 2. VOLATILNOSTNI FILTER ============
    
    def update_returns(self, daily_return: float) -> None:
        """Add daily return to history."""
        self.recent_returns.append(daily_return)
        if len(self.recent_returns) > self.vol_lookback:
            self.recent_returns.pop(0)
    
    def calculate_volatility(self) -> float:
        """Calculate current volatility."""
        if len(self.recent_returns) < 2:
            return 0.0
        returns_array = np.array(self.recent_returns)
        daily_vol = np.std(returns_array)
        annualized_vol = daily_vol * np.sqrt(252)
        return annualized_vol
    
    def check_volatility_threshold(self, vol_limit: float = 0.80) -> Tuple[bool, float]:
        """Check if volatility is above limit."""
        vol = self.calculate_volatility()
        exceeded = vol > vol_limit
        return exceeded, vol
    
    # ============ 3. VALUE-AT-RISK (VaR) ============
    
    def calculate_var(self, confidence: float = 0.99) -> float:
        """Calculate Value-at-Risk (VaR)."""
        if len(self.recent_returns) < 2:
            return 0.0
        returns_array = np.array(self.recent_returns)
        mean_return = np.mean(returns_array)
        std_return = np.std(returns_array)
        z_score = 2.33  # 99% confidence level
        var = mean_return - z_score * std_return
        return var
    
    def check_var_limit(self, var_limit: float = -0.04, portfolio_value: float = 100_000) -> Tuple[bool, float]:
        """Check if VaR exceeded limit."""
        var = self.calculate_var()
        exceeded = var < var_limit
        var_amount = var * portfolio_value
        return exceeded, var_amount
    
    # ============ 4. REZIMSKI FILTER ============
    
    def calculate_sma(self, prices: np.ndarray, window: int = 200) -> float:
        """Calculate simple moving average (SMA)."""
        if len(prices) < window:
            return prices.mean()
        return prices[-window:].mean()
    
    def detect_market_regime(self, price: float, sma_200: float) -> str:
        """Detect market regime."""
        if price > sma_200 * 1.05:
            return "UPTREND"
        elif price < sma_200 * 0.95:
            return "DOWNTREND"
        else:
            return "NEUTRAL"
    
    # ============ 5. LIKVIDNOSTNE KONTROLE ============
    
    def check_liquidity(self, volume_24h: float, spread_bps: float,
                        min_volume: float = 300_000_000, max_spread: float = 20) -> Tuple[bool, str]:
        """Check liquidity."""
        reasons = []
        if volume_24h < min_volume:
            reasons.append(f"Volume too low: ${volume_24h/1e6:.1f}M < ${min_volume/1e6:.1f}M")
        if spread_bps > max_spread:
            reasons.append(f"Spread too wide: {spread_bps:.0f} bps > {max_spread:.0f} bps")
        ok = len(reasons) == 0
        reason = "; ".join(reasons) if reasons else "OK"
        return ok, reason
    
    # ============ 6. STOP-LOSS IN TAKE-PROFIT ============
    
    def check_stop_loss(self, entry_price: float, current_price: float, 
                       stop_loss_pct: float = -0.12) -> Tuple[bool, float]:
        """Check stop-loss."""
        loss_pct = (current_price - entry_price) / entry_price
        triggered = loss_pct < stop_loss_pct
        return triggered, loss_pct
    
    def check_take_profit(self, entry_price: float, current_price: float,
                         take_profit_pct: float = 0.20) -> Tuple[bool, float]:
        """Check take-profit."""
        profit_pct = (current_price - entry_price) / entry_price
        triggered = profit_pct > take_profit_pct
        return triggered, profit_pct
    
    # ============ 7. REZIMSKI SWITCH ============
    
    def determine_risk_mode(self, drawdown: float, volatility: float, 
                           var_exceeded: bool) -> RiskMode:
        """Determine risk regime."""
        if drawdown < -0.30 or volatility > 1.50:
            return RiskMode.EMERGENCY
        if drawdown < -0.20 or volatility > 0.80 or var_exceeded:
            return RiskMode.RISK_OFF
        if drawdown < -0.10 or volatility > 0.60:
            return RiskMode.CAUTION
        return RiskMode.NORMAL
    
    def apply_mode_adjustment(self, target_btc_weight: float, mode: RiskMode) -> float:
        """Adjust BTC weight based on risk regime."""
        if mode == RiskMode.KILL_SWITCH:
            return 0.0
        elif mode == RiskMode.EMERGENCY:
            return 0.0
        elif mode == RiskMode.RISK_OFF:
            return min(target_btc_weight, 0.30)
        elif mode == RiskMode.CAUTION:
            return min(target_btc_weight, 0.60)
        else:
            return target_btc_weight
    
    # ============ 8. COMPREHENSIVE RISK EVALUATION ============
    
    def evaluate_risk(self, portfolio_value: float, daily_return: float,
                     btc_price: float, hashrate: float) -> Dict:
        """Comprehensive risk evaluation."""
        self.update_peak(portfolio_value, "today")
        self.current_value = portfolio_value
        
        dd_exceeded, drawdown = self.check_drawdown_limit(portfolio_value)
        vol_exceeded, volatility = self.check_volatility_threshold()
        var_exceeded, var_amount = self.check_var_limit(portfolio_value=portfolio_value)
        
        risk_mode = self.determine_risk_mode(drawdown, volatility, var_exceeded)
        
        report = {
            'date': pd.Timestamp.now(),
            'portfolio_value': portfolio_value,
            'daily_return_pct': daily_return * 100,
            'drawdown_pct': drawdown * 100,
            'drawdown_exceeded': dd_exceeded,
            'volatility_pct': volatility * 100,
            'volatility_exceeded': vol_exceeded,
            'var_daily_pct': self.calculate_var() * 100,
            'var_exceeded': var_exceeded,
            'var_amount_usd': var_amount,
            'risk_mode': risk_mode.value,
        }
        
        self.risk_log.append(report)
        self.current_mode = risk_mode
        
        return report
    
    def generate_risk_report(self) -> str:
        """Generate text risk report."""
        if len(self.risk_log) == 0:
            return "No risk data available yet."
        
        latest = self.risk_log[-1]
        dd_status = "EXCEEDED" if latest['drawdown_exceeded'] else "OK"
        vol_status = "HIGH" if latest['volatility_exceeded'] else "OK"
        var_status = "EXCEEDED" if latest['var_exceeded'] else "OK"
        
        report = f"""
{'='*70}
RISK MANAGEMENT REPORT
{'='*70}

PORTFOLIO METRICS
{'-'*70}
Value:                     ${latest['portfolio_value']:>15,.2f}
Daily Return:              {latest['daily_return_pct']:>14.2f}%

DRAWDOWN
{'-'*70}
Current Drawdown:          {latest['drawdown_pct']:>14.2f}%
Status:                    {dd_status:>15}

VOLATILITY
{'-'*70}
Annualized Vol:            {latest['volatility_pct']:>14.2f}%
Status:                    {vol_status:>15}

VALUE-AT-RISK (1-day, 99%)
{'-'*70}
Daily VaR:                 {latest['var_daily_pct']:>14.2f}%
VaR Amount:                ${latest['var_amount_usd']:>15,.2f}
Status:                    {var_status:>15}

RISK MODE
{'-'*70}
Current Mode:              {latest['risk_mode']:>15}

{'='*70}
"""
        return report


if __name__ == "__main__":
    print("="*70)
    print("RISK MANAGEMENT SYSTEM - PRIMERI")
    print("="*70)
    
    rm = RiskManager()
    
    print("\nPRIMER 1: Drawdown Control")
    print("-"*70)
    peak = 100_000
    current = 80_000
    rm.peak_value = peak
    exceeded, dd = rm.check_drawdown_limit(current, limit=-0.20)
    print(f"Peak:     ${peak:,.0f}")
    print(f"Current:  ${current:,.0f}")
    print(f"Drawdown: {dd:.1%}")
    print(f"Exceeded: {exceeded} (Limit: -20%)")
    
    print("\nPRIMER 2: Volatilityni Filter")
    print("-"*70)
    returns = np.random.normal(0.00-1, 0.03, 30)
    for r in returns:
        rm.update_returns(r)
    exceeded, vol = rm.check_volatility_threshold(vol_limit=0.80)
    print(f"30-day returns: {len(returns)} dni")
    print(f"Daily volatility: {np.std(returns):.2%}")
    print(f"Annualized Vol: {vol:.2%}")
    print(f"Exceeded: {exceeded} (Limit: 80%)")
    
    print("\nPRIMER 3: Value-at-Risk (99%)")
    print("-"*70)
    var = rm.calculate_var(confidence=0.99)
    var_exceeded, var_amount = rm.check_var_limit(var_limit=-0.04, portfolio_value=100_000)
    print(f"Daily VaR (99%): {var:.2%}")
    print(f"VaR Amount: ${var_amount:,.2f}")
    print(f"Exceeded: {var_exceeded} (Limit: -4%)")
    
    print("\nPRIMER 4: Risk Mode Determination")
    print("-"*70)
    test_scenarios = [
        (-0.05, 0.50, False, "Normal market"),
        (-0.15, 0.75, False, "Higher risk"),
        (-0.22, 0.85, True, "Risk-off zone"),
        (-0.35, 1.60, True, "Kill switch"),
    ]
    
    print(f"\n{'Drawdown':>10} | {'Volatility':>11} | {'VaR Exceeded':>12} | {'Mode':>10}")
    print("-"*50)
    
    for dd, vol, var_exc, label in test_scenarios:
        mode = rm.determine_risk_mode(dd, vol, var_exc)
        print(f"{dd:>9.1%} | {vol:>10.1%} | {str(var_exc):>12} | {mode.value:>10}")
        print(f"         ({label})")
    
    print("\n" + "="*70)
    print("Risk Management System Ready!")
    print("="*70)
