import pandas as pd
import numpy as np
from enum import Enum


class RiskMode(Enum):
    NORMAL = "NORMAL"
    CAUTION = "CAUTION"
    RISK_OFF = "RISK_OFF"
    EMERGENCY = "EMERGENCY"
    KILL_SWITCH = "KILL_SWITCH"


class RiskManager:
    """Risk management system."""
    
    def __init__(self):
        self.peak_value = 100_000
        self.current_value = 100_000
        self.peak_date = None
        
        self.recent_returns = []
        self.vol_lookback = 30
        
        self.current_mode = RiskMode.NORMAL
        self.mode_history = []
        self.risk_log = []
    
    def update_peak(self, val, date):
        if val > self.peak_value:
            self.peak_value = val
            self.peak_date = date
    
    def get_drawdown(self, val):
        if self.peak_value == 0:
            return 0.0
        return (val - self.peak_value) / self.peak_value
    
    def check_drawdown_limit(self, val, limit=-0.20):
        dd = self.get_drawdown(val)
        exceeded = dd < limit
        return exceeded, dd
    
    def update_returns(self, ret):
        self.recent_returns.append(ret)
        if len(self.recent_returns) > self.vol_lookback:
            self.recent_returns.pop(0)
    
    def get_volatility(self):
        if len(self.recent_returns) < 2:
            return 0.0
        returns = np.array(self.recent_returns)
        daily_vol = np.std(returns)
        annual_vol = daily_vol * np.sqrt(252)
        return annual_vol
    
    def check_volatility_threshold(self, vol_limit=0.80):
        vol = self.get_volatility()
        exceeded = vol > vol_limit
        return exceeded, vol
    
    def calculate_var(self, confidence=0.99):
        if len(self.recent_returns) < 2:
            return 0.0
        returns = np.array(self.recent_returns)
        mean_ret = np.mean(returns)
        std_ret = np.std(returns)
        z_score = 2.33
        var = mean_ret - z_score * std_ret
        return var
    
    def check_var_limit(self, var_limit=-0.04, portfolio_value=100_000):
        var = self.calculate_var()
        exceeded = var < var_limit
        var_amount = var * portfolio_value
        return exceeded, var_amount
    
    def calc_sma(self, prices, window=200):
        if len(prices) < window:
            return prices.mean()
        return prices[-window:].mean()
    
    def detect_regime(self, price, sma_200):
        if price > sma_200 * 1.05:
            return "UPTREND"
        elif price < sma_200 * 0.95:
            return "DOWNTREND"
        return "NEUTRAL"
    
    def check_liquidity(self, volume_24h, spread_bps, 
                       min_vol=300_000_000, max_spread=20):
        reasons = []
        if volume_24h < min_vol:
            reasons.append(f"Vol too low: ${volume_24h/1e6:.1f}M")
        if spread_bps > max_spread:
            reasons.append(f"Spread: {spread_bps:.0f} bps")
        ok = len(reasons) == 0
        reason = "; ".join(reasons) if reasons else "OK"
        return ok, reason
    
    def check_stop_loss(self, entry, current, stop_loss_pct=-0.12):
        loss_pct = (current - entry) / entry
        triggered = loss_pct < stop_loss_pct
        return triggered, loss_pct
    
    def check_take_profit(self, entry, current, take_profit_pct=0.20):
        profit_pct = (current - entry) / entry
        triggered = profit_pct > take_profit_pct
        return triggered, profit_pct
    
    def get_risk_mode(self, dd, vol, var_exc):
        if dd < -0.30 or vol > 1.50:
            return RiskMode.EMERGENCY
        if dd < -0.20 or vol > 0.80 or var_exc:
            return RiskMode.RISK_OFF
        if dd < -0.10 or vol > 0.60:
            return RiskMode.CAUTION
        return RiskMode.NORMAL
    
    def adjust_weight(self, target_weight, mode):
        if mode == RiskMode.EMERGENCY or mode == RiskMode.KILL_SWITCH:
            return 0.0
        if mode == RiskMode.RISK_OFF:
            return min(target_weight, 0.30)
        if mode == RiskMode.CAUTION:
            return min(target_weight, 0.60)
        return target_weight
    
    def evaluate_risk(self, portfolio_val, daily_ret, btc_price, hashrate):
        self.update_peak(portfolio_val, "today")
        self.current_value = portfolio_val
        
        dd_exc, dd = self.check_drawdown_limit(portfolio_val)
        vol_exc, vol = self.check_volatility_threshold()
        var_exc, var_amt = self.check_var_limit(portfolio_value=portfolio_val)
        
        mode = self.get_risk_mode(dd, vol, var_exc)
        
        report = {
            'date': pd.Timestamp.now(),
            'portfolio_value': portfolio_val,
            'daily_return_pct': daily_ret * 100,
            'drawdown_pct': dd * 100,
            'drawdown_exceeded': dd_exc,
            'volatility_pct': vol * 100,
            'volatility_exceeded': vol_exc,
            'var_daily_pct': self.calculate_var() * 100,
            'var_exceeded': var_exc,
            'var_amount_usd': var_amt,
            'risk_mode': mode.value,
        }
        
        self.risk_log.append(report)
        self.current_mode = mode
        
        return report
    
    def generate_risk_report(self):
        if not self.risk_log:
            return "No risk data available."
        
        latest = self.risk_log[-1]
        dd_status = "EXCEEDED" if latest['drawdown_exceeded'] else "OK"
        vol_status = "HIGH" if latest['volatility_exceeded'] else "OK"
        var_status = "EXCEEDED" if latest['var_exceeded'] else "OK"
        
        return f"""
{'='*70}
RISK MANAGEMENT REPORT
{'='*70}

PORTFOLIO
Value:                     ${latest['portfolio_value']:>15,.2f}
Daily Return:              {latest['daily_return_pct']:>14.2f}%

DRAWDOWN
Current:                   {latest['drawdown_pct']:>14.2f}%
Status:                    {dd_status:>15}

VOLATILITY
Annualized:                {latest['volatility_pct']:>14.2f}%
Status:                    {vol_status:>15}

VALUE-AT-RISK (1d, 99%)
Daily VaR:                 {latest['var_daily_pct']:>14.2f}%
Amount:                    ${latest['var_amount_usd']:>15,.2f}
Status:                    {var_status:>15}

RISK MODE
Current:                   {latest['risk_mode']:>15}

{'='*70}
"""


if __name__ == "__main__":
    print("=" * 70)
    print("RISK MANAGEMENT SYSTEM - EXAMPLES")
    print("=" * 70)
    
    rm = RiskManager()
    
    print("\nTest 1: Drawdown Control")
    print("-" * 70)
    peak = 100_000
    current = 80_000
    rm.peak_value = peak
    exceeded, dd = rm.check_drawdown_limit(current, limit=-0.20)
    print(f"Peak:      ${peak:,.0f}")
    print(f"Current:   ${current:,.0f}")
    print(f"Drawdown:  {dd:.1%}")
    print(f"Exceeded:  {exceeded} (Limit: -20%)")
    
    print("\nTest 2: Volatility Filter")
    print("-" * 70)
    returns = np.random.normal(0.001, 0.03, 30)
    for r in returns:
        rm.update_returns(r)
    exceeded, vol = rm.check_volatility_threshold(vol_limit=0.80)
    print(f"Days:           {len(returns)}")
    print(f"Daily vol:      {np.std(returns):.2%}")
    print(f"Annual vol:     {vol:.2%}")
    print(f"Exceeded:       {exceeded} (Limit: 80%)")
    
    print("\nTest 3: Value-at-Risk")
    print("-" * 70)
    var = rm.calculate_var(confidence=0.99)
    var_exc, var_amt = rm.check_var_limit(var_limit=-0.04, portfolio_value=100_000)
    print(f"Daily VaR:      {var:.2%}")
    print(f"Amount:         ${var_amt:,.2f}")
    print(f"Exceeded:       {var_exc} (Limit: -4%)")
    
    print("\nTest 4: Risk Mode")
    print("-" * 70)
    scenarios = [
        (-0.05, 0.50, False, "Normal"),
        (-0.15, 0.75, False, "Higher risk"),
        (-0.22, 0.85, True, "Risk-off"),
        (-0.35, 1.60, True, "Emergency"),
    ]
    
    for dd, vol, var_exc, label in scenarios:
        mode = rm.get_risk_mode(dd, vol, var_exc)
        print(f"{dd:>8.1%} | {vol:>8.1%} | {mode.value:>12} ({label})")
    
    print("\n" + "=" * 70)
    print("Risk Manager Ready")
    print("=" * 70)
