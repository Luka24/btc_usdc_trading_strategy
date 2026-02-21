"""
Professional Trading Strategy Engine
=====================================
Implements the complete BTC/USDC adaptive allocation strategy.

Components:
- Signal Layer: Ensemble signals (Trend, Momentum, Production Cost)
- Decision Layer: Score computation and target allocation
- Execution Layer: Confirmation checks and staged execution

Professional trading rulebook implementation v3.0
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Tuple, Optional
from signal_calculator import SignalCalculator
from confirmation_layer import ConfirmationLayer


class TradingStrategy:
    """Professional BTC/USDC adaptive allocation strategy."""
    
    def __init__(self, initial_capital: float = 100_000):
        """
        Initialize strategy.
        
        Args:
            initial_capital: Initial capital (USD)
        """
        self.initial_capital = initial_capital
        self.current_position_btc = 0.7  # Start 70% BTC (long-term growth asset)
        self.current_position_usdc = 0.3  # 30% USDC (defensive reserve)
        
        # Signal calculator - optimized for Sharpe
        self.signal_calc = SignalCalculator(
            trend_window=3,  # Increased to 3 for smoother signals
            momentum_alpha=0.4,  # Reduced to 0.4 for stability
            prodcost_window=5,  # Keep at 5 for reasonable reaction
            score_alpha=0.5  # Increased to 0.5 for more stable scores (less volatility)
        )
        
        # Confirmation layer - Optimized for Sharpe ratio
        self.confirmation = ConfirmationLayer(
            zone_consistency_days=2,
            magnitude_threshold=0.35,  # Higher filtering for stability
            direction_threshold=0.10,  # More deliberate moves
            extreme_score_threshold=1.8,  # Quick extreme detection
            min_rebalance_threshold=0.10  # 10% minimum - reduces volatility, improves Sharpe
        )
        
        # Execution state
        self.execution_stage = None  # "Day1" or "Day2"
        self.pending_execution = 0.0
        
        # History
        self.execution_log = []
        self.price_history = []
        self.cost_ratios = []  # For extreme detection
    
    def run_daily_cycle(self, date, btc_price: float, production_cost: float,
                       prices_last_200: list, prices_last_90: list,
                       daily_returns_30: np.ndarray,
                       hashrate: float = None) -> Dict:
        """
        Run complete daily trading cycle at 00:00 UTC.
        
        Args:
            date: Current date
            btc_price: Current BTC price
            production_cost: Production cost calculated from mining module
            prices_last_200: Last 200 daily prices (for SMA200)
            prices_last_90: Last 90 daily prices (for momentum lookback)
            daily_returns_30: Array of last 30 daily returns
            hashrate: Current hashrate (optional)
            
        Returns:
            Dict with all calculation details and execution decision
        """
        
        # ========== STEP 1: CALCULATE RAW SIGNALS ==========
        
        # Trend signal
        sma_200 = np.mean(prices_last_200[-200:]) if len(prices_last_200) >= 200 else np.mean(prices_last_200)
        trend_raw = self.signal_calc.calculate_trend_signal_raw(btc_price, sma_200)
        
        # Momentum signal
        price_60d_ago = prices_last_90[-61] if len(prices_last_90) >= 61 else prices_last_90[0]
        momentum_raw = self.signal_calc.calculate_momentum_signal_raw(btc_price, price_60d_ago)
        
        # Production cost signal
        cost_ratio = btc_price / production_cost if production_cost > 0 else 1.0
        prodcost_raw = self.signal_calc.calculate_prodcost_signal_raw(cost_ratio)
        
        # Volatility regime
        vol_regime, annual_vol = self.signal_calc.calculate_volatility_regime(daily_returns_30)
        
        # ========== STEP 2: SMOOTH INDIVIDUAL SIGNALS ==========
        
        trend_smooth = self.signal_calc.smooth_trend_signal(trend_raw)
        momentum_smooth = self.signal_calc.smooth_momentum_signal_ema(momentum_raw)
        prodcost_smooth = self.signal_calc.smooth_prodcost_signal(prodcost_raw)
        
        # ========== STEP 3: COMPUTE RAW SCORE ==========
        
        score_raw = self.signal_calc.calculate_score_raw(
            trend_smooth, momentum_smooth, prodcost_smooth,
            w_trend=1.0, w_momentum=1.0, w_prodcost=1.5
        )
        
        # ========== STEP 4: SMOOTH SCORE (Double Smoothing) ==========
        
        score_smooth = self.signal_calc.smooth_score_ema(score_raw)
        
        # ========== STEP 5: ADJUST FOR VOLATILITY ==========
        
        score_adjusted = self.signal_calc.adjust_score_for_volatility(score_smooth, vol_regime)
        
        # ========== STEP 6: MAP TO TARGET ALLOCATION ==========
        
        target_btc = self.signal_calc.map_score_to_target(score_adjusted)
        
        # ========== STEP 7: CHECK EXTREME PRODUCTION COST ==========
        
        self.cost_ratios.append(cost_ratio)
        extreme_flag, z_score = self.signal_calc.check_extreme_production_cost(
            cost_ratio, 
            self.cost_ratios[-90:] if len(self.cost_ratios) >= 10 else self.cost_ratios
        )
        target_btc = self.signal_calc.apply_extreme_adjustment(target_btc, extreme_flag)
        
        # ========== STEP 8: GET PREVIOUS TARGET (FOR CONFIRMATION) ==========
        
        if self.confirmation.target_history:
            target_yesterday = self.confirmation.target_history[-1]['target']
            score_yesterday = self.confirmation.target_history[-1]['score']
        else:
            target_yesterday = self.current_position_btc
            score_yesterday = score_adjusted
        
        # ========== STEP 9: MULTI-LAYER CONFIRMATION ==========
        
        # IMPORTANT: If we are in middle of staged execution (Day1), 
        # we MUST continue with Day2 regardless of new signals
        if self.execution_stage == "Day1":
            # Force execution of Day 2 to complete the staged trade
            confirmed = True
            confirmation_reason = "Completing Day 2 of staged execution"
            confidence = 1.0
        else:
            # Normal confirmation check
            confirmed, confirmation_reason, confidence = self.confirmation.check_confirmation(
                self.current_position_btc,
                target_btc,
                target_yesterday,
                score_adjusted,
                score_yesterday
            )
        
        # ========== STEP 10: EXECUTE (if confirmed) ==========
        
        execution_info = {
            'executed': False,
            'execution_stage': None,
            'amount_executed': 0.0,
            'new_position': self.current_position_btc
        }
        
        if confirmed:
            execution_info = self._execute_staged(target_btc)
        
        # ========== STEP 11: LOG & UPDATE STATE ==========
        
        self.confirmation.add_target_to_history(date, target_btc, score_adjusted)
        
        log_entry = {
            'date': date,
            'btc_price': btc_price,
            'production_cost': production_cost,
            'cost_ratio': cost_ratio,
            'sma_200': sma_200,
            'annual_vol': annual_vol,
            'vol_regime': vol_regime,
            
            # Raw signals
            'trend_raw': trend_raw,
            'momentum_raw': momentum_raw,
            'prodcost_raw': prodcost_raw,
            
            # Smoothed signals
            'trend_smooth': trend_smooth,
            'momentum_smooth': momentum_smooth,
            'prodcost_smooth': prodcost_smooth,
            
            # Scores
            'score_raw': score_raw,
            'score_smooth': score_smooth,
            'score_adjusted': score_adjusted,
            
            # Confirmation
            'target_btc': target_btc,
            'confirmed': confirmed,
            'confirmation_reason': confirmation_reason,
            'confidence': confidence,
            'extreme_flag': extreme_flag,
            'z_score': z_score,
            
            # Execution
            **execution_info,
            
            # Position
            'current_position_btc': self.current_position_btc,
            'current_position_usdc': self.current_position_usdc,
        }
        
        self.execution_log.append(log_entry)
        self.price_history.append(btc_price)
        
        return log_entry
    
    def _execute_staged(self, target_btc: float) -> Dict:
        """
        Execute trade in 2 stages (60/40 split).
        
        Args:
            target_btc: Target BTC allocation
            
        Returns:
            Dict with execution details
        """
        change_needed = target_btc - self.current_position_btc
        
        if self.execution_stage is None:
            # Start new 2-day execution
            execute_pct_1 = 0.60 * change_needed
            self.current_position_btc += execute_pct_1
            self.current_position_usdc -= execute_pct_1
            self.execution_stage = "Day1"
            self.pending_execution = 0.40 * change_needed
            
            return {
                'executed': True,
                'execution_stage': 'Day 1 of 2',
                'amount_executed': execute_pct_1,
                'new_position': self.current_position_btc,
            }
        
        elif self.execution_stage == "Day1":
            # Continue 2-day execution (Day 2)
            execute_pct_2 = self.pending_execution
            self.current_position_btc += execute_pct_2
            self.current_position_usdc -= execute_pct_2
            self.execution_stage = None
            self.pending_execution = 0.0
            
            return {
                'executed': True,
                'execution_stage': 'Day 2 (complete)',
                'amount_executed': execute_pct_2,
                'new_position': self.current_position_btc,
            }
        
        else:
            # No execution if already in 2-day cycle
            return {
                'executed': False,
                'execution_stage': 'Waiting for Day 2',
                'amount_executed': 0.0,
                'new_position': self.current_position_btc,
            }
    
    def get_execution_log_df(self) -> pd.DataFrame:
        """Get execution log as DataFrame."""
        if not self.execution_log:
            return pd.DataFrame()
        return pd.DataFrame(self.execution_log)
    
    def get_summary(self) -> str:
        """Get strategy summary."""
        df = self.get_execution_log_df()
        
        if df.empty:
            return "No execution log yet."
        
        summary = f"""
BTC/USDC ADAPTIVE ALLOCATION STRATEGY - SUMMARY
{'='*70}

Period:              {df['date'].iloc[0]} to {df['date'].iloc[-1]}
Days:                {len(df)}

Latest Position:
  BTC:               {self.current_position_btc:.1%}
  USDC:              {self.current_position_usdc:.1%}

Price:
  Current:           ${df['btc_price'].iloc[-1]:,.0f}
  Range:             ${df['btc_price'].min():,.0f} - ${df['btc_price'].max():,.0f}

Score (Latest):
  Raw:               {df['score_raw'].iloc[-1]:+.2f}
  Smoothed:          {df['score_smooth'].iloc[-1]:+.2f}
  Adjusted:          {df['score_adjusted'].iloc[-1]:+.2f}

Volatility:
  Current:           {df['annual_vol'].iloc[-1]:.1%}
  Regime:            {df['vol_regime'].iloc[-1]}

Confirmations:
  Total:             {len(df)}
  Confirmed:         {df['confirmed'].sum()}
  Confirmed %:       {df['confirmed'].sum()/len(df):.1%}

Executions:
  Total Executed:    {df['executed'].sum()}

{'='*70}
"""
        return summary
