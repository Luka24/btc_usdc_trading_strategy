"""
Signal Calculator Module
=========================
Calculates all three ensemble signals (Trend, Momentum, Production Cost)
with proper smoothing (SMA/EMA).

Professional trading rulebook implementation.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict
from collections import deque


class SignalCalculator:
    """Calculates trend, momentum, and production cost signals with smoothing."""
    
    def __init__(self, 
                 trend_window: int = 3,
                 momentum_alpha: float = 0.4,
                 prodcost_window: int = 7,
                 score_alpha: float = 0.3):
        """
        Initialize signal calculator.
        
        Args:
            trend_window: SMA window for trend signal (days)
            momentum_alpha: EMA alpha for momentum (0-1)
            prodcost_window: SMA window for production cost (days)
            score_alpha: EMA alpha for final score (0-1)
        """
        self.trend_window = trend_window
        self.momentum_alpha = momentum_alpha
        self.prodcost_window = prodcost_window
        self.score_alpha = score_alpha
        
        # History for smoothing
        self.trend_raw_history = deque(maxlen=trend_window)
        self.momentum_raw_history = deque(maxlen=prodcost_window)  # For prodcost SMA
        self.prodcost_raw_history = deque(maxlen=prodcost_window)
        
        # EMA states
        self.momentum_ema = None
        self.score_ema = None
        
        # Signal history
        self.signal_history = []
    
    # ========== TREND SIGNAL ==========
    
    def calculate_trend_signal_raw(self, price: float, sma_200: float, 
                                   hysteresis: float = 0.01) -> float:
        """
        Calculate raw trend signal.
        
        Args:
            price: Current BTC price
            sma_200: 200-day SMA
            hysteresis: Buffer ±hysteresis around SMA (default 1% - more sensitive)
            
        Returns:
            float: Raw trend signal (+1, 0, -1)
        """
        upper_buffer = sma_200 * (1 + hysteresis)
        lower_buffer = sma_200 * (1 - hysteresis)
        
        if price > upper_buffer:
            return 1.0
        elif price < lower_buffer:
            return -1.0
        else:
            return 0.0
    
    def smooth_trend_signal(self, raw_signal: float) -> float:
        """
        Smooth trend signal using SMA.
        
        Args:
            raw_signal: Raw trend signal
            
        Returns:
            float: Smoothed trend signal
        """
        self.trend_raw_history.append(raw_signal)
        
        if len(self.trend_raw_history) < self.trend_window:
            return raw_signal
        
        return float(np.mean(list(self.trend_raw_history)))
    
    # ========== MOMENTUM SIGNAL ==========
    
    def calculate_momentum_signal_raw(self, price_today: float, price_60d_ago: float,
                                      threshold: float = 0.03) -> float:
        """
        Calculate raw momentum signal.
        
        Args:
            price_today: Current price
            price_60d_ago: Price 60 days ago
            threshold: Threshold for bullish/bearish (default 3% - more sensitive)
            
        Returns:
            float: Raw momentum signal (+1, 0, -1)
        """
        momentum = (price_today - price_60d_ago) / price_60d_ago
        
        if momentum > threshold:
            return 1.0
        elif momentum < -threshold:
            return -1.0
        else:
            return 0.0
    
    def smooth_momentum_signal_ema(self, raw_signal: float) -> float:
        """
        Smooth momentum signal using EMA (α=0.4).
        
        Args:
            raw_signal: Raw momentum signal
            
        Returns:
            float: Smoothed momentum signal
        """
        if self.momentum_ema is None:
            self.momentum_ema = raw_signal
        else:
            self.momentum_ema = (self.momentum_alpha * raw_signal + 
                               (1 - self.momentum_alpha) * self.momentum_ema)
        
        return self.momentum_ema
    
    # ========== PRODUCTION COST SIGNAL ==========
    
    def calculate_prodcost_signal_raw(self, cost_ratio: float) -> float:
        """
        Calculate raw production cost signal.
        
        Maps cost ratio to signal based on 5 zones (more aggressive).
        
        Args:
            cost_ratio: Price / Production Cost ratio
            
        Returns:
            float: Raw production cost signal (+1.0, +0.5, 0, -0.5, -1.0)
        """
        if cost_ratio >= 1.15:  # Reduced from 1.20 for earlier bullish signal
            return 1.0  # Profit zone
        elif 1.05 <= cost_ratio < 1.15:  # Reduced from 1.10
            return 0.5  # Healthy
        elif 0.85 <= cost_ratio < 1.05:  # Adjusted range
            return 0.0  # Fair value
        elif 0.75 <= cost_ratio < 0.85:  # Reduced from 0.80
            return -0.5  # Stress
        else:  # cost_ratio < 0.75 (reduced from 0.80)
            return -1.0  # Distress
    
    def smooth_prodcost_signal(self, raw_signal: float) -> float:
        """
        Smooth production cost signal using SMA (7-day).
        
        Args:
            raw_signal: Raw production cost signal
            
        Returns:
            float: Smoothed production cost signal
        """
        self.prodcost_raw_history.append(raw_signal)
        
        if len(self.prodcost_raw_history) < self.prodcost_window:
            return raw_signal
        
        return float(np.mean(list(self.prodcost_raw_history)))
    
    # ========== VOLATILITY REGIME ==========
    
    def calculate_volatility_regime(self, returns: np.ndarray) -> Tuple[str, float]:
        """
        Calculate volatility regime (LOW/NORMAL/HIGH).
        
        Args:
            returns: Array of daily returns (e.g., last 30 days)
            
        Returns:
            Tuple: (regime_name, annualized_vol)
        """
        if len(returns) < 2:
            return "NORMAL", 0.0
        
        daily_vol = np.std(returns)
        annual_vol = daily_vol * np.sqrt(252)  # Annualize
        
        if annual_vol < 0.50:
            regime = "LOW"
        elif annual_vol < 0.80:
            regime = "NORMAL"
        else:
            regime = "HIGH"
        
        return regime, annual_vol
    
    # ========== ENSEMBLE SCORE ==========
    
    def calculate_score_raw(self, trend_smooth: float, momentum_smooth: float,
                           prodcost_smooth: float,
                           w_trend: float = 1.2, w_momentum: float = 1.2,
                           w_prodcost: float = 1.3) -> float:
        """
        Calculate raw ensemble score.
        
        REBALANCED: Less emphasis on prodcost, more on trend/momentum for better bull market performance.
        
        Args:
            trend_smooth: Smoothed trend signal
            momentum_smooth: Smoothed momentum signal
            prodcost_smooth: Smoothed production cost signal
            w_trend: Weight for trend (1.2 - increased)
            w_momentum: Weight for momentum (1.2 - increased)
            w_prodcost: Weight for prodcost (1.3 - reduced from 1.5)
            
        Returns:
            float: Raw score [-3.7, +3.7]
        """
        score = (w_trend * trend_smooth + 
                w_momentum * momentum_smooth + 
                w_prodcost * prodcost_smooth)
        
        return score
    
    def smooth_score_ema(self, raw_score: float) -> float:
        """
        Smooth score using EMA (α=0.3, double smoothing).
        
        Args:
            raw_score: Raw ensemble score
            
        Returns:
            float: Smoothed score
        """
        if self.score_ema is None:
            self.score_ema = raw_score
        else:
            self.score_ema = (self.score_alpha * raw_score + 
                            (1 - self.score_alpha) * self.score_ema)
        
        return self.score_ema
    
    def adjust_score_for_volatility(self, score: float, vol_regime: str) -> float:
        """
        Adjust score based on volatility regime.
        Less dampening for more aggressive trading.
        
        Args:
            score: Smoothed score
            vol_regime: "LOW", "NORMAL", or "HIGH"
            
        Returns:
            float: Adjusted score
        """
        if vol_regime == "LOW":
            return score
        elif vol_regime == "NORMAL":
            return 0.95 * score  # Reduced dampening from 0.8 to 0.95
        elif vol_regime == "HIGH":
            return 0.85 * score  # Reduced dampening from 0.6 to 0.85
        else:
            return score
    
    def check_extreme_production_cost(self, cost_ratio: float, 
                                      recent_ratios: list) -> Tuple[str, float]:
        """
        Check if production cost ratio is extreme (Z-score).
        
        Args:
            cost_ratio: Current cost ratio
            recent_ratios: List of recent cost ratios (e.g., 90 days)
            
        Returns:
            Tuple: (flag, z_score)
        """
        if len(recent_ratios) < 10:
            return "NORMAL", 0.0
        
        mu = np.mean(recent_ratios)
        sigma = np.std(recent_ratios)
        
        if sigma == 0:
            return "NORMAL", 0.0
        
        z_score = (cost_ratio - mu) / sigma
        
        if z_score > 2.0:
            return "EXTREME_BULLISH", z_score
        elif z_score < -2.0:
            return "EXTREME_BEARISH", z_score
        else:
            return "NORMAL", z_score
    
    def apply_extreme_adjustment(self, target_btc: float, extreme_flag: str) -> float:
        """
        Apply adjustment for extreme production cost conditions.
        
        Args:
            target_btc: Target BTC allocation
            extreme_flag: "EXTREME_BULLISH", "EXTREME_BEARISH", or "NORMAL"
            
        Returns:
            float: Adjusted target
        """
        if extreme_flag in ["EXTREME_BULLISH", "EXTREME_BEARISH"]:
            return target_btc * 0.8  # Reduce by 20%
        return target_btc
    
    # ========== TARGET ALLOCATION ==========
    
    def map_score_to_target(self, score_adjusted: float) -> float:
        """Map adjusted score to target BTC allocation.
        
        ASYMMETRIC BULL/BEAR: BTC is the default long-term asset.
        - Bull market (positive score): Stay HIGH in BTC (70-100%)
        - Neutral (small neg): Still hold majority BTC (50-70%)
        - Bear market (large neg): Reduce to protection levels (5-30%)
        
        Philosophy: "Ride the bull, flee the bear"
        
        Args:
            score_adjusted: Adjusted ensemble score
            
        Returns:
            float: Target BTC allocation [0.05, 1.0]
        """
        # BULL ZONE - Easy to stay high
        if score_adjusted >= 0.5:
            return 1.0  # Full BTC in bull
        elif score_adjusted >= 0.0:
            return 0.85  # High BTC in mild bull
        # NEUTRAL ZONE - Still favor BTC (it's the growth asset)
        elif score_adjusted >= -0.3:
            return 0.70  # Majority BTC in neutral
        elif score_adjusted >= -0.7:
            return 0.50  # Balanced in slight bear
        # BEAR ZONE - Protect capital
        elif score_adjusted >= -1.3:
            return 0.30  # Reduce in bear
        elif score_adjusted >= -2.0:
            return 0.15  # Low in strong bear
        else:  # score_adjusted < -2.0
            return 0.05  # Minimal in extreme bear
    
    # ========== LOGGING ==========
    
    def log_signal_calculation(self, date, signals_dict: Dict) -> None:
        """Log signal calculation details."""
        log_entry = {
            'date': date,
            **signals_dict
        }
        self.signal_history.append(log_entry)
    
    def get_signal_history_df(self) -> pd.DataFrame:
        """Get signal history as DataFrame."""
        if not self.signal_history:
            return pd.DataFrame()
        return pd.DataFrame(self.signal_history)


# ========== UTILITY FUNCTIONS ==========

def calculate_sma_200(prices: np.ndarray) -> float:
    """Calculate 200-day SMA."""
    if len(prices) < 200:
        return np.mean(prices)
    return np.mean(prices[-200:])


def calculate_price_60d_ago(prices: np.ndarray) -> Optional[float]:
    """Get price from 60 days ago."""
    if len(prices) < 61:
        return None
    return prices[-61]


def get_recent_daily_returns(prices: np.ndarray, days: int = 30) -> np.ndarray:
    """Calculate recent daily returns."""
    if len(prices) < 2:
        return np.array([])
    
    prices = np.asarray(prices)
    recent = prices[-days:] if len(prices) >= days else prices
    returns = np.diff(recent) / recent[:-1]
    
    return returns
