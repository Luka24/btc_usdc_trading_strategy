"""
Confirmation Layer Module
==========================
Multi-layer confirmation system for trade execution.
Prevents whipsaw and implements professional confirmation rules.

Professional trading rulebook implementation.
"""

from typing import Tuple, Optional
from enum import Enum


class Zone(Enum):
    """Market zone based on target allocation."""
    BULLISH = "BULLISH"      # target >= 0.6
    NEUTRAL = "NEUTRAL"      # 0.3 <= target < 0.6
    BEARISH = "BEARISH"      # target < 0.3


class Direction(Enum):
    """Direction of movement."""
    UP = "UP"                 # target > current + 0.15
    DOWN = "DOWN"             # target < current - 0.15
    HOLD = "HOLD"             # no meaningful change


class ConfirmationLayer:
    """Multi-layer confirmation system."""
    
    def __init__(self,
                 zone_consistency_days: int = 2,
                 magnitude_threshold: float = 0.50,
                 direction_threshold: float = 0.15,
                 extreme_score_threshold: float = 2.5,
                 min_rebalance_threshold: float = 0.15):
        """
        Initialize confirmation layer.
        
        Args:
            zone_consistency_days: Days for zone consistency
            magnitude_threshold: 50% for magnitude override
            direction_threshold: 15% for directional check
            extreme_score_threshold: ±2.5 for extreme events
            min_rebalance_threshold: 15% minimum trade size
        """
        self.zone_consistency_days = zone_consistency_days
        self.magnitude_threshold = magnitude_threshold
        self.direction_threshold = direction_threshold
        self.extreme_score_threshold = extreme_score_threshold
        self.min_rebalance_threshold = min_rebalance_threshold
        
        # History
        self.target_history = []
        self.score_history = []
    
    # ========== ZONE-BASED CONFIRMATION ==========
    
    @staticmethod
    def get_zone(target: float) -> Zone:
        """
        Get zone from target allocation.
        
        Args:
            target: Target BTC allocation (0-1)
            
        Returns:
            Zone: BULLISH, NEUTRAL, or BEARISH
        """
        if target >= 0.6:
            return Zone.BULLISH
        elif target >= 0.3:
            return Zone.NEUTRAL
        else:
            return Zone.BEARISH
    
    def check_zone_consistency(self, target_today: float, 
                              target_yesterday: float) -> Tuple[bool, str]:
        """
        Check if zone is consistent between days.
        
        Rule: Zone(today) == Zone(yesterday)
        
        Args:
            target_today: Target for today
            target_yesterday: Target for yesterday
            
        Returns:
            Tuple: (is_valid, reason)
        """
        zone_today = self.get_zone(target_today)
        zone_yesterday = self.get_zone(target_yesterday)
        
        if zone_today == zone_yesterday:
            return True, f"Zone match: {zone_today.value} for 2 days"
        else:
            return False, f"Zone mismatch: {zone_yesterday.value} → {zone_today.value}"
    
    # ========== MAGNITUDE OVERRIDE ==========
    
    def check_magnitude_override(self, target_today: float, current_position: float,
                                score_today: float, score_yesterday: float) -> Tuple[bool, str]:
        """
        Check for large change override (emergency execution).
        
        Rule: If |Target - Current| >= 50% AND score is directionally consistent
        
        Args:
            target_today: Target for today
            current_position: Current position
            score_today: Today's score
            score_yesterday: Yesterday's score
            
        Returns:
            Tuple: (is_valid, reason)
        """
        change = abs(target_today - current_position)
        
        # Check magnitude
        if change < self.magnitude_threshold:
            return False, f"Change {change:.1%} < {self.magnitude_threshold:.0%}"
        
        # Check score consistency
        score_consistent = (
            (score_today > 0 and score_yesterday > 0) or
            (score_today < 0 and score_yesterday < 0)
        )
        
        if not score_consistent:
            return False, f"Score flip: {score_yesterday:.2f} → {score_today:.2f}"
        
        return True, f"Large change override: {change:.1%}, score consistent"
    
    # ========== DIRECTIONAL CONFIRMATION ==========
    
    @staticmethod
    def get_direction(target: float, current: float, 
                     threshold: float = 0.15) -> Direction:
        """
        Get direction of movement.
        
        Args:
            target: Target allocation
            current: Current position
            threshold: Threshold for meaningful change
            
        Returns:
            Direction: UP, DOWN, or HOLD
        """
        if target > current + threshold:
            return Direction.UP
        elif target < current - threshold:
            return Direction.DOWN
        else:
            return Direction.HOLD
    
    def check_directional_match(self, target_today: float, target_yesterday: float,
                               current_position: float) -> Tuple[bool, str]:
        """
        Check if direction is consistent.
        
        Rule: Direction(today) == Direction(yesterday) AND Direction != HOLD
        
        Args:
            target_today: Target for today
            target_yesterday: Target for yesterday
            current_position: Current position
            
        Returns:
            Tuple: (is_valid, reason)
        """
        direction_today = self.get_direction(target_today, current_position, 
                                            self.direction_threshold)
        direction_yesterday = self.get_direction(target_yesterday, current_position,
                                               self.direction_threshold)
        
        if (direction_today == direction_yesterday and 
            direction_today != Direction.HOLD):
            return True, f"Directional match: {direction_today.value} for 2 days"
        else:
            return False, f"Direction mismatch or no move: {direction_yesterday.value} → {direction_today.value}"
    
    # ========== EXTREME EVENTS ==========
    
    def check_extreme_events(self, score_today: float, current_position: float) -> Tuple[bool, str]:
        """
        Check for extreme score events (emergency exit/entry).
        
        Rules:
        - Score <= -2.5 AND Current > 0.3 → Emergency exit
        - Score >= +2.5 AND Current < 0.7 → Emergency entry
        
        Args:
            score_today: Today's score
            current_position: Current BTC position
            
        Returns:
            Tuple: (is_valid, reason)
        """
        extreme_bearish = (score_today <= -self.extreme_score_threshold and 
                          current_position > 0.3)
        extreme_bullish = (score_today >= self.extreme_score_threshold and 
                          current_position < 0.7)
        
        if extreme_bearish:
            return True, f"EXTREME BEARISH: score {score_today:.2f}, emergency exit"
        
        if extreme_bullish:
            return True, f"EXTREME BULLISH: score {score_today:.2f}, emergency entry"
        
        return False, f"No extreme event detected"
    
    # ========== MINIMUM REBALANCING THRESHOLD ==========
    
    def check_minimum_threshold(self, target_today: float, 
                               current_position: float) -> Tuple[bool, str]:
        """
        Check if change meets minimum rebalancing threshold.
        
        Rule: |Target - Current| >= 15%
        
        Args:
            target_today: Target allocation
            current_position: Current position
            
        Returns:
            Tuple: (is_valid, reason)
        """
        change = abs(target_today - current_position)
        
        if change >= self.min_rebalance_threshold:
            return True, f"Threshold passed: {change:.1%} >= {self.min_rebalance_threshold:.0%}"
        else:
            return False, f"Too small: {change:.1%} < {self.min_rebalance_threshold:.0%}"
    
    # ========== MAIN CONFIRMATION LOGIC ==========
    
    def check_confirmation(self, current_position: float, target_today: float,
                          target_yesterday: float, score_today: float,
                          score_yesterday: float) -> Tuple[bool, str, float]:
        """
        Multi-layer confirmation check.
        
        Layers (in order):
        1. Zone-based (primary)
        2. Magnitude override (emergency)
        3. Directional match (secondary)
        4. Extreme events (emergency exit/entry)
        
        Args:
            current_position: Current BTC position
            target_today: Target for today
            target_yesterday: Target for yesterday
            score_today: Today's score
            score_yesterday: Yesterday's score
            
        Returns:
            Tuple: (is_valid, reason, confidence)
        """
        
        # Layer 1: Zone-based confirmation
        zone_valid, zone_reason = self.check_zone_consistency(target_today, 
                                                             target_yesterday)
        if zone_valid:
            # But check minimum threshold
            min_valid, min_reason = self.check_minimum_threshold(target_today, 
                                                                current_position)
            if not min_valid:
                return False, min_reason, 0.0
            return True, zone_reason, 0.85
        
        # Layer 2: Magnitude override
        mag_valid, mag_reason = self.check_magnitude_override(target_today, 
                                                             current_position,
                                                             score_today, 
                                                             score_yesterday)
        if mag_valid:
            return True, mag_reason, 0.90
        
        # Layer 3: Directional match
        dir_valid, dir_reason = self.check_directional_match(target_today, 
                                                            target_yesterday,
                                                            current_position)
        if dir_valid:
            # Check minimum threshold
            min_valid, min_reason = self.check_minimum_threshold(target_today, 
                                                                current_position)
            if not min_valid:
                return False, min_reason, 0.0
            return True, dir_reason, 0.70
        
        # Layer 4: Extreme events (override all)
        extreme_valid, extreme_reason = self.check_extreme_events(score_today, 
                                                                 current_position)
        if extreme_valid:
            return True, extreme_reason, 0.95
        
        # No confirmation
        return False, "No confirmation: zone/direction/magnitude mismatch", 0.0
    
    # ========== HISTORY TRACKING ==========
    
    def add_target_to_history(self, date, target: float, score: float) -> None:
        """Track target and score history."""
        self.target_history.append({
            'date': date,
            'target': target,
            'score': score
        })
        self.score_history.append(score)
    
    def get_target_history(self):
        """Get target history."""
        return self.target_history
    
    def get_score_history(self):
        """Get score history."""
        return self.score_history
