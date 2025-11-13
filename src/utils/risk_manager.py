from __future__ import annotations
import pandas as pd
import numpy as np
from datetime import datetime, date, time
from typing import List, Dict, Optional
from dataclasses import dataclass
from loguru import logger

from src.core.models import Signal
from config.config import config

@dataclass
class RiskMetrics:
    """Simplified risk metrics for tracking"""
    total_trades: int = 0
    winning_trades: int = 0
    current_positions: int = 0
    last_signal_price: float = 0

class RiskManager:
    """SIMPLIFIED risk management focused on signal QUALITY not quantity"""
    
    def __init__(self):
        self.risk_metrics = RiskMetrics()
        self.max_positions = 3  # Allow more positions
        self.min_signal_separation = 10  # Points between signals
        
        # Track recent signals to avoid clustering
        self.recent_long_entry = None
        self.recent_short_entry = None
        
        logger.info("🎯 RiskManager initialized - QUALITY FOCUSED mode")
    
    def filter_signals(self, signals: List[Signal]) -> List[Signal]:
        """Filter signals for QUALITY not arbitrary restrictions"""
        filtered_signals = []
        
        for signal in signals:
            if self._is_quality_signal(signal):
                filtered_signals.append(signal)
                self._update_recent_entries(signal)
            else:
                logger.debug(f"Signal filtered: {signal.strategy} {signal.signal_type} conf={signal.confidence:.2f}")
        
        if filtered_signals:
            logger.info(f"✅ Passed {len(filtered_signals)}/{len(signals)} signals through quality filter")
        
        return filtered_signals
    
    def _is_quality_signal(self, signal: Signal) -> bool:
        """Check if signal meets QUALITY criteria"""
        
        # 1. Minimum confidence threshold (higher bar)
        if signal.confidence < 0.60:
            logger.debug(f"Low confidence: {signal.confidence:.2f}")
            return False
        
        # 2. Minimum risk/reward
        if signal.risk_reward < 1.5:
            logger.debug(f"Poor risk/reward: {signal.risk_reward:.2f}")
            return False
        
        # 3. Reasonable stop loss (not too tight, not too wide)
        stop_distance = abs(signal.entry_price - signal.stop_loss)
        if stop_distance < 5:  # Too tight for NQ
            logger.debug(f"Stop too tight: {stop_distance:.2f} points")
            return False
        if stop_distance > 50:  # Too wide
            logger.debug(f"Stop too wide: {stop_distance:.2f} points")
            return False
        
        # 4. Avoid signal clustering (same direction too close)
        if signal.signal_type == 'long' and self.recent_long_entry:
            if abs(signal.entry_price - self.recent_long_entry) < self.min_signal_separation:
                logger.debug(f"Long signal too close to recent: {abs(signal.entry_price - self.recent_long_entry):.2f} points")
                return False
        
        if signal.signal_type == 'short' and self.recent_short_entry:
            if abs(signal.entry_price - self.recent_short_entry) < self.min_signal_separation:
                logger.debug(f"Short signal too close to recent: {abs(signal.entry_price - self.recent_short_entry):.2f} points")
                return False
        
        # 5. Check volatility is reasonable (not dead market)
        if signal.atr < 5.0:  # NQ needs some movement
            logger.debug(f"Market too quiet: ATR={signal.atr:.2f}")
            return False
        
        # 6. Volume check if available
        if signal.volume and signal.volume < 100:  # Extremely low volume
            logger.debug(f"Volume too low: {signal.volume}")
            return False
        
        # Signal passes all quality checks
        logger.info(f"✅ QUALITY SIGNAL: {signal.strategy} {signal.signal_type} @ {signal.entry_price:.2f} (Conf: {signal.confidence:.2f}, RR: {signal.risk_reward:.1f})")
        return True
    
    def _update_recent_entries(self, signal: Signal):
        """Track recent entries to avoid clustering"""
        if signal.signal_type == 'long':
            self.recent_long_entry = signal.entry_price
        else:
            self.recent_short_entry = signal.entry_price
    
    def calculate_position_size(self, signal: Signal, account_balance: float) -> int:
        """Calculate position size based on signal confidence"""
        try:
            # Base risk: 1-2% of account based on confidence
            base_risk_pct = 0.01 if signal.confidence < 0.7 else 0.015 if signal.confidence < 0.8 else 0.02
            dollar_risk = account_balance * base_risk_pct
            
            # Adjust for signal confidence
            dollar_risk *= signal.confidence
            
            # Calculate contracts for Micro NQ ($2 per point)
            stop_distance_points = abs(signal.entry_price - signal.stop_loss)
            risk_per_contract = stop_distance_points * 2  # $2 per point for MNQ
            
            if risk_per_contract <= 0:
                return 1
            
            contracts = dollar_risk / risk_per_contract
            
            # Position sizing based on confidence
            if signal.confidence >= 0.85:
                max_contracts = 3
            elif signal.confidence >= 0.75:
                max_contracts = 2
            else:
                max_contracts = 1
            
            position_size = int(min(contracts, max_contracts))
            position_size = max(position_size, 1)  # At least 1 contract
            
            logger.info(f"Position size: {position_size} contracts (Confidence: {signal.confidence:.2f}, Risk: ${dollar_risk:.2f})")
            
            return position_size
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 1
    
    def get_risk_summary(self) -> Dict:
        """Get simplified risk summary"""
        win_rate = 0.0
        if self.risk_metrics.total_trades > 0:
            win_rate = self.risk_metrics.winning_trades / self.risk_metrics.total_trades
        
        return {
            'total_trades': self.risk_metrics.total_trades,
            'win_rate': win_rate,
            'current_positions': self.risk_metrics.current_positions,
            'max_positions': self.max_positions,
            'trading_enabled': True  # Always enabled in quality mode
        }
    
    def update_signal_result(self, signal: Signal, pnl: float, r_multiple: float):
        """Update metrics with signal result"""
        self.risk_metrics.total_trades += 1
        
        if pnl > 0:
            self.risk_metrics.winning_trades += 1
            logger.info(f"✅ WIN: {signal.strategy} - PNL: ${pnl:.2f}, R: {r_multiple:.2f}")
        else:
            logger.info(f"❌ LOSS: {signal.strategy} - PNL: ${pnl:.2f}, R: {r_multiple:.2f}")
    
    def reset_daily_metrics(self):
        """Reset metrics (simplified)"""
        self.recent_long_entry = None
        self.recent_short_entry = None
        logger.info("Risk metrics reset")