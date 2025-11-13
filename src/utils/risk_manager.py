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
    """Risk management metrics"""
    daily_pnl: float = 0.0
    max_drawdown: float = 0.0
    consecutive_losses: int = 0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    current_positions: int = 0

class RiskManager:
    """Global risk management and position sizing for NQ futures"""
    
    def __init__(self):
        self.risk_metrics = RiskMetrics()
        self.daily_pnl = 0.0
        self.max_daily_loss = config.MAX_DAILY_LOSS_DOLLAR
        self.max_daily_loss_r = config.MAX_DAILY_LOSS_R
        self.max_positions = config.MAX_CONCURRENT_POSITIONS
        self.min_atr = config.MIN_ATR_FOR_TRADING
        
        # Track daily performance
        self.today = date.today()
        self.signals_today = []
        
        # Session filters (NQ-specific)
        self.session_start = time(9, 30)
        self.lunch_start = time(11, 30)  # Avoid lunch chop
        self.lunch_end = time(13, 30)
        self.last_hour = time(15, 45)    # Skip last 15 min
        
        logger.info(f"RiskManager initialized - Max positions: {self.max_positions}, Max daily loss: ${self.max_daily_loss}")
    
    def filter_signals(self, signals: List[Signal]) -> List[Signal]:
        """Filter signals based on risk management rules"""
        filtered_signals = []
        
        # Check if we should trade today
        if not self._should_trade_today():
            logger.info("Trading paused due to risk management rules")
            return filtered_signals
        
        # Check session time (NQ-specific)
        current_time = datetime.now().time()
        if self.lunch_start <= current_time <= self.lunch_end:
            logger.info("Lunch chop period - skipping signals")
            return filtered_signals
        
        if current_time >= self.last_hour:
            logger.info("End of day - skipping signals")
            return filtered_signals
        
        # Filter correlated signals (similar entries)
        signals = self._filter_correlated_signals(signals)
        
        for signal in signals:
            if self._is_signal_allowed(signal):
                filtered_signals.append(signal)
            else:
                logger.info(f"Signal filtered by risk management: {signal.strategy} {signal.signal_type}")
        
        return filtered_signals
    
    def _filter_correlated_signals(self, signals: List[Signal]) -> List[Signal]:
        """Remove signals that are too similar (same direction, close entries)"""
        if len(signals) <= 1:
            return signals
        
        # Group by direction
        longs = [s for s in signals if s.signal_type == 'long']
        shorts = [s for s in signals if s.signal_type == 'short']
        
        filtered = []
        
        # For each direction, pick highest confidence and filter by entry distance
        if longs:
            # Sort by confidence
            longs_sorted = sorted(longs, key=lambda s: s.confidence, reverse=True)
            filtered.append(longs_sorted[0])  # Take best
            
            # Add others if entries are > 5 ATR apart
            for sig in longs_sorted[1:]:
                if abs(sig.entry_price - longs_sorted[0].entry_price) > (5 * sig.atr):
                    filtered.append(sig)
        
        if shorts:
            shorts_sorted = sorted(shorts, key=lambda s: s.confidence, reverse=True)
            filtered.append(shorts_sorted[0])
            
            for sig in shorts_sorted[1:]:
                if abs(sig.entry_price - shorts_sorted[0].entry_price) > (5 * sig.atr):
                    filtered.append(sig)
        
        return filtered
    
    def _should_trade_today(self) -> bool:
        """Check if we should trade based on daily risk limits"""
        # Check daily loss limit
        if self.daily_pnl <= -self.max_daily_loss:
            logger.warning(f"Daily loss limit reached: ${self.daily_pnl:.2f}")
            return False
        
        # Check consecutive losses
        if self.risk_metrics.consecutive_losses >= 5:  # Pause after 5 consecutive losses
            logger.warning(f"Too many consecutive losses: {self.risk_metrics.consecutive_losses}")
            return False
        
        # Check max drawdown
        if self.risk_metrics.max_drawdown > 0.15:  # 15% drawdown
            logger.warning(f"Max drawdown exceeded: {self.risk_metrics.max_drawdown:.2%}")
            return False
        
        return True
    
    def _is_signal_allowed(self, signal: Signal) -> bool:
        """Check if individual signal meets risk criteria"""
        # Check position limit
        if self.risk_metrics.current_positions >= self.max_positions:
            logger.warning(f"Position limit reached: {self.risk_metrics.current_positions}/{self.max_positions}")
            return False
        
        # Check minimum confidence
        if signal.confidence < 0.4:  # Minimum 40% confidence
            logger.info(f"Signal confidence too low: {signal.confidence:.2f}")
            return False
        
        # Check minimum risk/reward
        if signal.risk_reward < 1.0:  # Minimum 1:1 risk/reward
            logger.info(f"Risk/reward too low: {signal.risk_reward:.2f}")
            return False
        
        # Check ATR (avoid dead tape)
        if signal.atr < self.min_atr:
            logger.info(f"ATR too low: {signal.atr:.2f}")
            return False
        
        # Check for chop conditions
        if self._is_chop_condition(signal):
            logger.info(f"Chop condition detected for {signal.strategy}")
            return False
        
        # Strategy-specific filters
        if signal.strategy == 'orb':
            # Only allow ORB during open drive
            current_time = datetime.now().time()
            if not (self.session_start <= current_time <= time(10, 30)):
                return False
        
        return True
    
    def _is_chop_condition(self, signal: Signal) -> bool:
        """Enhanced chop detection for NQ futures"""
        # Ultra-low ATR
        if signal.atr < 8.0:
            return True
        
        # Ultra-low volume
        if signal.volume < 500:
            return True
        
        # Tight range (from metadata if available)
        if signal.metadata and 'bb_width' in signal.metadata:
            bb_width_pct = signal.metadata['bb_width'] / signal.entry_price
            if bb_width_pct < 0.003:  # Less than 0.3% range
                return True
        
        # Check for extended consolidation (ADX < 15)
        if hasattr(signal, 'metadata') and signal.metadata.get('adx', 0) < 15:
            return True
        
        return False
    
    def update_signal_result(self, signal: Signal, pnl: float, r_multiple: float):
        """Update risk metrics with signal result"""
        try:
            self.risk_metrics.total_trades += 1
            self.daily_pnl += pnl
            
            if pnl > 0:
                self.risk_metrics.winning_trades += 1
                self.risk_metrics.consecutive_losses = 0
                logger.info(f"WIN: {signal.strategy} {signal.signal_type} - PNL: ${pnl:.2f}, R: {r_multiple:.2f}")
            else:
                self.risk_metrics.losing_trades += 1
                self.risk_metrics.consecutive_losses += 1
                logger.info(f"LOSS: {signal.strategy} {signal.signal_type} - PNL: ${pnl:.2f}, R: {r_multiple:.2f}")
            
            # Update max drawdown
            if self.daily_pnl < 0:
                current_drawdown = abs(self.daily_pnl) / self.max_daily_loss
                self.risk_metrics.max_drawdown = max(self.risk_metrics.max_drawdown, current_drawdown)
            
        except Exception as e:
            logger.error(f"Error updating signal result: {e}")
    
    def calculate_position_size(self, signal: Signal, account_balance: float) -> int:
        """Volatility-adjusted position sizing for NQ Micro futures"""
        try:
            # Base risk: 1% of account
            dollar_risk = account_balance * 0.01
            
            # Confidence adjustment
            dollar_risk *= signal.confidence
            
            # Volatility regime adjustment
            if signal.atr > 20:  # High volatility
                dollar_risk *= 0.7  # Reduce size
                logger.info(f"High volatility detected (ATR: {signal.atr:.2f}) - reducing position size")
            elif signal.atr < 10:  # Low volatility (potential chop)
                dollar_risk *= 0.5
                logger.info(f"Low volatility detected (ATR: {signal.atr:.2f}) - reducing position size")
            
            # Calculate contracts (Micro NQ = $2 per per point per contract)
            # Dollar risk per contract = (stop distance in points) * $2
            risk_per_contract = abs(signal.entry_price - signal.stop_loss) * 2
            
            if risk_per_contract <= 0:
                return 0
            
            contracts = dollar_risk / risk_per_contract
            
            # Maximum position: 10% of account or 3 contracts
            max_position_value = account_balance * 0.10
            max_contracts = max_position_value / (signal.entry_price * 2)
            
            position_size = int(min(contracts, max_contracts, 3))  # Cap at 3 contracts
            
            logger.info(f"Position size: {position_size} contracts (Risk: ${dollar_risk:.2f})")
            
            return position_size
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 1  # Default to 1 contract on error
    
    def get_risk_summary(self) -> Dict:
        """Get current risk summary"""
        win_rate = 0.0
        if self.risk_metrics.total_trades > 0:
            win_rate = self.risk_metrics.winning_trades / self.risk_metrics.total_trades
        
        return {
            'daily_pnl': self.daily_pnl,
            'total_trades': self.risk_metrics.total_trades,
            'winning_trades': self.risk_metrics.winning_trades,
            'losing_trades': self.risk_metrics.losing_trades,
            'win_rate': win_rate,
            'consecutive_losses': self.risk_metrics.consecutive_losses,
            'max_drawdown': self.risk_metrics.max_drawdown,
            'current_positions': self.risk_metrics.current_positions,
            'trading_allowed': self._should_trade_today(),
            'positions_remaining': self.max_positions - self.risk_metrics.current_positions
        }
    
    def reset_daily_metrics(self):
        """Reset daily metrics (call at market open)"""
        self.today = date.today()
        self.daily_pnl = 0.0
        self.signals_today = []
        self.risk_metrics.consecutive_losses = 0
        logger.info("Daily risk metrics reset")
    
    def add_position(self):
        """Increment position count"""
        self.risk_metrics.current_positions += 1
    
    def remove_position(self):
        """Decrement position count"""
        self.risk_metrics.current_positions = max(0, self.risk_metrics.current_positions - 1)