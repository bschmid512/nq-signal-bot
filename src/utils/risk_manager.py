from __future__ import annotations
import pandas as pd
import numpy as np
from datetime import datetime, date
from typing import List, Dict, Optional
from dataclasses import dataclass
from loguru import logger

from src.core.engine import Signal
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
    """Global risk management and position sizing"""
    
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
        
        logger.info(f"RiskManager initialized - Max daily loss: ${self.max_daily_loss}")
    
    def filter_signals(self, signals: List[Signal]) -> List[Signal]:
        """Filter signals based on risk management rules"""
        filtered_signals = []
        
        # Check if we should trade today
        if not self._should_trade_today():
            logger.info("Trading paused due to risk management rules")
            return filtered_signals
        
        for signal in signals:
            if self._is_signal_allowed(signal):
                filtered_signals.append(signal)
            else:
                logger.info(f"Signal filtered by risk management: {signal.strategy} {signal.signal_type}")
        
        return filtered_signals
    
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
            return False
        
        # Check minimum confidence
        if signal.confidence < 0.4:  # Minimum 40% confidence
            return False
        
        # Check minimum risk/reward
        if signal.risk_reward < 1.0:  # Minimum 1:1 risk/reward
            return False
        
        # Check ATR (avoid dead tape)
        if signal.atr < self.min_atr:
            return False
        
        # Check for chop conditions
        if self._is_chop_condition(signal):
            return False
        
        return True
    
    def _is_chop_condition(self, signal: Signal) -> bool:
        """Check if market is in chop conditions"""
        # Low ATR indicates potential chop
        if signal.atr < 5.0:
            return True
        
        # Check if signal is from ORB strategy (allowed in chop)
        if signal.strategy == 'orb':
            return False
        
        return False
    
    def update_signal_result(self, signal: Signal, pnl: float, r_multiple: float):
        """Update risk metrics with signal result"""
        try:
            self.risk_metrics.total_trades += 1
            self.daily_pnl += pnl
            
            if pnl > 0:
                self.risk_metrics.winning_trades += 1
                self.risk_metrics.consecutive_losses = 0
            else:
                self.risk_metrics.losing_trades += 1
                self.risk_metrics.consecutive_losses += 1
            
            # Update max drawdown
            if self.daily_pnl < 0:
                current_drawdown = abs(self.daily_pnl) / self.max_daily_loss
                self.risk_metrics.max_drawdown = max(self.risk_metrics.max_drawdown, current_drawdown)
            
            logger.info(f"Signal result: {signal.strategy} {signal.signal_type} - "
                       f"PNL: ${pnl:.2f}, R: {r_multiple:.2f}, "
                       f"Daily PNL: ${self.daily_pnl:.2f}")
            
        except Exception as e:
            logger.error(f"Error updating signal result: {e}")
    
    def calculate_position_size(self, signal: Signal, account_balance: float) -> float:
        """Calculate position size based on risk"""
        try:
            # Risk per trade (1% of account balance)
            risk_per_trade = account_balance * 0.01
            
            # Calculate dollar risk per share
            if signal.signal_type == 'long':
                dollar_risk = signal.entry_price - signal.stop_loss
            else:  # short
                dollar_risk = signal.stop_loss - signal.entry_price
            
            if dollar_risk <= 0:
                return 0
            
            # Calculate position size
            position_size = risk_per_trade / dollar_risk
            
            # Apply confidence adjustment
            position_size *= signal.confidence
            
            # Apply maximum position limit (10% of account)
            max_position_value = account_balance * 0.10
            max_shares = max_position_value / signal.entry_price
            
            position_size = min(position_size, max_shares)
            
            return int(position_size)
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0
    
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
            'trading_allowed': self._should_trade_today()
        }
    
    def reset_daily_metrics(self):
        """Reset daily metrics (call at market open)"""
        self.today = date.today()
        self.daily_pnl = 0.0
        self.signals_today = []
        logger.info("Daily risk metrics reset")
    
    def add_position(self):
        """Increment position count"""
        self.risk_metrics.current_positions += 1
    
    def remove_position(self):
        """Decrement position count"""
        self.risk_metrics.current_positions = max(0, self.risk_metrics.current_positions - 1)

class MarketEnvironment:
    """Market environment assessment"""
    
    def __init__(self):
        self.vix_threshold = 20.0
        self.volume_threshold = 100000  # Minimum volume
    
    def assess_market_conditions(self, df: pd.DataFrame) -> Dict:
        """Assess current market conditions"""
        try:
            if len(df) < 20:
                return {'chop_guard': True, 'low_volatility': True}
            
            current_bar = df.iloc[-1]
            recent_bars = df.tail(20)
            
            # Check volatility (ATR)
            avg_atr = recent_bars['atr'].mean()
            chop_guard = avg_atr < 5.0
            
            # Check volume
            avg_volume = recent_bars['volume'].mean()
            low_volume = avg_volume < self.volume_threshold
            
            # Check range (Bollinger Band width)
            if 'bb_width' in recent_bars.columns:
                avg_bb_width = recent_bars['bb_width'].mean()
                current_price = current_bar['close']
                low_volatility = (avg_bb_width / current_price) < 0.004  # 0.4% of price
            else:
                low_volatility = False
            
            return {
                'chop_guard': chop_guard,
                'low_volatility': low_volatility,
                'low_volume': low_volume,
                'avg_atr': avg_atr,
                'avg_volume': avg_volume
            }
            
        except Exception as e:
            logger.error(f"Error assessing market conditions: {e}")
            return {'chop_guard': True, 'low_volatility': True}