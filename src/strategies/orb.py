import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta
from typing import List, Dict
from dataclasses import dataclass
from loguru import logger

from src.core.engine import Signal

@dataclass
class ORBConfig:
    """Configuration for Opening Range Breakout strategy"""
    base_confidence: float = 0.7
    range_minutes: int = 15
    buffer_atr_factor: float = 0.15
    min_buffer_points: float = 3.0
    volume_confirm: float = 1.5
    max_gap_atr: float = 0.7
    alternative_range: int = 5

class ORBStrategy:
    """Opening Range Breakout Strategy"""
    
    def __init__(self, config: Dict):
        self.config = ORBConfig(**config)
        self.last_signal_time = {}
        self.orb_levels = {}  # Store ORB levels by date
        self.session_start = time(9, 30)
        self.session_end = time(16, 0)
    
    def generate_signals(self, df: pd.DataFrame, symbol: str) -> List[Signal]:
        """Generate Opening Range Breakout signals"""
        signals = []
        
        try:
            if len(df) < 30:
                return signals
            
            current_bar = df.iloc[-1]
            current_time = self._get_bar_time(current_bar)
            
            # Only trade during market hours
            if not (self.session_start <= current_time <= self.session_end):
                return signals
            
            # Get or calculate ORB levels
            orb_levels = self._get_orb_levels(df, symbol)
            
            if not orb_levels:
                return signals
            
            # Check for breakout
            signals.extend(self._check_breakout(df, current_bar, orb_levels, symbol))
            
            # Check for ORB retake (fakeout reclaim)
            signals.extend(self._check_orb_retake(df, current_bar, orb_levels, symbol))
            
        except Exception as e:
            logger.error(f"Error in ORB strategy: {e}")
        
        return signals
    
    def _get_orb_levels(self, df: pd.DataFrame, symbol: str) -> Dict:
        """Get opening range high and low levels"""
        try:
            current_date = df.iloc[-1]['timestamp'].date()
            date_key = str(current_date)
            
            # Check cache first
            if date_key in self.orb_levels:
                return self.orb_levels[date_key]
            
            # Calculate ORB levels for the current day
            session_bars = self._get_session_bars(df)
            
            if len(session_bars) < self.config.range_minutes:
                return {}
            
            # Get opening range bars
            orb_bars = session_bars.head(self.config.range_minutes)
            
            orb_high = orb_bars['high'].max()
            orb_low = orb_bars['low'].min()
            
            # Calculate buffer
            atr = session_bars['atr'].iloc[0] if 'atr' in session_bars.columns else 10
            buffer = max(
                self.config.buffer_atr_factor * atr,
                self.config.min_buffer_points
            )
            
            orb_levels = {
                'high': orb_high,
                'low': orb_low,
                'buffer': buffer,
                'date': current_date,
                'range_start': orb_bars.iloc[0]['timestamp'],
                'range_end': orb_bars.iloc[-1]['timestamp']
            }
            
            # Store in cache
            self.orb_levels[date_key] = orb_levels
            
            return orb_levels
            
        except Exception as e:
            logger.error(f"Error getting ORB levels: {e}")
            return {}
    
    def _get_session_bars(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get bars from current trading session"""
        try:
            # Filter for today's bars
            latest_bar = df.iloc[-1]
            current_date = latest_bar['timestamp'].date()
            
            session_bars = df[df['timestamp'].dt.date == current_date].copy()
            
            return session_bars
            
        except Exception as e:
            logger.error(f"Error getting session bars: {e}")
            return pd.DataFrame()
    
    def _check_breakout(self, df: pd.DataFrame, current_bar: pd.Series, orb_levels: Dict, symbol: str) -> List[Signal]:
        """Check for ORB breakout"""
        signals = []
        
        current_price = current_bar['close']
        orb_high = orb_levels['high']
        orb_low = orb_levels['low']
        buffer = orb_levels['buffer']
        
        # Calculate breakout levels with buffer
        breakout_high = orb_high + buffer
        breakout_low = orb_low - buffer
        
        # Check for gap handling
        if self._check_gap(current_bar, orb_levels, df):
            # Wait for retest after gap
            return signals
        
        # Check for breakout above ORB high
        if current_price >= breakout_high:
            signals.extend(self._generate_orb_signal(
                current_bar, 'long', breakout_high, orb_levels, symbol
            ))
        
        # Check for breakout below ORB low
        elif current_price <= breakout_low:
            signals.extend(self._generate_orb_signal(
                current_bar, 'short', breakout_low, orb_levels, symbol
            ))
        
        return signals
    
    def _check_orb_retake(self, df: pd.DataFrame, current_bar: pd.Series, orb_levels: Dict, symbol: str) -> List[Signal]:
        """Check for ORB retake (fakeout reclaim)"""
        signals = []
        
        # This would need tracking of previous breakout attempts
        # Simplified implementation for now
        
        return signals
    
    def _generate_orb_signal(self, current_bar: pd.Series, signal_type: str, 
                           breakout_level: float, orb_levels: Dict, symbol: str) -> List[Signal]:
        """Generate ORB breakout signal"""
        signals = []
        
        # Volume confirmation
        if 'volume' in current_bar and 'volume_avg' in current_bar:
            if current_bar['volume'] < current_bar['volume_avg'] * self.config.volume_confirm:
                return signals
        
        # Calculate signal parameters
        entry_price = current_bar['close']
        atr = current_bar['atr']
        
        if pd.isna(atr) or atr < 1:
            return signals
        
        if signal_type == 'long':
            stop_loss = orb_levels['low'] - (0.5 * atr)  # Below ORB low
            take_profit = entry_price + (2 * (entry_price - stop_loss))  # 2R target
        else:  # short
            stop_loss = orb_levels['high'] + (0.5 * atr)  # Above ORB high
            take_profit = entry_price - (2 * (stop_loss - entry_price))  # 2R target
        
        # Calculate confidence
        confidence = self.config.base_confidence
        
        # Boost for strong volume
        if 'volume' in current_bar:
            avg_volume = self._get_avg_volume(current_bar)
            if current_bar['volume'] > avg_volume * 2.0:
                confidence += 0.1
        
        # Boost for clean breakout (no immediate pullback)
        if self._is_clean_breakout(current_bar, breakout_level, signal_type):
            confidence += 0.05
        
        # Cap confidence
        confidence = min(confidence, 0.9)
        
        # Calculate risk/reward
        if signal_type == 'long':
            risk = entry_price - stop_loss
            reward = take_profit - entry_price
        else:
            risk = stop_loss - entry_price
            reward = entry_price - take_profit
        
        risk_reward = reward / risk if risk > 0 else 0
        
        signal = Signal(
            timestamp=datetime.now(),
            strategy='orb',
            signal_type=signal_type,
            symbol=symbol,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=confidence,
            risk_reward=risk_reward,
            atr=atr,
            volume=int(current_bar.get('volume', 0)),
            metadata={
                'orb_type': 'standard_breakout',
                'orb_high': orb_levels['high'],
                'orb_low': orb_levels['low'],
                'orb_range_minutes': self.config.range_minutes,
                'buffer_used': orb_levels['buffer'],
                'clean_breakout': self._is_clean_breakout(current_bar, breakout_level, signal_type)
            }
        )
        
        signals.append(signal)
        return signals
    
    def _check_gap(self, current_bar: pd.Series, orb_levels: Dict, df: pd.DataFrame) -> bool:
        """Check if there's a significant gap at open"""
        try:
            # Get previous close (would need previous day's data)
            # Simplified implementation
            if len(df) > 30:
                prev_close = df.iloc[-30]['close']  # Approximate
                gap_size = abs(current_bar['open'] - prev_close)
                atr = current_bar['atr']
                
                return gap_size > self.config.max_gap_atr * atr
            
            return False
            
        except Exception:
            return False
    
    def _is_clean_breakout(self, current_bar: pd.Series, breakout_level: float, signal_type: str) -> bool:
        """Check if breakout is clean (no immediate pullback)"""
        try:
            if signal_type == 'long':
                return current_bar['close'] > breakout_level and current_bar['low'] > breakout_level
            else:
                return current_bar['close'] < breakout_level and current_bar['high'] < breakout_level
        except Exception:
            return False
    
    def _get_avg_volume(self, current_bar: pd.Series) -> float:
        """Get average volume (simplified)"""
        try:
            # In a real implementation, this would calculate 20-period average
            return current_bar.get('volume', 1000)
        except Exception:
            return 1000
    
    def _get_bar_time(self, bar: pd.Series) -> time:
        """Extract time from bar timestamp"""
        try:
            if 'timestamp' in bar and hasattr(bar['timestamp'], 'time'):
                return bar['timestamp'].time()
            else:
                return datetime.now().time()
        except Exception:
            return datetime.now().time()