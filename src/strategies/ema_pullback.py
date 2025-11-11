import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict
from dataclasses import dataclass
from loguru import logger

from src.core.engine import Signal

@dataclass
class EMAPullbackConfig:
    """Configuration for EMA pullback strategy"""
    base_confidence: float = 0.6
    fast_ema: int = 21
    medium_ema: int = 50
    slow_ema: int = 200
    min_slope: float = 0.1  # % per bar
    volume_multiplier: float = 1.2
    atr_period: int = 14

class EMAPullbackStrategy:
    """Trend-following strategy using EMA pullbacks"""
    
    def __init__(self, config: Dict):
        self.config = EMAPullbackConfig(**config)
        self.last_signal_time = {}
    
    def generate_signals(self, df: pd.DataFrame, symbol: str) -> List[Signal]:
        """Generate EMA pullback trading signals"""
        signals = []
        
        try:
            # Ensure we have required columns
            required_cols = ['close', 'high', 'low', 'volume', 'atr', 'vwap']
            ema_cols = [f'ema_{self.config.fast_ema}', f'ema_{self.config.medium_ema}', f'ema_{self.config.slow_ema}']
            required_cols.extend(ema_cols)
            
            if not all(col in df.columns for col in required_cols):
                logger.warning(f"Missing required columns for EMA pullback strategy")
                return signals
            
            if len(df) < self.config.slow_ema + 10:
                return signals
            
            current_bar = df.iloc[-1]
            
            # Determine trend direction
            trend_direction = self._determine_trend_direction(df)
            
            if trend_direction == 'bullish':
                signals.extend(self._check_bullish_pullback(df, current_bar, symbol))
            elif trend_direction == 'bearish':
                signals.extend(self._check_bearish_pullback(df, current_bar, symbol))
            
        except Exception as e:
            logger.error(f"Error in EMA pullback strategy: {e}")
        
        return signals
    
    def _determine_trend_direction(self, df: pd.DataFrame) -> str:
        """Determine the current trend direction based on EMA alignment"""
        try:
            current_bar = df.iloc[-1]
            prev_bar = df.iloc[-2]
            
            ema_fast = current_bar[f'ema_{self.config.fast_ema}']
            ema_medium = current_bar[f'ema_{self.config.medium_ema}']
            ema_slow = current_bar[f'ema_{self.config.slow_ema}']
            
            # Bullish trend: fast > medium > slow
            if ema_fast > ema_medium > ema_slow:
                # Check slope of fast EMA
                prev_ema_fast = prev_bar[f'ema_{self.config.fast_ema}']
                ema_slope = (ema_fast - prev_ema_fast) / prev_ema_fast * 100
                
                if ema_slope >= self.config.min_slope:
                    return 'bullish'
            
            # Bearish trend: fast < medium < slow
            elif ema_fast < ema_medium < ema_slow:
                # Check slope of fast EMA
                prev_ema_fast = prev_bar[f'ema_{self.config.fast_ema}']
                ema_slope = (ema_fast - prev_ema_fast) / prev_ema_fast * 100
                
                if ema_slope <= -self.config.min_slope:
                    return 'bearish'
            
            return 'neutral'
            
        except Exception as e:
            logger.error(f"Error determining trend direction: {e}")
            return 'neutral'
    
    def _check_bullish_pullback(self, df: pd.DataFrame, current_bar: pd.Series, symbol: str) -> List[Signal]:
        """Check for bullish pullback entries"""
        signals = []
        
        # Get recent bars for context
        recent_bars = df.tail(20)
        
        # Look for pullback to EMA21 or EMA50
        fast_ema_col = f'ema_{self.config.fast_ema}'
        medium_ema_col = f'ema_{self.config.medium_ema}'
        
        # Check if we're in a pullback
        pullback_detected = False
        support_level = None
        
        for i in range(2, len(recent_bars)):
            bar = recent_bars.iloc[i]
            prev_bar = recent_bars.iloc[i-1]
            
            # Check if price pulled back to EMA21
            if (bar['low'] <= bar[fast_ema_col] <= bar['high'] and 
                prev_bar['close'] < prev_bar[fast_ema_col] and
                bar['close'] > bar[fast_ema_col]):
                pullback_detected = True
                support_level = bar[fast_ema_col]
                break
            
            # Check if price pulled back to EMA50
            elif (bar['low'] <= bar[medium_ema_col] <= bar['high'] and 
                  prev_bar['close'] < prev_bar[medium_ema_col] and
                  bar['close'] > bar[medium_ema_col]):
                pullback_detected = True
                support_level = bar[medium_ema_col]
                break
        
        if not pullback_detected:
            return signals
        
        # Check for confirmation candle
        if (current_bar['close'] > current_bar['open'] and  # Bullish candle
            current_bar['close'] > current_bar[fast_ema_col]):  # Above fast EMA
            
            # Volume confirmation
            avg_volume = df['volume'].tail(20).mean()
            if current_bar['volume'] < avg_volume * self.config.volume_multiplier:
                return signals
            
            # VWAP and session bias check
            if 'vwap' in df.columns:
                vwap = current_bar['vwap']
                session_open = self._get_session_open(df)
                
                # Only long above VWAP and session open
                if current_bar['close'] < vwap or current_bar['close'] < session_open:
                    return signals
            
            # Volatility filter
            atr = current_bar['atr']
            if pd.isna(atr) or atr < 6.0:  # Skip if ATR too low
                return signals
            
            # Calculate signal parameters
            entry_price = current_bar['close']
            stop_loss = support_level - (0.5 * atr)  # Below support
            take_profit = entry_price + (2 * (entry_price - stop_loss))  # 2R target
            
            # Calculate confidence
            confidence = self.config.base_confidence
            
            # Boost for strong EMA slope
            ema_slope = self._calculate_ema_slope(df, fast_ema_col)
            if ema_slope > 0.2:
                confidence += 0.1
            
            # Boost for volume
            if current_bar['volume'] > avg_volume * 1.5:
                confidence += 0.05
            
            # Cap confidence
            confidence = min(confidence, 0.9)
            
            # Calculate risk/reward
            risk = entry_price - stop_loss
            reward = take_profit - entry_price
            risk_reward = reward / risk if risk > 0 else 0
            
            signal = Signal(
                timestamp=datetime.now(),
                strategy='ema_pullback',
                signal_type='long',
                symbol=symbol,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                confidence=confidence,
                risk_reward=risk_reward,
                atr=atr,
                volume=int(current_bar.get('volume', 0)),
                metadata={
                    'trend_direction': 'bullish',
                    'support_level': support_level,
                    'ema_slope': ema_slope,
                    'volume_ratio': current_bar['volume'] / avg_volume
                }
            )
            
            signals.append(signal)
        
        return signals
    
    def _check_bearish_pullback(self, df: pd.DataFrame, current_bar: pd.Series, symbol: str) -> List[Signal]:
        """Check for bearish pullback entries"""
        signals = []
        
        # Get recent bars for context
        recent_bars = df.tail(20)
        
        # Look for pullback to EMA21 or EMA50
        fast_ema_col = f'ema_{self.config.fast_ema}'
        medium_ema_col = f'ema_{self.config.medium_ema}'
        
        # Check if we're in a pullback
        pullback_detected = False
        resistance_level = None
        
        for i in range(2, len(recent_bars)):
            bar = recent_bars.iloc[i]
            prev_bar = recent_bars.iloc[i-1]
            
            # Check if price pulled back to EMA21
            if (bar['low'] <= bar[fast_ema_col] <= bar['high'] and 
                prev_bar['close'] > prev_bar[fast_ema_col] and
                bar['close'] < bar[fast_ema_col]):
                pullback_detected = True
                resistance_level = bar[fast_ema_col]
                break
            
            # Check if price pulled back to EMA50
            elif (bar['low'] <= bar[medium_ema_col] <= bar['high'] and 
                  prev_bar['close'] > prev_bar[medium_ema_col] and
                  bar['close'] < bar[medium_ema_col]):
                pullback_detected = True
                resistance_level = bar[medium_ema_col]
                break
        
        if not pullback_detected:
            return signals
        
        # Check for confirmation candle
        if (current_bar['close'] < current_bar['open'] and  # Bearish candle
            current_bar['close'] < current_bar[fast_ema_col]):  # Below fast EMA
            
            # Volume confirmation
            avg_volume = df['volume'].tail(20).mean()
            if current_bar['volume'] < avg_volume * self.config.volume_multiplier:
                return signals
            
            # VWAP and session bias check
            if 'vwap' in df.columns:
                vwap = current_bar['vwap']
                session_open = self._get_session_open(df)
                
                # Only short below VWAP and session open
                if current_bar['close'] > vwap or current_bar['close'] > session_open:
                    return signals
            
            # Volatility filter
            atr = current_bar['atr']
            if pd.isna(atr) or atr < 6.0:  # Skip if ATR too low
                return signals
            
            # Calculate signal parameters
            entry_price = current_bar['close']
            stop_loss = resistance_level + (0.5 * atr)  # Above resistance
            take_profit = entry_price - (2 * (stop_loss - entry_price))  # 2R target
            
            # Calculate confidence
            confidence = self.config.base_confidence
            
            # Boost for strong EMA slope
            ema_slope = self._calculate_ema_slope(df, fast_ema_col)
            if ema_slope < -0.2:
                confidence += 0.1
            
            # Boost for volume
            if current_bar['volume'] > avg_volume * 1.5:
                confidence += 0.05
            
            # Cap confidence
            confidence = min(confidence, 0.9)
            
            # Calculate risk/reward
            risk = stop_loss - entry_price
            reward = entry_price - take_profit
            risk_reward = reward / risk if risk > 0 else 0
            
            signal = Signal(
                timestamp=datetime.now(),
                strategy='ema_pullback',
                signal_type='short',
                symbol=symbol,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                confidence=confidence,
                risk_reward=risk_reward,
                atr=atr,
                volume=int(current_bar.get('volume', 0)),
                metadata={
                    'trend_direction': 'bearish',
                    'resistance_level': resistance_level,
                    'ema_slope': ema_slope,
                    'volume_ratio': current_bar['volume'] / avg_volume
                }
            )
            
            signals.append(signal)
        
        return signals
    
    def _calculate_ema_slope(self, df: pd.DataFrame, ema_col: str) -> float:
        """Calculate the slope of EMA"""
        try:
            if len(df) < 10:
                return 0
            
            recent_ema = df[ema_col].tail(10)
            current_ema = recent_ema.iloc[-1]
            prev_ema = recent_ema.iloc[-5]  # 5 bars ago
            
            if pd.isna(current_ema) or pd.isna(prev_ema):
                return 0
            
            slope = (current_ema - prev_ema) / prev_ema * 100 / 5  # % per bar
            return slope
            
        except Exception:
            return 0
    
    def _get_session_open(self, df: pd.DataFrame) -> float:
        """Get session open price"""
        try:
            # Simple implementation - use first bar of the day
            # In a real implementation, you'd track session boundaries
            return df.iloc[0]['open']
        except:
            return df.iloc[-1]['close']