import pandas as pd
import numpy as np
from datetime import datetime, time
from typing import List, Dict
from dataclasses import dataclass
from loguru import logger

from src.indicators.custom import CustomIndicators
from src.core.models import Signal

@dataclass
class ReversalBreakoutConfig:
    """Configuration for Reversal-Breakout strategy"""
    base_confidence: float = 0.65
    rsi_oversold: float = 25  # Lower than standard 30 (stricter)
    rsi_overbought: float = 75  # Higher than standard 70 (stricter)
    rsi_period: int = 14
    atr_period: int = 14
    stop_loss_atr: float = 1.0  # Tight stop (reversal fails fast)
    take_profit_r_mult: float = 2.5  # High reward for catching turns
    volume_confirmation: float = 1.5  # 50% above average volume
    breakout_buffer: float = 0.1  # How far beyond level to trigger

class ReversalBreakoutStrategy:
    """Reversal-Breakout Hybrid Strategy for NQ Futures
    
    Identifies:
    1. Reversal setup: RSI extreme + momentum exhaustion
    2. Breakout confirmation: Break of prior swing point
    3. Volume validation: Above-average participation
    
    Best for: Post-lunch reversals, V-turns, false breakout traps
    Avoid: First 30 min (chop), news spikes (whipsaws)
    """
    
    def __init__(self, config: Dict):
        self.config = ReversalBreakoutConfig(**config)
        self.custom_indicators = CustomIndicators()
        self.last_signal_time = {}
        
        # Session filters
        self.session_start = time(9, 30)
        self.lunch_start = time(11, 30)
        self.lunch_end = time(13, 30)
        self.last_hour = time(15, 45)
    
    def generate_signals(self, df: pd.DataFrame, symbol: str) -> List[Signal]:
        """Generate reversal-breakout trading signals"""
        signals = []
        
        try:
            # Ensure we have required columns
            required_cols = ['close', 'high', 'low', 'volume', 'atr']
            if not all(col in df.columns for col in required_cols):
                logger.warning(f"Missing required columns for reversal-breakout strategy")
                return signals
            
            if len(df) < 50:
                return signals
            
            current_bar = df.iloc[-1]
            
            # Session filters
            current_time = datetime.now().time()
            if current_time < self.session_start:
                return signals
            if self.lunch_start <= current_time <= self.lunch_end:
                return signals
            if current_time >= self.last_hour:
                return signals
            
            # Calculate RSI
            df['rsi'] = self._calculate_rsi(df['close'], self.config.rsi_period)
            
            # Find swing points
            pivot_highs, pivot_lows = self.custom_indicators.find_pivot_points(df, strength=5)
            
            # Check for bullish reversal-breakout
            signals.extend(self._check_bullish_reversal_breakout(df, current_bar, pivot_lows, symbol))
            
            # Check for bearish reversal-breakout
            signals.extend(self._check_bearish_reversal_breakout(df, current_bar, pivot_highs, symbol))
            
        except Exception as e:
            logger.error(f"Error in reversal-breakout strategy: {e}")
        
        return signals
    
    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate RSI manually (pandas_ta may not be available)"""
        delta = prices.diff()
        
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _check_bullish_reversal_breakout(self, df: pd.DataFrame, current_bar: pd.Series, 
                                       pivot_lows: List[int], symbol: str) -> List[Signal]:
        """Check for bullish reversal followed by breakout"""
        signals = []
        
        # Check RSI oversold
        rsi = current_bar['rsi']
        if pd.isna(rsi) or rsi > self.config.rsi_oversold:
            return signals
        
        # Ensure we have pivot points
        if len(pivot_lows) < 2:
            return signals
        
        # Find most recent swing low
        last_swing_idx = pivot_lows[-1]
        swing_low_price = df.iloc[last_swing_idx]['low']
        
        # Check if we're breaking above the swing low with buffer
        breakout_level = swing_low_price + (self.config.breakout_buffer * current_bar['atr'])
        
        if current_bar['close'] <= breakout_level:
            return signals
        
        # Look for rejection candle (bullish reversal sign)
        recent_bars = df.tail(5)
        rejection_found = False
        
        for i in range(len(recent_bars) - 1):
            bar = recent_bars.iloc[i]
            
            # Bullish rejection: long wick + close near high
            wick_size = bar['close'] - bar['low']
            body_size = abs(bar['open'] - bar['close'])
            
            if wick_size > body_size * 2 and bar['close'] > bar['open']:
                rejection_found = True
                break
        
        if not rejection_found:
            return signals
        
        # Volume confirmation
        avg_volume = df['volume'].tail(20).mean()
        if current_bar['volume'] < avg_volume * self.config.volume_confirmation:
            return signals
        
        # Calculate signal parameters
        entry_price = current_bar['close']
        atr = current_bar['atr']
        
        if pd.isna(atr) or atr < 1:
            return signals
        
        # Stop loss: below swing low
        stop_loss = swing_low_price - (self.config.stop_loss_atr * atr)
        
        # Take profit: dynamic R-mult
        risk = entry_price - stop_loss
        take_profit = entry_price + (self.config.take_profit_r_mult * risk)
        
        # Calculate confidence
        confidence = self.config.base_confidence
        
        # Boost for extreme RSI
        if rsi < 20:
            confidence += 0.1
        
        # Boost for high volume
        if current_bar['volume'] > avg_volume * 2:
            confidence += 0.05
        
        # Boost for strong breakout (close near high)
        if current_bar['close'] >= current_bar['high'] - (0.2 * (current_bar['high'] - current_bar['low'])):
            confidence += 0.05
        
        # Cap confidence
        confidence = min(confidence, 0.9)
        
        # Calculate risk/reward
        risk_reward = self.config.take_profit_r_mult
        
        signal = Signal(
            timestamp=datetime.now(),
            strategy='reversal_breakout',
            signal_type='long',
            symbol=symbol,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=confidence,
            risk_reward=risk_reward,
            atr=atr,
            volume=int(current_bar['volume']),
            metadata={
                'setup_type': 'bullish_reversal_breakout',
                'swing_low_level': swing_low_price,
                'rsi_value': rsi,
                'volume_ratio': current_bar['volume'] / avg_volume,
                'rejection_candle': True,
                'breakout_buffer': self.config.breakout_buffer
            }
        )
        
        signals.append(signal)
        return signals
    
    def _check_bearish_reversal_breakout(self, df: pd.DataFrame, current_bar: pd.Series, 
                                        pivot_highs: List[int], symbol: str) -> List[Signal]:
        """Check for bearish reversal followed by breakout"""
        signals = []
        
        # Check RSI overbought
        rsi = current_bar['rsi']
        if pd.isna(rsi) or rsi < self.config.rsi_overbought:
            return signals
        
        # Ensure we have pivot points
        if len(pivot_highs) < 2:
            return signals
        
        # Find most recent swing high
        last_swing_idx = pivot_highs[-1]
        swing_high_price = df.iloc[last_swing_idx]['high']
        
        # Check if we're breaking below the swing high with buffer
        breakout_level = swing_high_price - (self.config.breakout_buffer * current_bar['atr'])
        
        if current_bar['close'] >= breakout_level:
            return signals
        
        # Look for rejection candle (bearish reversal sign)
        recent_bars = df.tail(5)
        rejection_found = False
        
        for i in range(len(recent_bars) - 1):
            bar = recent_bars.iloc[i]
            
            # Bearish rejection: long upper wick + close near low
            wick_size = bar['high'] - bar['close']
            body_size = abs(bar['open'] - bar['close'])
            
            if wick_size > body_size * 2 and bar['close'] < bar['open']:
                rejection_found = True
                break
        
        if not rejection_found:
            return signals
        
        # Volume confirmation
        avg_volume = df['volume'].tail(20).mean()
        if current_bar['volume'] < avg_volume * self.config.volume_confirmation:
            return signals
        
        # Calculate signal parameters
        entry_price = current_bar['close']
        atr = current_bar['atr']
        
        if pd.isna(atr) or atr < 1:
            return signals
        
        # Stop loss: above swing high
        stop_loss = swing_high_price + (self.config.stop_loss_atr * atr)
        
        # Take profit: dynamic R-mult
        risk = stop_loss - entry_price
        take_profit = entry_price - (self.config.take_profit_r_mult * risk)
        
        # Calculate confidence
        confidence = self.config.base_confidence
        
        # Boost for extreme RSI
        if rsi > 80:
            confidence += 0.1
        
        # Boost for high volume
        if current_bar['volume'] > avg_volume * 2:
            confidence += 0.05
        
        # Boost for strong breakout (close near low)
        if current_bar['close'] <= current_bar['low'] + (0.2 * (current_bar['high'] - current_bar['low'])):
            confidence += 0.05
        
        # Cap confidence
        confidence = min(confidence, 0.9)
        
        # Calculate risk/reward
        risk_reward = self.config.take_profit_r_mult
        
        signal = Signal(
            timestamp=datetime.now(),
            strategy='reversal_breakout',
            signal_type='short',
            symbol=symbol,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=confidence,
            risk_reward=risk_reward,
            atr=atr,
            volume=int(current_bar['volume']),
            metadata={
                'setup_type': 'bearish_reversal_breakout',
                'swing_high_level': swing_high_price,
                'rsi_value': rsi,
                'volume_ratio': current_bar['volume'] / avg_volume,
                'rejection_candle': True,
                'breakout_buffer': self.config.breakout_buffer
            }
        )
        
        signals.append(signal)
        return signals