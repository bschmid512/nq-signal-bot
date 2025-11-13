"""
ADAPTED HTF SUPERTREND STRATEGY
Fixed: Added take_profit_r_mult config field
"""

import pandas as pd
import numpy as np
from datetime import datetime, time
from typing import List, Dict
from dataclasses import dataclass
from loguru import logger

from src.core.models import Signal

@dataclass
class HTFSupertrendConfig:
    """Configuration for HTF Supertrend strategy"""
    base_confidence: float = 0.65
    htf_ema: int = 200  # Use EMA(200) on 5-min as HTF proxy
    supertrend_atr: int = 10
    supertrend_multiplier: float = 3.0
    news_buffer_minutes: int = 2
    max_stop_atr: float = 1.5
    take_profit_r_mult: float = 2.5  # Dynamic R-mult instead of fixed

class HTFSupertrendStrategy:
    """Higher Time Frame Confirmation with Supertrend Strategy
    
    ADAPTED FOR 5-MIN DATA ONLY:
    - Uses EMA(200) on 5-min as proxy for HTF bias (simulates 15-min trend)
    - Uses Supertrend on 5-min for entries (instead of 1-min)
    """
    
    def __init__(self, config: Dict):
        self.config = HTFSupertrendConfig(**config)
        self.last_signal_time = {}
        self.session_start = time(9, 30)  # Market open
        self.session_end = time(16, 0)    # Market close
        
        logger.info(f"HTF Supertrend adapted for 5-min data (using EMA({self.config.htf_ema}) as HTF proxy)")
    
    def generate_signals(self, df: pd.DataFrame, symbol: str) -> List[Signal]:
        """Generate HTF Supertrend trading signals"""
        signals = []
        
        try:
            # Ensure we have required columns
            required_cols = ['close', 'high', 'low', f'ema_{self.config.htf_ema}', 'supertrend', 'atr']
            if not all(col in df.columns for col in required_cols):
                logger.warning(f"Missing required columns for HTF Supertrend strategy")
                return signals
            
            if len(df) < 50:
                return signals
            
            current_bar = df.iloc[-1]
            
            # Determine HTF bias using EMA(200) as proxy
            htf_bias = self._determine_htf_bias(df)
            
            if htf_bias == 'bullish':
                signals.extend(self._check_bullish_supertrend(df, current_bar, symbol))
            elif htf_bias == 'bearish':
                signals.extend(self._check_bearish_supertrend(df, current_bar, symbol))
            
        except Exception as e:
            logger.error(f"Error in HTF Supertrend strategy: {e}")
        
        return signals
    
    def _determine_htf_bias(self, df: pd.DataFrame) -> str:
        """Determine higher timeframe directional bias using EMA(200) proxy"""
        try:
            current_bar = df.iloc[-1]
            htf_ema_col = f'ema_{self.config.htf_ema}'
            
            if htf_ema_col not in df.columns:
                return 'neutral'
            
            # Price position relative to long-term EMA (HTF proxy)
            price = current_bar['close']
            htf_ema = current_bar[htf_ema_col]
            
            if pd.isna(htf_ema):
                return 'neutral'
            
            # Calculate strength of bias
            distance_pct = (price - htf_ema) / htf_ema * 100
            
            # Strong bias required (>0.3% away from EMA)
            if distance_pct > 0.3:
                return 'bullish'
            elif distance_pct < -0.3:
                return 'bearish'
            else:
                return 'neutral'
                
        except Exception as e:
            logger.error(f"Error determining HTF bias: {e}")
            return 'neutral'
    
    def _check_bullish_supertrend(self, df: pd.DataFrame, current_bar: pd.Series, symbol: str) -> List[Signal]:
        """Check for bullish supertrend entries"""
        signals = []
        
        # Check if we're in uptrend according to supertrend
        if pd.isna(current_bar['supertrend']):
            return signals
            
        if current_bar['close'] <= current_bar['supertrend']:
            return signals
        
        # Get recent bars for context
        recent_bars = df.tail(10)
        
        # Look for pullback to supertrend line (entry trigger)
        pullback_detected = False
        entry_candle_idx = None
        
        for i in range(2, len(recent_bars)):
            bar = recent_bars.iloc[i]
            prev_bar = recent_bars.iloc[i-1]
            
            if pd.isna(bar['supertrend']) or pd.isna(prev_bar['supertrend']):
                continue
            
            # Check for pullback to supertrend
            if (prev_bar['low'] <= prev_bar['supertrend'] and 
                bar['close'] > bar['supertrend'] and
                bar['close'] > bar['open']):  # Bullish candle
                
                pullback_detected = True
                entry_candle_idx = i
                break
        
        if not pullback_detected:
            return signals
        
        # Additional filters
        if self._should_skip_signal(current_bar):
            return signals
        
        # Calculate signal parameters
        entry_price = current_bar['close']
        atr = current_bar['atr']
        
        if pd.isna(atr) or atr < 1:
            return signals
        
        # Stop loss at supertrend flip or max ATR
        supertrend_stop = current_bar['supertrend']
        atr_stop = entry_price - (self.config.max_stop_atr * atr)
        stop_loss = max(supertrend_stop, atr_stop)
        
        # Take profit - dynamic R-mult
        risk = entry_price - stop_loss
        hard_tp = entry_price + (self.config.take_profit_r_mult * risk)
        
        # Calculate confidence
        confidence = self.config.base_confidence
        
        # Boost for strong HTF bias
        htf_ema_col = f'ema_{self.config.htf_ema}'
        if htf_ema_col in df.columns:
            price_vs_ema = (entry_price - current_bar[htf_ema_col]) / current_bar[htf_ema_col] * 100
            if price_vs_ema > 0.5:  # Well above HTF EMA
                confidence += 0.1
        
        # Boost for volume
        if 'volume' in current_bar:
            avg_volume = df['volume'].tail(20).mean()
            if current_bar['volume'] > avg_volume * 1.5:
                confidence += 0.05
        
        # Cap confidence
        confidence = min(confidence, 0.9)
        
        # Calculate risk/reward
        reward = hard_tp - entry_price
        risk_reward = reward / risk if risk > 0 else 0
        
        signal = Signal(
            timestamp=datetime.now(),
            strategy='htf_supertrend',
            signal_type='long',
            symbol=symbol,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=hard_tp,
            confidence=confidence,
            risk_reward=risk_reward,
            atr=atr,
            volume=int(current_bar.get('volume', 0)),
            metadata={
                'htf_bias': 'bullish',
                'htf_proxy': f'ema_{self.config.htf_ema}',
                'trend': 'uptrend',
                'pullback_to_supertrend': True,
                'supertrend_level': current_bar['supertrend'],
                'adapted_for_5min': True
            }
        )
        
        signals.append(signal)
        return signals
    
    def _check_bearish_supertrend(self, df: pd.DataFrame, current_bar: pd.Series, symbol: str) -> List[Signal]:
        """Check for bearish supertrend entries"""
        signals = []
        
        # Check if we're in downtrend according to supertrend
        if pd.isna(current_bar['supertrend']):
            return signals
            
        if current_bar['close'] >= current_bar['supertrend']:
            return signals
        
        # Get recent bars for context
        recent_bars = df.tail(10)
        
        # Look for pullback to supertrend line
        pullback_detected = False
        
        for i in range(2, len(recent_bars)):
            bar = recent_bars.iloc[i]
            prev_bar = recent_bars.iloc[i-1]
            
            if pd.isna(bar['supertrend']) or pd.isna(prev_bar['supertrend']):
                continue
            
            # Check for pullback to supertrend
            if (prev_bar['high'] >= prev_bar['supertrend'] and 
                bar['close'] < bar['supertrend'] and
                bar['close'] < bar['open']):  # Bearish candle
                
                pullback_detected = True
                break
        
        if not pullback_detected:
            return signals
        
        # Additional filters
        if self._should_skip_signal(current_bar):
            return signals
        
        # Calculate signal parameters
        entry_price = current_bar['close']
        atr = current_bar['atr']
        
        if pd.isna(atr) or atr < 1:
            return signals
        
        # Stop loss at supertrend flip or max ATR
        supertrend_stop = current_bar['supertrend']
        atr_stop = entry_price + (self.config.max_stop_atr * atr)
        stop_loss = min(supertrend_stop, atr_stop)
        
        # Take profit - dynamic R-mult
        risk = stop_loss - entry_price
        hard_tp = entry_price - (self.config.take_profit_r_mult * risk)
        
        # Calculate confidence
        confidence = self.config.base_confidence
        
        # Boost for strong HTF bias
        htf_ema_col = f'ema_{self.config.htf_ema}'
        if htf_ema_col in df.columns:
            price_vs_ema = (entry_price - current_bar[htf_ema_col]) / current_bar[htf_ema_col] * 100
            if price_vs_ema < -0.5:  # Well below HTF EMA
                confidence += 0.1
        
        # Boost for volume
        if 'volume' in current_bar:
            avg_volume = df['volume'].tail(20).mean()
            if current_bar['volume'] > avg_volume * 1.5:
                confidence += 0.05
        
        # Cap confidence
        confidence = min(confidence, 0.9)
        
        # Calculate risk/reward
        reward = entry_price - hard_tp
        risk_reward = reward / risk if risk > 0 else 0
        
        signal = Signal(
            timestamp=datetime.now(),
            strategy='htf_supertrend',
            signal_type='short',
            symbol=symbol,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=hard_tp,
            confidence=confidence,
            risk_reward=risk_reward,
            atr=atr,
            volume=int(current_bar.get('volume', 0)),
            metadata={
                'htf_bias': 'bearish',
                'htf_proxy': f'ema_{self.config.htf_ema}',
                'trend': 'downtrend',
                'pullback_to_supertrend': True,
                'supertrend_level': current_bar['supertrend'],
                'adapted_for_5min': True
            }
        )
        
        signals.append(signal)
        return signals
    
    def _should_skip_signal(self, current_bar: pd.Series) -> bool:
        """Check if we should skip this signal"""
        # News buffer check (simplified)
        current_time = datetime.now().time()
        
        # Skip if near market open (first 15 minutes)
        if self.session_start <= current_time <= time(9, 45):
            return True
        
        # Skip if low volatility
        atr = current_bar.get('atr', 0)
        if pd.isna(atr) or atr < 5.0:
            return True
        
        return False