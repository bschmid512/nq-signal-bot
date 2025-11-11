import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass
from loguru import logger

from src.indicators.custom import CustomIndicators
from src.core.engine import Signal

@dataclass
class DivergenceConfig:
    """Configuration for divergence strategy"""
    base_confidence: float = 0.55
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    min_divergence_points: int = 4
    pivot_strength: int = 5
    ema_period: int = 21
    atr_period: int = 14
    stop_loss_atr: float = 1.0
    take_profit_atr: float = 1.5

class DivergenceStrategy:
    """RSI/MACD Divergence with Swing Pivots Strategy"""
    
    def __init__(self, config: Dict):
        self.config = DivergenceConfig(**config)
        self.custom_indicators = CustomIndicators()
        self.last_signal_time = {}
    
    def generate_signals(self, df: pd.DataFrame, symbol: str) -> List[Signal]:
        """Generate divergence-based trading signals"""
        signals = []
        
        try:
            # Ensure we have required columns
            required_cols = ['close', 'high', 'low', 'rsi', 'macd', 'atr', f'ema_{self.config.ema_period}']
            if not all(col in df.columns for col in required_cols):
                logger.warning(f"Missing required columns for divergence strategy: {required_cols}")
                return signals
            
            # Get the last few bars for analysis
            if len(df) < 50:
                return signals
            
            current_bar = df.iloc[-1]
            recent_bars = df.tail(120)  # Last 120 bars for context
            
            # Detect divergences
            rsi_divergences = self.custom_indicators.detect_divergence(
                recent_bars, recent_bars['rsi'], min_bars=5, max_bars=120
            )
            
            macd_divergences = self.custom_indicators.detect_divergence(
                recent_bars, recent_bars['macd'], min_bars=5, max_bars=120
            )
            
            # Check for recent divergences
            recent_rsi_div = [d for d in rsi_divergences if d['price_index'] >= len(recent_bars) - 10]
            recent_macd_div = [d for d in macd_divergences if d['price_index'] >= len(recent_bars) - 10]
            
            # Generate signals based on divergences
            signals.extend(self._check_bullish_divergence(recent_bars, current_bar, recent_rsi_div, recent_macd_div, symbol))
            signals.extend(self._check_bearish_divergence(recent_bars, current_bar, recent_rsi_div, recent_macd_div, symbol))
            
        except Exception as e:
            logger.error(f"Error in divergence strategy: {e}")
        
        return signals
    
    def _check_bullish_divergence(self, df: pd.DataFrame, current_bar: pd.Series, 
                                 rsi_divs: List[Dict], macd_divs: List[Dict], symbol: str) -> List[Signal]:
        """Check for bullish divergence signals"""
        signals = []
        
        # Check if we have any bullish divergences
        bullish_rsi = [d for d in rsi_divs if d['type'] == 'bullish']
        bullish_macd = [d for d in macd_divs if d['type'] == 'bullish']
        
        if not bullish_rsi and not bullish_macd:
            return signals
        
        # Check for confirmation: price closing above EMA
        ema_col = f'ema_{self.config.ema_period}'
        if ema_col not in df.columns:
            return signals
        
        # Look for recent price action confirmation
        recent_bars = df.tail(10)
        
        # Check if price is reclaiming EMA after divergence
        for i in range(2, len(recent_bars)):
            prev_bar = recent_bars.iloc[i-1]
            curr_bar = recent_bars.iloc[i]
            
            # Price was below EMA and now closing above
            if (prev_bar['close'] < prev_bar[ema_col] and 
                curr_bar['close'] > curr_bar[ema_col]):
                
                # Calculate signal parameters
                entry_price = curr_bar['close']
                atr = curr_bar['atr']
                
                if pd.isna(atr) or atr < 1:
                    continue
                
                stop_loss = entry_price - (self.config.stop_loss_atr * atr)
                take_profit = entry_price + (self.config.take_profit_atr * atr)
                
                # Calculate confidence
                confidence = self.config.base_confidence
                
                # Boost confidence for multiple divergences
                if bullish_rsi and bullish_macd:
                    confidence += 0.1
                
                # Boost confidence for strong divergence
                max_strength = max([d['strength'] for d in (bullish_rsi + bullish_macd)], default=0)
                if max_strength > 0.1:
                    confidence += 0.05
                
                # VWAP confluence boost
                if 'vwap' in df.columns:
                    vwap = curr_bar['vwap']
                    if entry_price > vwap:  # Above VWAP
                        confidence += 0.1
                
                # Cap confidence at 0.9
                confidence = min(confidence, 0.9)
                
                # Calculate risk/reward
                risk = entry_price - stop_loss
                reward = take_profit - entry_price
                risk_reward = reward / risk if risk > 0 else 0
                
                signal = Signal(
                    timestamp=datetime.now(),
                    strategy='divergence',
                    signal_type='long',
                    symbol=symbol,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    confidence=confidence,
                    risk_reward=risk_reward,
                    atr=atr,
                    volume=int(curr_bar.get('volume', 0)),
                    metadata={
                        'divergence_type': 'bullish',
                        'rsi_divergence': len(bullish_rsi) > 0,
                        'macd_divergence': len(bullish_macd) > 0,
                        'ema_reclaim': True,
                        'max_divergence_strength': max_strength
                    }
                )
                
                signals.append(signal)
                break  # Only one signal per divergence setup
        
        return signals
    
    def _check_bearish_divergence(self, df: pd.DataFrame, current_bar: pd.Series,
                                 rsi_divs: List[Dict], macd_divs: List[Dict], symbol: str) -> List[Signal]:
        """Check for bearish divergence signals"""
        signals = []
        
        # Check if we have any bearish divergences
        bearish_rsi = [d for d in rsi_divs if d['type'] == 'bearish']
        bearish_macd = [d for d in macd_divs if d['type'] == 'bearish']
        
        if not bearish_rsi and not bearish_macd:
            return signals
        
        # Check for confirmation: price closing below EMA
        ema_col = f'ema_{self.config.ema_period}'
        if ema_col not in df.columns:
            return signals
        
        # Look for recent price action confirmation
        recent_bars = df.tail(10)
        
        # Check if price is breaking below EMA after divergence
        for i in range(2, len(recent_bars)):
            prev_bar = recent_bars.iloc[i-1]
            curr_bar = recent_bars.iloc[i]
            
            # Price was above EMA and now closing below
            if (prev_bar['close'] > prev_bar[ema_col] and 
                curr_bar['close'] < curr_bar[ema_col]):
                
                # Calculate signal parameters
                entry_price = curr_bar['close']
                atr = curr_bar['atr']
                
                if pd.isna(atr) or atr < 1:
                    continue
                
                stop_loss = entry_price + (self.config.stop_loss_atr * atr)
                take_profit = entry_price - (self.config.take_profit_atr * atr)
                
                # Calculate confidence
                confidence = self.config.base_confidence
                
                # Boost confidence for multiple divergences
                if bearish_rsi and bearish_macd:
                    confidence += 0.1
                
                # Boost confidence for strong divergence
                max_strength = max([d['strength'] for d in (bearish_rsi + bearish_macd)], default=0)
                if max_strength > 0.1:
                    confidence += 0.05
                
                # VWAP confluence boost
                if 'vwap' in df.columns:
                    vwap = curr_bar['vwap']
                    if entry_price < vwap:  # Below VWAP
                        confidence += 0.1
                
                # Cap confidence at 0.9
                confidence = min(confidence, 0.9)
                
                # Calculate risk/reward
                risk = stop_loss - entry_price
                reward = entry_price - take_profit
                risk_reward = reward / risk if risk > 0 else 0
                
                signal = Signal(
                    timestamp=datetime.now(),
                    strategy='divergence',
                    signal_type='short',
                    symbol=symbol,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    confidence=confidence,
                    risk_reward=risk_reward,
                    atr=atr,
                    volume=int(curr_bar.get('volume', 0)),
                    metadata={
                        'divergence_type': 'bearish',
                        'rsi_divergence': len(bearish_rsi) > 0,
                        'macd_divergence': len(bearish_macd) > 0,
                        'ema_break': True,
                        'max_divergence_strength': max_strength
                    }
                )
                
                signals.append(signal)
                break  # Only one signal per divergence setup
        
        return signals
    
    def should_skip_signal(self, symbol: str) -> bool:
        """Check if we should skip generating signals for this symbol"""
        # Check last signal time to avoid over-signaling
        last_time = self.last_signal_time.get(symbol, datetime.min)
        if datetime.now() - last_time < timedelta(minutes=5):
            return True
        
        return False