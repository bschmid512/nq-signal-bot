import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict
from dataclasses import dataclass
from loguru import logger

from src.core.engine import Signal

@dataclass
class VWAPConfig:
    """Configuration for VWAP strategy"""
    base_confidence: float = 0.55
    session_start: str = "09:30"
    band_std_dev: list = None
    adx_period: int = 14
    adx_threshold: int = 18
    bb_period: int = 60

class VWAPStrategy:
    """VWAP Mean Reversion and Trend-Following Strategy"""
    
    def __init__(self, config: Dict):
        self.config = VWAPConfig(**config)
        if self.config.band_std_dev is None:
            self.config.band_std_dev = [1.0, 2.0]
        self.last_signal_time = {}
    
    def generate_signals(self, df: pd.DataFrame, symbol: str) -> List[Signal]:
        """Generate VWAP-based trading signals"""
        signals = []
        
        try:
            # Ensure we have required columns
            required_cols = ['close', 'high', 'low', 'vwap', 'vwap_upper_2', 'vwap_lower_2', 'atr', 'adx']
            if not all(col in df.columns for col in required_cols):
                logger.warning(f"Missing required columns for VWAP strategy")
                return signals
            
            if len(df) < 50:
                return signals
            
            current_bar = df.iloc[-1]
            
            # Determine market regime
            regime = self._determine_regime(df, current_bar)
            
            if regime == 'ranging':
                signals.extend(self._check_mean_reversion(df, current_bar, symbol))
            elif regime == 'trending':
                signals.extend(self._check_trend_following(df, current_bar, symbol))
            
        except Exception as e:
            logger.error(f"Error in VWAP strategy: {e}")
        
        return signals
    
    def _determine_regime(self, df: pd.DataFrame, current_bar: pd.Series) -> str:
        """Determine if market is ranging or trending"""
        try:
            # Use ADX for trend detection
            adx = current_bar['adx']
            
            if pd.isna(adx):
                return 'ranging'
            
            if adx < self.config.adx_threshold:
                return 'ranging'
            else:
                return 'trending'
                
        except Exception as e:
            logger.error(f"Error determining regime: {e}")
            return 'ranging'
    
    def _check_mean_reversion(self, df: pd.DataFrame, current_bar: pd.Series, symbol: str) -> List[Signal]:
        """Check for mean reversion opportunities"""
        signals = []
        
        current_price = current_bar['close']
        vwap = current_bar['vwap']
        upper_band = current_bar['vwap_upper_2']
        lower_band = current_bar['vwap_lower_2']
        atr = current_bar['atr']
        
        if pd.isna(atr) or atr < 1:
            return signals
        
        # Check for touch of 2σ band
        if current_price >= upper_band:
            # Bearish mean reversion
            signals.extend(self._generate_vwap_signal(
                df, current_bar, 'short', upper_band, symbol
            ))
        
        elif current_price <= lower_band:
            # Bullish mean reversion
            signals.extend(self._generate_vwap_signal(
                df, current_bar, 'long', lower_band, symbol
            ))
        
        return signals
    
    def _check_trend_following(self, df: pd.DataFrame, current_bar: pd.Series, symbol: str) -> List[Signal]:
        """Check for trend-following opportunities"""
        signals = []
        
        current_price = current_bar['close']
        vwap = current_bar['vwap']
        atr = current_bar['atr']
        
        if pd.isna(atr) or atr < 1:
            return signals
        
        # Determine trend direction relative to VWAP
        if current_price > vwap:
            # Bullish trend - look for pullback to VWAP
            signals.extend(self._check_vwap_pullback(df, current_bar, 'long', symbol))
        
        elif current_price < vwap:
            # Bearish trend - look for pullback to VWAP
            signals.extend(self._check_vwap_pullback(df, current_bar, 'short', symbol))
        
        return signals
    
    def _generate_vwap_signal(self, df: pd.DataFrame, current_bar: pd.Series, 
                            signal_type: str, band_level: float, symbol: str) -> List[Signal]:
        """Generate VWAP mean reversion signal"""
        signals = []
        
        # Check for rejection candle
        recent_bars = df.tail(5)
        rejection_found = False
        
        for i in range(1, len(recent_bars)):
            bar = recent_bars.iloc[i]
            
            if signal_type == 'long':
                # Bullish rejection at lower band
                if (bar['low'] <= current_bar['vwap_lower_2'] and 
                    bar['close'] > bar['open'] and  # Bullish candle
                    bar['close'] > current_bar['vwap_lower_1']):  # Back inside 1σ
                    rejection_found = True
                    break
            
            elif signal_type == 'short':
                # Bearish rejection at upper band
                if (bar['high'] >= current_bar['vwap_upper_2'] and 
                    bar['close'] < bar['open'] and  # Bearish candle
                    bar['close'] < current_bar['vwap_upper_1']):  # Back inside 1σ
                    rejection_found = True
                    break
        
        if not rejection_found:
            return signals
        
        # Calculate signal parameters
        entry_price = current_bar['close']
        atr = current_bar['atr']
        vwap = current_bar['vwap']
        
        if signal_type == 'long':
            stop_loss = entry_price - (0.75 * atr)
            take_profit = vwap  # Target VWAP line
        else:  # short
            stop_loss = entry_price + (0.75 * atr)
            take_profit = vwap  # Target VWAP line
        
        # Calculate confidence
        confidence = self.config.base_confidence
        
        # Boost for strong rejection
        recent_bars = df.tail(10)
        if signal_type == 'long':
            wick_size = recent_bars.iloc[-1]['close'] - recent_bars.iloc[-1]['low']
            body_size = abs(recent_bars.iloc[-1]['open'] - recent_bars.iloc[-1]['close'])
        else:
            wick_size = recent_bars.iloc[-1]['high'] - recent_bars.iloc[-1]['close']
            body_size = abs(recent_bars.iloc[-1]['open'] - recent_bars.iloc[-1]['close'])
        
        if wick_size > body_size * 2:
            confidence += 0.1
        
        # Cap confidence
        confidence = min(confidence, 0.8)
        
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
            strategy='vwap',
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
                'regime': 'ranging',
                'band_touched': 'upper' if signal_type == 'short' else 'lower',
                'rejection_candle': True,
                'target': 'vwap_line'
            }
        )
        
        signals.append(signal)
        return signals
    
    def _check_vwap_pullback(self, df: pd.DataFrame, current_bar: pd.Series, 
                           signal_type: str, symbol: str) -> List[Signal]:
        """Check for VWAP pullback in trending market"""
        signals = []
        
        current_price = current_bar['close']
        vwap = current_bar['vwap']
        atr = current_bar['atr']
        
        # Check if price is near VWAP (pullback)
        distance_to_vwap = abs(current_price - vwap)
        
        if distance_to_vwap > atr:  # Too far from VWAP
            return signals
        
        # Look for confirmation candle
        recent_bars = df.tail(5)
        confirmation_found = False
        
        for i in range(2, len(recent_bars)):
            bar = recent_bars.iloc[i]
            prev_bar = recent_bars.iloc[i-1]
            
            if signal_type == 'long':
                # Bullish confirmation after pullback
                if (prev_bar['close'] < vwap and 
                    bar['close'] > vwap and 
                    bar['close'] > bar['open']):  # Bullish candle
                    confirmation_found = True
                    break
            
            elif signal_type == 'short':
                # Bearish confirmation after pullback
                if (prev_bar['close'] > vwap and 
                    bar['close'] < vwap and 
                    bar['close'] < bar['open']):  # Bearish candle
                    confirmation_found = True
                    break
        
        if not confirmation_found:
            return signals
        
        # Calculate signal parameters
        entry_price = current_bar['close']
        
        if signal_type == 'long':
            stop_loss = entry_price - (1.0 * atr)  # Trail under EMA21
            take_profit = entry_price + (1.5 * (entry_price - stop_loss))
        else:  # short
            stop_loss = entry_price + (1.0 * atr)  # Trail above EMA21
            take_profit = entry_price - (1.5 * (stop_loss - entry_price))
        
        # Calculate confidence
        confidence = self.config.base_confidence + 0.05  # Slightly higher for trend following
        
        # Boost for strong trend (ADX)
        adx = current_bar['adx']
        if adx > 25:
            confidence += 0.1
        
        # Cap confidence
        confidence = min(confidence, 0.85)
        
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
            strategy='vwap',
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
                'regime': 'trending',
                'pullback_to_vwap': True,
                'confirmation_candle': True,
                'adx': adx
            }
        )
        
        signals.append(signal)
        return signals