import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple
from dataclasses import dataclass
from loguru import logger

from src.indicators.custom import CustomIndicators
from src.core.models import Signal

@dataclass
class SupplyDemandConfig:
    """Configuration for supply and demand strategy"""
    base_confidence: float = 0.6
    pivot_strength: int = 8
    min_impulse_atr: float = 1.5
    zone_buffer: float = 2.0
    max_touches: int = 2
    stop_buffer_atr: float = 0.5

class SupplyDemandStrategy:
    """Supply and Demand Zone Trading Strategy"""
    
    def __init__(self, config: Dict):
        self.config = SupplyDemandConfig(**config)
        self.custom_indicators = CustomIndicators()
        self.zone_cache = {}  # Cache zones by symbol
        self.last_signal_time = {}
    
    def generate_signals(self, df: pd.DataFrame, symbol: str) -> List[Signal]:
        """Generate supply/demand zone trading signals"""
        signals = []
        
        try:
            if len(df) < 100:
                return signals
            
            # Identify supply and demand zones
            zones = self._identify_zones(df, symbol)
            
            current_bar = df.iloc[-1]
            
            # Check for zone interactions
            for zone in zones:
                if zone['type'] == 'demand':
                    signals.extend(self._check_demand_zone(df, current_bar, zone, symbol))
                elif zone['type'] == 'supply':
                    signals.extend(self._check_supply_zone(df, current_bar, zone, symbol))
            
        except Exception as e:
            logger.error(f"Error in supply/demand strategy: {e}")
        
        return signals
    
    def _identify_zones(self, df: pd.DataFrame, symbol: str) -> List[Dict]:
        """Identify supply and demand zones"""
        zones = []
        
        # Find pivot points
        pivot_highs, pivot_lows = self.custom_indicators.find_pivot_points(df, self.config.pivot_strength)
        
        # Identify demand zones (support)
        for i in range(len(pivot_lows) - 1):
            if i + 1 >= len(pivot_lows):
                break
            
            pivot_idx = pivot_lows[i]
            next_pivot_idx = pivot_lows[i + 1]
            
            # Calculate impulse move
            pivot_low = df.iloc[pivot_idx]['low']
            next_pivot_low = df.iloc[next_pivot_idx]['low']
            
            # Check if we have a significant impulse
            impulse = abs(df.iloc[next_pivot_idx]['high'] - pivot_low)
            atr_at_pivot = df.iloc[pivot_idx]['atr']
            
            if pd.isna(atr_at_pivot) or impulse < self.config.min_impulse_atr * atr_at_pivot:
                continue
            
            # Create demand zone
            base_candle = df.iloc[pivot_idx]
            zone_top = base_candle['close'] + self.config.zone_buffer
            zone_bottom = base_candle['low'] - self.config.zone_buffer
            
            zones.append({
                'type': 'demand',
                'top': zone_top,
                'bottom': zone_bottom,
                'base_index': pivot_idx,
                'strength': impulse / atr_at_pivot,
                'touches': 0
            })
        
        # Identify supply zones (resistance)
        for i in range(len(pivot_highs) - 1):
            if i + 1 >= len(pivot_highs):
                break
            
            pivot_idx = pivot_highs[i]
            next_pivot_idx = pivot_highs[i + 1]
            
            # Calculate impulse move
            pivot_high = df.iloc[pivot_idx]['high']
            next_pivot_high = df.iloc[next_pivot_idx]['high']
            
            # Check if we have a significant impulse
            impulse = abs(pivot_high - df.iloc[next_pivot_idx]['low'])
            atr_at_pivot = df.iloc[pivot_idx]['atr']
            
            if pd.isna(atr_at_pivot) or impulse < self.config.min_impulse_atr * atr_at_pivot:
                continue
            
            # Create supply zone
            base_candle = df.iloc[pivot_idx]
            zone_top = base_candle['high'] + self.config.zone_buffer
            zone_bottom = base_candle['close'] - self.config.zone_buffer
            
            zones.append({
                'type': 'supply',
                'top': zone_top,
                'bottom': zone_bottom,
                'base_index': pivot_idx,
                'strength': impulse / atr_at_pivot,
                'touches': 0
            })
        
        return zones
    
    def _check_demand_zone(self, df: pd.DataFrame, current_bar: pd.Series, zone: Dict, symbol: str) -> List[Signal]:
        """Check for demand zone entries"""
        signals = []
        
        current_price = current_bar['close']
        
        # Check if price is in the demand zone
        if not (zone['bottom'] <= current_price <= zone['top']):
            return signals
        
        # Check touch count
        if zone['touches'] >= self.config.max_touches:
            return signals
        
        # Look for rejection candle
        recent_bars = df.tail(5)
        rejection_found = False
        
        for i in range(1, len(recent_bars)):
            bar = recent_bars.iloc[i]
            
            # Bullish rejection: wick below zone, close above zone midpoint
            zone_midpoint = (zone['top'] + zone['bottom']) / 2
            wick_size = bar['low'] - min(bar['open'], bar['close'])
            body_size = abs(bar['open'] - bar['close'])
            
            if (bar['low'] <= zone['bottom'] and 
                bar['close'] > zone_midpoint and 
                wick_size > body_size * 0.5):  # Significant wick
                rejection_found = True
                break
        
        if not rejection_found:
            return signals
        
        # Calculate signal parameters
        entry_price = current_bar['close']
        atr = current_bar['atr']
        
        if pd.isna(atr) or atr < 1:
            return signals
        
        stop_loss = zone['bottom'] - (self.config.stop_buffer_atr * atr)
        take_profit = entry_price + (2 * (entry_price - stop_loss))  # 2R target
        
        # Calculate confidence
        confidence = self.config.base_confidence
        
        # Boost for strong zone
        if zone['strength'] > 2.0:
            confidence += 0.1
        
        # Boost for fresh zone (no touches)
        if zone['touches'] == 0:
            confidence += 0.1
        
        # Boost for HTF confluence
        if self._check_htf_confluence(df, 'bullish'):
            confidence += 0.1
        
        # Cap confidence
        confidence = min(confidence, 0.9)
        
        # Calculate risk/reward
        risk = entry_price - stop_loss
        reward = take_profit - entry_price
        risk_reward = reward / risk if risk > 0 else 0
        
        signal = Signal(
            timestamp=datetime.now(),
            strategy='supply_demand',
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
                'zone_type': 'demand',
                'zone_strength': zone['strength'],
                'zone_touches': zone['touches'],
                'fresh_zone': zone['touches'] == 0
            }
        )
        
        signals.append(signal)
        return signals
    
    def _check_supply_zone(self, df: pd.DataFrame, current_bar: pd.Series, zone: Dict, symbol: str) -> List[Signal]:
        """Check for supply zone entries"""
        signals = []
        
        current_price = current_bar['close']
        
        # Check if price is in the supply zone
        if not (zone['bottom'] <= current_price <= zone['top']):
            return signals
        
        # Check touch count
        if zone['touches'] >= self.config.max_touches:
            return signals
        
        # Look for rejection candle
        recent_bars = df.tail(5)
        rejection_found = False
        
        for i in range(1, len(recent_bars)):
            bar = recent_bars.iloc[i]
            
            # Bearish rejection: wick above zone, close below zone midpoint
            zone_midpoint = (zone['top'] + zone['bottom']) / 2
            wick_size = max(bar['open'], bar['close']) - bar['high']
            body_size = abs(bar['open'] - bar['close'])
            
            if (bar['high'] >= zone['top'] and 
                bar['close'] < zone_midpoint and 
                abs(wick_size) > body_size * 0.5):  # Significant wick
                rejection_found = True
                break
        
        if not rejection_found:
            return signals
        
        # Calculate signal parameters
        entry_price = current_bar['close']
        atr = current_bar['atr']
        
        if pd.isna(atr) or atr < 1:
            return signals
        
        stop_loss = zone['top'] + (self.config.stop_buffer_atr * atr)
        take_profit = entry_price - (2 * (stop_loss - entry_price))  # 2R target
        
        # Calculate confidence
        confidence = self.config.base_confidence
        
        # Boost for strong zone
        if zone['strength'] > 2.0:
            confidence += 0.1
        
        # Boost for fresh zone (no touches)
        if zone['touches'] == 0:
            confidence += 0.1
        
        # Boost for HTF confluence
        if self._check_htf_confluence(df, 'bearish'):
            confidence += 0.1
        
        # Cap confidence
        confidence = min(confidence, 0.9)
        
        # Calculate risk/reward
        risk = stop_loss - entry_price
        reward = entry_price - take_profit
        risk_reward = reward / risk if risk > 0 else 0
        
        signal = Signal(
            timestamp=datetime.now(),
            strategy='supply_demand',
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
                'zone_type': 'supply',
                'zone_strength': zone['strength'],
                'zone_touches': zone['touches'],
                'fresh_zone': zone['touches'] == 0
            }
        )
        
        signals.append(signal)
        return signals
    
    def _check_htf_confluence(self, df: pd.DataFrame, direction: str) -> bool:
        """Check for higher timeframe confluence"""
        try:
            if len(df) < 50:
                return False
            
            current_bar = df.iloc[-1]
            
            # Check EMA alignment
            if 'ema_50' in df.columns and 'ema_200' in df.columns:
                ema_50 = current_bar['ema_50']
                ema_200 = current_bar['ema_200']
                
                if direction == 'bullish':
                    return ema_50 > ema_200
                elif direction == 'bearish':
                    return ema_50 < ema_200
            
            return False
            
        except Exception:
            return False