import pandas as pd
import numpy as np
from datetime import datetime, time
from typing import List, Dict
from dataclasses import dataclass
from loguru import logger

from src.core.models import Signal

@dataclass
class EMAPullbackConfig:
    """Configuration for EMA pullback strategy"""
    base_confidence: float = 0.6
    fast_ema: int = 21
    medium_ema: int = 50
    slow_ema: int = 200
    min_slope: float = 0.1
    volume_multiplier: float = 1.2
    atr_period: int = 14
    stop_loss_atr: float = 1.0
    take_profit_r_mult: float = 2.0  # Dynamic R-mult

class EMAPullbackStrategy:
    """Trend-following strategy using EMA pullbacks with session filters"""
    
    def __init__(self, config: Dict):
        self.config = EMAPullbackConfig(**config)
        self.last_signal_time = {}
        
        # Session filters
        self.lunch_start = time(11, 30)
        self.lunch_end = time(13, 30)
        self.last_hour = time(15, 45)
    
    def generate_signals(self, df: pd.DataFrame, symbol: str) -> List[Signal]:
        """Generate EMA pullback trading signals with detailed logging."""
        signals: List[Signal] = []

        logger.debug(
            f"[EMAPullback] start generate_signals for {symbol} "
            f"df.shape={df.shape}, cols={list(df.columns)}"
        )

        # Session filter
        current_time = datetime.now().time()
        if self.lunch_start <= current_time <= self.lunch_end:
            logger.debug("[EMAPullback] exiting – lunch window.")
            return signals

        if current_time >= self.last_hour:
            logger.debug("[EMAPullback] exiting – last hour of session.")
            return signals

        try:
            # Required columns for this strategy
            required_cols = ["close", "high", "low", "volume", "atr", "vwap"]
            ema_cols = [
                f"ema_{self.config.fast_ema}",
                f"ema_{self.config.medium_ema}",
                f"ema_{self.config.slow_ema}",
            ]
            required_cols.extend(ema_cols)

            missing = [c for c in required_cols if c not in df.columns]
            if missing:
                logger.debug(f"[EMAPullback] missing required columns: {missing}")
                return signals

            # Need enough bars to have a stable slow EMA
            min_bars = self.config.slow_ema + 10
            if len(df) < min_bars:
                logger.debug(
                    f"[EMAPullback] exiting – not enough bars ({len(df)} < {min_bars})"
                )
                return signals

            current_bar = df.iloc[-1]

            # Determine trend direction (already implemented helper)
            trend_direction = self._determine_trend_direction(df)
            logger.debug(f"[EMAPullback] trend_direction={trend_direction}")

            if trend_direction == "bullish":
                # ... your existing bullish pullback logic ...
                # At decision points, sprinkle logs, e.g.:
                # if not pullback_ok:
                #     logger.debug("[EMAPullback] bullish setup rejected – pullback not valid.")
                # else:
                #     logger.debug("[EMAPullback] bullish setup accepted – building signal.")
                pass

            elif trend_direction == "bearish":
                # ... your existing bearish pullback logic ...
                pass
            else:
                logger.debug("[EMAPullback] no clear trend – no signals.")
                return signals

            # When you actually create signals, you can log them:
            # signals.append(Signal(...))
            # logger.debug(
            #     f"[EMAPullback] created {direction} signal @ {entry_price:.2f}, "
            #     f"SL={stop_loss:.2f}, TP={take_profit:.2f}, conf={confidence:.2f}"
            # )

        except Exception as e:
            logger.error(f"[EMAPullback] Error in EMA pullback strategy for {symbol}: {e}")

        logger.info(f"[EMAPullback] generated {len(signals)} signals for {symbol}.")
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
            if pd.isna(atr) or atr < 8.0:  # Skip if ATR too low
                return signals
            
            # Calculate signal parameters
            entry_price = current_bar['close']
            stop_loss = support_level - (0.5 * atr)  # Below support
            
            # Dynamic take profit based on R-mult
            risk = entry_price - stop_loss
            take_profit = entry_price + (self.config.take_profit_r_mult * risk)
            
            # Calculate confidence
            confidence = self.config.base_confidence
            
            # Boost for strong EMA slope
            ema_slope = self._calculate_ema_slope(df, fast_ema_col)
            if ema_slope > 0.2:
                confidence += 0.1
            
            # Boost for volume
            if current_bar['volume'] > avg_volume * 1.5:
                confidence += 0.05
            
            # Boost for trending regime
            if 'adx' in current_bar and current_bar['adx'] > 25:
                confidence += 0.05
            
            # Cap confidence
            confidence = min(confidence, 0.9)
            
            # Calculate risk/reward
            risk_reward = self.config.take_profit_r_mult
            
            # Skip if risk is too large
            if risk > entry_price * 0.02:  # Risk > 2% of price
                return signals
            
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
                    'volume_ratio': current_bar['volume'] / avg_volume,
                    'bb_width': df.iloc[-1].get('bb_width', 0)
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
            if pd.isna(atr) or atr < 8.0:  # Skip if ATR too low
                return signals
            
            # Calculate signal parameters
            entry_price = current_bar['close']
            stop_loss = resistance_level + (0.5 * atr)  # Above resistance
            
            # Dynamic take profit based on R-mult
            risk = stop_loss - entry_price
            take_profit = entry_price - (self.config.take_profit_r_mult * risk)
            
            # Calculate confidence
            confidence = self.config.base_confidence
            
            # Boost for strong EMA slope
            ema_slope = self._calculate_ema_slope(df, fast_ema_col)
            if ema_slope < -0.2:
                confidence += 0.1
            
            # Boost for volume
            if current_bar['volume'] > avg_volume * 1.5:
                confidence += 0.05
            
            # Boost for trending regime
            if 'adx' in current_bar and current_bar['adx'] > 25:
                confidence += 0.05
            
            # Cap confidence
            confidence = min(confidence, 0.9)
            
            # Calculate risk/reward
            risk_reward = self.config.take_profit_r_mult
            
            # Skip if risk is too large
            if risk > entry_price * 0.02:  # Risk > 2% of price
                return signals
            
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
                    'volume_ratio': current_bar['volume'] / avg_volume,
                    'bb_width': df.iloc[-1].get('bb_width', 0)
                }
            )
            
            signals.append(signal)
        
        return signals
    
    def _calculate_ema_slope(self, df: pd.DataFrame, ema_col: str) -> float:
        """Calculate the slope of EMA (percentage per bar)"""
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
        """Get session open price (first bar of the day)"""
        try:
            if 'timestamp' in df.columns:
                current_date = pd.to_datetime(df['timestamp']).dt.date.iloc[-1]
                session_bars = df[pd.to_datetime(df['timestamp']).dt.date == current_date]
                if not session_bars.empty:
                    return session_bars.iloc[0]['open']
            
            return df.iloc[0]['open']
        except:
            return df.iloc[-1]['close']