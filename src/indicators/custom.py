import pandas as pd
import numpy as np
from typing import Tuple, List, Optional, Dict
from datetime import datetime  # kept in case you want it later
import warnings
warnings.filterwarnings('ignore')

class CustomIndicators:
    """Custom technical indicators for NQ signal bot with session-aware VWAP"""
    
    @staticmethod
    def calculate_vwap(df: pd.DataFrame, session_start: str = "09:30") -> pd.Series:
        """Calculate session-based VWAP that properly resets at session start"""
        if 'volume' not in df.columns:
            raise ValueError("DataFrame must contain 'volume' column")
        
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['time_str'] = df['timestamp'].dt.strftime('%H:%M')
        
        # Find session start indices
        session_starts = df[df['time_str'] == session_start].index
        
        if len(session_starts) == 0:
            # No session start found, calculate cumulative VWAP
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            cum_tp_vol = (typical_price * df['volume']).cumsum()
            cum_vol = df['volume'].cumsum()
            return cum_tp_vol / cum_vol
        
        vwap = pd.Series(index=df.index, dtype=float)
        
        # Calculate VWAP for each session segment
        for i, idx in enumerate(session_starts):
            start_idx = idx
            end_idx = session_starts[i+1] if i+1 < len(session_starts) else df.index[-1]
            
            session_data = df.loc[start_idx:end_idx]
            typical_price = (session_data['high'] + session_data['low'] + session_data['close']) / 3
            cum_tp_vol = (typical_price * session_data['volume']).cumsum()
            cum_vol = session_data['volume'].cumsum()
            
            vwap.loc[start_idx:end_idx] = cum_tp_vol / cum_vol
        
        return vwap.ffill()
    
    @staticmethod
    def calculate_vwap_bands(df: pd.DataFrame, periods: int = 60, std_devs: List[float] = [1.0, 2.0]) -> pd.DataFrame:
        """Calculate VWAP bands using rolling standard deviation"""
        vwap = CustomIndicators.calculate_vwap(df)
        
        # Calculate typical price
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        
        # Rolling standard deviation (session-aware)
        df['time_str'] = pd.to_datetime(df['timestamp']).dt.strftime('%H:%M')
        session_starts = df[df['time_str'] == "09:30"].index
        
        rolling_std = pd.Series(index=df.index, dtype=float)
        
        for i, idx in enumerate(session_starts):
            start_idx = idx
            end_idx = session_starts[i+1] if i+1 < len(session_starts) else df.index[-1]
            session_data = df.loc[start_idx:end_idx]
            
            session_tp = typical_price.loc[start_idx:end_idx]
            rolling_std.loc[start_idx:end_idx] = session_tp.rolling(window=periods, min_periods=10).std()
        
        rolling_std = rolling_std.ffill()
        
        bands = pd.DataFrame({'vwap': vwap, 'typical_price': typical_price, 'rolling_std': rolling_std})
        
        for std in std_devs:
            bands[f'upper_band_{std}'] = vwap + (rolling_std * std)
            bands[f'lower_band_{std}'] = vwap - (rolling_std * std)
        
        return bands
    
    @staticmethod
    def find_pivot_points(df: pd.DataFrame, strength: int = 5) -> Tuple[List[int], List[int]]:
        """Find pivot high and low points using fractal method"""
        highs = []
        lows = []
        
        for i in range(strength, len(df) - strength):
            # Check for pivot high
            is_pivot_high = True
            for j in range(1, strength + 1):
                if (df.iloc[i]['high'] <= df.iloc[i - j]['high'] or 
                    df.iloc[i]['high'] <= df.iloc[i + j]['high']):
                    is_pivot_high = False
                    break
            
            if is_pivot_high:
                highs.append(i)
            
            # Check for pivot low
            is_pivot_low = True
            for j in range(1, strength + 1):
                if (df.iloc[i]['low'] >= df.iloc[i - j]['low'] or 
                    df.iloc[i]['low'] >= df.iloc[i + j]['low']):
                    is_pivot_low = False
                    break
            
            if is_pivot_low:
                lows.append(i)
        
        return highs, lows
    
    @staticmethod
    def detect_divergence(df: pd.DataFrame, indicator: pd.Series, 
                         min_bars: int = 5, max_bars: int = 120) -> List[Dict]:
        """Detect bullish and bearish divergences between price and indicator"""
        divergences = []
        
        # Find pivot points in price
        pivot_highs, pivot_lows = CustomIndicators.find_pivot_points(df, strength=5)
        
        # Find pivot points in indicator
        indicator_highs, indicator_lows = CustomIndicators.find_pivot_points(
            pd.DataFrame({'high': indicator, 'low': indicator}), strength=5
        )
        
        # Check for bullish divergence (price lower low, indicator higher low)
        for i in range(len(pivot_lows) - 1):
            if i >= len(indicator_lows) - 1:
                break
                
            price_low1 = df.iloc[pivot_lows[i]]['low']
            price_low2 = df.iloc[pivot_lows[i + 1]]['low']
            
            if pivot_lows[i + 1] < len(indicator) and pivot_lows[i] < len(indicator):
                ind_low1 = indicator.iloc[indicator_lows[i]] if i < len(indicator_lows) else indicator.iloc[pivot_lows[i]]
                ind_low2 = indicator.iloc[indicator_lows[i + 1]] if i + 1 < len(indicator_lows) else indicator.iloc[pivot_lows[i + 1]]
                
                # Validate divergence exists
                if (price_low2 < price_low1 and ind_low2 > ind_low1 and 
                    pivot_lows[i + 1] - pivot_lows[i] <= max_bars and
                    pivot_lows[i + 1] - pivot_lows[i] >= min_bars):
                    
                    # Calculate strength
                    strength_val = abs(ind_low2 - ind_low1) / ind_low1 if ind_low1 != 0 else 0
                    
                    divergences.append({
                        'type': 'bullish',
                        'price_index': pivot_lows[i + 1],
                        'price_value': price_low2,
                        'indicator_value': ind_low2,
                        'strength': strength_val
                    })
        
        # Check for bearish divergence (price higher high, indicator lower high)
        for i in range(len(pivot_highs) - 1):
            if i >= len(indicator_highs) - 1:
                break
                
            price_high1 = df.iloc[pivot_highs[i]]['high']
            price_high2 = df.iloc[pivot_highs[i + 1]]['high']
            
            if pivot_highs[i + 1] < len(indicator) and pivot_highs[i] < len(indicator):
                ind_high1 = indicator.iloc[indicator_highs[i]] if i < len(indicator_highs) else indicator.iloc[pivot_highs[i]]
                ind_high2 = indicator.iloc[indicator_highs[i + 1]] if i + 1 < len(indicator_highs) else indicator.iloc[pivot_highs[i + 1]]
                
                if (price_high2 > price_high1 and ind_high2 < ind_high1 and 
                    pivot_highs[i + 1] - pivot_highs[i] <= max_bars and
                    pivot_highs[i + 1] - pivot_highs[i] >= min_bars):
                    
                    strength_val = abs(ind_high1 - ind_high2) / ind_high1 if ind_high1 != 0 else 0
                    
                    divergences.append({
                        'type': 'bearish',
                        'price_index': pivot_highs[i + 1],
                        'price_value': price_high2,
                        'indicator_value': ind_high2,
                        'strength': strength_val
                    })
        
        return divergences
    
    @staticmethod
    def calculate_supertrend(df: pd.DataFrame, atr_period: int = 10, multiplier: float = 3.0) -> pd.Series:
        """Calculate Supertrend indicator (reliable manual implementation)"""
        # Calculate True Range
        high_low = df['high'] - df['low']
        high_close_prev = np.abs(df['high'] - df['close'].shift(1))
        low_close_prev = np.abs(df['low'] - df['close'].shift(1))
        
        true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        atr = true_range.rolling(window=atr_period).mean()
        
        # Calculate basic bands
        basic_upper_band = (df['high'] + df['low']) / 2 + (multiplier * atr)
        basic_lower_band = (df['high'] + df['low']) / 2 - (multiplier * atr)
        
        # Initialize final bands
        final_upper_band = pd.Series(index=df.index, dtype=float)
        final_lower_band = pd.Series(index=df.index, dtype=float)
        supertrend = pd.Series(index=df.index, dtype=float)
        trend_direction = pd.Series(index=df.index, dtype=int)  # 1 = up, -1 = down
        
        for i in range(len(df)):
            if i == 0:
                final_upper_band.iloc[i] = basic_upper_band.iloc[i]
                final_lower_band.iloc[i] = basic_lower_band.iloc[i]
                supertrend.iloc[i] = basic_upper_band.iloc[i]
                trend_direction.iloc[i] = 1
            else:
                # Upper band logic
                if basic_upper_band.iloc[i] > final_upper_band.iloc[i-1]:
                    final_upper_band.iloc[i] = basic_upper_band.iloc[i]
                else:
                    final_upper_band.iloc[i] = final_upper_band.iloc[i-1]
                
                # Lower band logic
                if basic_lower_band.iloc[i] < final_lower_band.iloc[i-1]:
                    final_lower_band.iloc[i] = basic_lower_band.iloc[i]
                else:
                    final_lower_band.iloc[i] = final_lower_band.iloc[i-1]
                
                # Trend direction and supertrend
                if trend_direction.iloc[i-1] == 1:  # Previous uptrend
                    if df['close'].iloc[i] <= final_lower_band.iloc[i]:
                        trend_direction.iloc[i] = -1
                        supertrend.iloc[i] = final_upper_band.iloc[i]
                    else:
                        trend_direction.iloc[i] = 1
                        supertrend.iloc[i] = final_lower_band.iloc[i]
                else:  # Previous downtrend
                    if df['close'].iloc[i] >= final_upper_band.iloc[i]:
                        trend_direction.iloc[i] = 1
                        supertrend.iloc[i] = final_lower_band.iloc[i]
                    else:
                        trend_direction.iloc[i] = -1
                        supertrend.iloc[i] = final_upper_band.iloc[i]
        
        return supertrend
    
    @staticmethod
    def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average Directional Index (ADX)"""
        # Calculate True Range
        high_low = df['high'] - df['low']
        high_close_prev = np.abs(df['high'] - df['close'].shift(1))
        low_close_prev = np.abs(df['low'] - df['close'].shift(1))
        
        true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        
        # Calculate +DM and -DM
        plus_dm = df['high'].diff()
        minus_dm = -df['low'].diff()
        
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        
        # Smooth using Wilder's method
        atr_smooth = true_range.ewm(alpha=1/period).mean()
        plus_dm_smooth = plus_dm.ewm(alpha=1/period).mean()
        minus_dm_smooth = minus_dm.ewm(alpha=1/period).mean()
        
        # Calculate +DI and -DI
        plus_di = 100 * (plus_dm_smooth / atr_smooth)
        minus_di = 100 * (minus_dm_smooth / atr_smooth)
        
        # Calculate DX and ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.ewm(alpha=1/period).mean()
        
        return adx
