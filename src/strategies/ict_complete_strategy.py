"""
================================================================================
ICT (Inner Circle Trader) COMPLETE STRATEGY + ML FEATURE FACTORY
================================================================================

Roles of this module:
1. Compute ICT-style context per bar:
   - Market structure (bias, HH/HL vs LH/LL, support/resistance)
   - Order block context
   - Liquidity (equal highs/lows, sweeps, FVGs)
   - Session / killzones
   - Technicals (RSI, ATR, EMAs, momentum, etc.)

2. Generate ICTSignals with full contextual payload.

3. Build a feature matrix from ICTSignals suitable for ML.

4. Provide ICTMLTrainer to:
   - Label signals with "did +2R happen before -1R within N bars?"
   - Train a classifier to estimate P(win | features).
   - Save the trained model for use by your live engine.

Author: You + GPT (feature factory / ML wrapper)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import datetime, time
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from loguru import logger

# ==============================================================================
# CORE DATA STRUCTURES
# ==============================================================================

@dataclass
class ICTSignal:
    """Complete ICT trading signal with all institutional context."""
    timestamp: datetime
    signal_type: str  # 'long' or 'short'
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: float

    # ICT-specific context
    order_block_info: Dict
    market_structure: Dict
    session_info: Dict
    liquidity_context: Dict

    # Technical analysis
    rsi: float
    atr: float
    volume_ratio: float

    # Multi-timeframe context
    higher_tf_bias: str
    momentum_score: float


@dataclass
class ICTConfig:
    """ICT strategy configuration parameters (and ML defaults)."""

    # Risk Management
    risk_per_trade: float = 0.02  # 2% risk per trade
    max_portfolio_risk: float = 0.06  # 6% max total risk
    min_rr_ratio: float = 2.0       # Minimum 2:1 R:R
    target_rr_ratio: float = 2.5    # Target R:R used for TP
    commission: float = 0.0         # Per-trade commission fraction

    # Technical Indicators
    ema_fast_period: int = 20
    ema_slow_period: int = 50
    ema_trend_period: int = 200
    atr_period: int = 14
    rsi_period: int = 14

    # Market Structure
    swing_strength: int = 5
    ob_lookback: int = 50
    ob_min_strength: float = 0.005  # 0.5% minimum move

    # Session Timing (EST)
    london_open: time = time(3, 0)
    london_close: time = time(12, 0)
    ny_open: time = time(8, 0)
    ny_close: time = time(17, 0)
    asian_open: time = time(20, 0)
    asian_close: time = time(4, 0)

    # Killzones (EST)
    killzone_am_start: time = time(9, 30)
    killzone_am_end: time = time(11, 30)
    killzone_pm_start: time = time(13, 0)
    killzone_pm_end: time = time(15, 30)

    # Execution / Filters
    max_spread: float = 2.0         # Max spread in points
    min_volume_ratio: float = 0.8   # Min volume vs avg
    min_atr: float = 5.0            # Min ATR for trading

    # Confidence Scoring (baseline, *before* ML overrides)
    base_confidence: float = 0.6
    min_confidence: float = 0.75
    max_confidence: float = 0.95

    # ML / labeling defaults
    label_rr_win: float = 2.0       # +2R target
    label_rr_loss: float = 1.0      # -1R stop
    label_horizon_bars: int = 36    # e.g. next 36 bars (~3h on 5m)


# ==============================================================================
# MARKET STRUCTURE ANALYZER
# ==============================================================================

class ICTMarketStructureAnalyzer:
    """Analyzes market structure using ICT methodology."""

    def __init__(self, config: ICTConfig):
        self.config = config

    def find_swing_points(self, df: pd.DataFrame, strength: Optional[int] = None
                          ) -> Tuple[List[int], List[int]]:
        """Find swing highs and lows using ICT strength parameters."""
        if strength is None:
            strength = self.config.swing_strength

        highs, lows = [], []

        for i in range(strength, len(df) - strength):
            # Swing high
            if (all(df['high'].iloc[i] >= df['high'].iloc[i - j] for j in range(1, strength + 1)) and
                all(df['high'].iloc[i] >= df['high'].iloc[i + j] for j in range(1, strength + 1))):
                highs.append(i)

            # Swing low
            if (all(df['low'].iloc[i] <= df['low'].iloc[i - j] for j in range(1, strength + 1)) and
                all(df['low'].iloc[i] <= df['low'].iloc[i + j] for j in range(1, strength + 1))):
                lows.append(i)

        return highs, lows

    def determine_market_bias(self, df: pd.DataFrame) -> str:
        """Determine market bias using swings + EMAs."""
        if len(df) < self.config.ema_trend_period:
            return 'neutral'

        close = df['close']
        ema_fast = close.ewm(span=self.config.ema_fast_period).mean().iloc[-1]
        ema_slow = close.ewm(span=self.config.ema_slow_period).mean().iloc[-1]
        ema_trend = close.ewm(span=self.config.ema_trend_period).mean().iloc[-1]
        current_price = close.iloc[-1]

        recent_data = df.tail(100)
        highs, lows = self.find_swing_points(recent_data)

        if len(highs) < 2 or len(lows) < 2:
            return 'neutral'

        last_high_idx, prev_high_idx = highs[-1], highs[-2]
        last_low_idx, prev_low_idx = lows[-1], lows[-2]

        last_high = recent_data.iloc[last_high_idx]['high']
        prev_high = recent_data.iloc[prev_high_idx]['high']
        last_low = recent_data.iloc[last_low_idx]['low']
        prev_low = recent_data.iloc[prev_low_idx]['low']

        higher_highs = last_high > prev_high
        higher_lows = last_low > prev_low
        lower_highs = last_high < prev_high
        lower_lows = last_low < prev_low

        above_fast = current_price > ema_fast
        above_slow = current_price > ema_slow
        above_trend = current_price > ema_trend
        below_fast = current_price < ema_fast
        below_slow = current_price < ema_slow

        if higher_highs and higher_lows and above_fast and above_slow and above_trend:
            return 'strong_bullish'
        if higher_highs and higher_lows and above_fast:
            return 'bullish'
        if lower_highs and lower_lows and below_fast and below_slow and current_price < ema_trend:
            return 'strong_bearish'
        if lower_highs and lower_lows and below_fast:
            return 'bearish'
        return 'neutral'

    def find_support_resistance_levels(self, df: pd.DataFrame, lookback: int = 50
                                       ) -> Dict[str, List[float]]:
        """Find key support and resistance levels from swings."""
        recent_data = df.tail(lookback)
        highs, lows = self.find_swing_points(recent_data)

        resistance_levels = [recent_data.iloc[i]['high'] for i in highs]
        support_levels = [recent_data.iloc[i]['low'] for i in lows]

        atr = df['atr'].iloc[-1] if 'atr' in df.columns else (df['high'] - df['low']).rolling(14).mean().iloc[-1]
        tolerance = atr * 0.5

        resistance_levels = self._group_price_levels(resistance_levels, tolerance)
        support_levels = self._group_price_levels(support_levels, tolerance)

        return {
            'resistance': sorted(resistance_levels, reverse=True)[:3],
            'support': sorted(support_levels)[:3]
        }

    def _group_price_levels(self, levels: List[float], tolerance: float) -> List[float]:
        if not levels:
            return []
        levels = sorted(levels)
        grouped = [levels[0]]
        for level in levels[1:]:
            if abs(level - grouped[-1]) > tolerance:
                grouped.append(level)
        return grouped


# ==============================================================================
# ORDER BLOCK ANALYZER
# ==============================================================================

class ICTOrderBlockAnalyzer:
    """Identifies and analyzes ICT order blocks."""

    def __init__(self, config: ICTConfig):
        self.config = config

    def find_order_blocks(self, df: pd.DataFrame, bias: str) -> List[Dict]:
        if len(df) < self.config.ob_lookback:
            return []

        recent = df.tail(self.config.ob_lookback)
        order_blocks: List[Dict] = []

        for i in range(10, len(recent) - 15):
            candle = recent.iloc[i]
            if i + 15 >= len(recent):
                break
            start_price = candle['close']
            end_price = recent.iloc[i + 15]['close']
            move_pct = abs((end_price - start_price) / start_price)

            if move_pct <= self.config.ob_min_strength:
                continue

            is_bearish_ob = (candle['close'] < candle['open'] and bias in ['bullish', 'strong_bullish'])
            is_bullish_ob = (candle['close'] > candle['open'] and bias in ['bearish', 'strong_bearish'])

            if not (is_bearish_ob or is_bullish_ob):
                continue

            zone_high = max(candle['open'], candle['close'])
            zone_low = min(candle['open'], candle['close'])
            ob_type = 'bearish' if is_bearish_ob else 'bullish'

            order_blocks.append({
                'index': i,
                'type': ob_type,
                'zone_high': float(zone_high),
                'zone_low': float(zone_low),
                'candle_high': float(candle['high']),
                'candle_low': float(candle['low']),
                'strength': float(move_pct),
                'timestamp': recent.index[i]
            })

        order_blocks.sort(key=lambda x: x['strength'], reverse=True)
        return order_blocks[:3]

    def is_price_in_order_block(self, price: float, ob: Dict, buffer_atr: float = 0.0) -> bool:
        zone_low = ob['zone_low'] - buffer_atr
        zone_high = ob['zone_high'] + buffer_atr
        if zone_low <= price <= zone_high:
            return True
        near_zone = (
            abs(price - ob['zone_high']) <= buffer_atr * 0.5 or
            abs(price - ob['zone_low']) <= buffer_atr * 0.5
        )
        return near_zone

    def calculate_ob_confluence(self, price: float, obs: List[Dict], atr: float) -> float:
        if not obs or atr <= 0:
            return 0.0
        confluence = 0.0
        buffer_atr = atr * 0.2
        for ob in obs:
            if not self.is_price_in_order_block(price, ob, buffer_atr):
                continue
            center = (ob['zone_high'] + ob['zone_low']) / 2
            distance = abs(price - center)
            max_distance = atr * 0.5
            if distance <= max_distance:
                confluence += (1.0 - distance / max_distance) * ob['strength']
        return float(min(confluence, 1.0))


# ==============================================================================
# LIQUIDITY ANALYZER
# ==============================================================================

class ICTLiquidityAnalyzer:
    """Analyzes liquidity pools and imbalances."""

    def __init__(self, config: ICTConfig):
        self.config = config

    def find_liquidity_pools(self, df: pd.DataFrame, lookback: int = 30) -> Dict[str, List[float]]:
        recent = df.tail(lookback)
        current_price = df['close'].iloc[-1]
        atr = df['atr'].iloc[-1] if 'atr' in df.columns else (df['high'] - df['low']).rolling(14).mean().iloc[-1]
        threshold = atr * 0.3

        highs_above = []
        lows_below = []

        for i in range(len(recent) - 1):
            for j in range(i + 1, len(recent)):
                if abs(recent.iloc[i]['high'] - recent.iloc[j]['high']) < threshold:
                    if recent.iloc[i]['high'] > current_price:
                        highs_above.append(float(recent.iloc[i]['high']))
                if abs(recent.iloc[i]['low'] - recent.iloc[j]['low']) < threshold:
                    if recent.iloc[i]['low'] < current_price:
                        lows_below.append(float(recent.iloc[i]['low']))

        return {
            'above': list(sorted(set(highs_above))),
            'below': list(sorted(set(lows_below)))
        }

    def detect_fair_value_gaps(self, df: pd.DataFrame) -> List[Dict]:
        if len(df) < 5:
            return []
        atr_series = df['atr'] if 'atr' in df.columns else (df['high'] - df['low']).rolling(14).mean()
        imbalances: List[Dict] = []
        for i in range(2, len(df) - 2):
            candles = df.iloc[i-2:i+3]
            atr_val = atr_series.iloc[i]
            if pd.isna(atr_val):
                continue

            # Bearish FVG (gap down)
            if (candles.iloc[1]['close'] < candles.iloc[1]['open'] and
                candles.iloc[2]['close'] > candles.iloc[2]['open'] and
                candles.iloc[3]['close'] < candles.iloc[3]['open']):
                gap = candles.iloc[2]['low'] - candles.iloc[1]['high']
                if gap > atr_val * 0.3:
                    imbalances.append({
                        'type': 'bearish',
                        'index': i,
                        'gap_size': float(gap)
                    })

            # Bullish FVG (gap up)
            if (candles.iloc[1]['close'] > candles.iloc[1]['open'] and
                candles.iloc[2]['close'] < candles.iloc[2]['open'] and
                candles.iloc[3]['close'] > candles.iloc[3]['open']):
                gap = candles.iloc[1]['high'] - candles.iloc[2]['low']
                if gap > atr_val * 0.3:
                    imbalances.append({
                        'type': 'bullish',
                        'index': i,
                        'gap_size': float(gap)
                    })

        return imbalances

    def detect_liquidity_sweep(self, df: pd.DataFrame) -> bool:
        if len(df) < 10:
            return False
        recent = df.tail(10)
        pools = self.find_liquidity_pools(recent)
        current_high = recent['high'].iloc[-1]
        current_low = recent['low'].iloc[-1]

        for resistance in pools['above']:
            if recent['high'].iloc[-2] < resistance <= current_high:
                return True
        for support in pools['below']:
            if recent['low'].iloc[-2] > support >= current_low:
                return True
        return False


# ==============================================================================
# SESSION MANAGER
# ==============================================================================

class ICTSessionManager:
    """Handles sessions and killzones."""

    def __init__(self, config: ICTConfig):
        self.config = config

    def get_current_session(self, current_time: time) -> str:
        if current_time >= self.config.asian_open or current_time <= self.config.asian_close:
            return 'asian'
        if self.config.london_open <= current_time <= self.config.london_close:
            return 'london'
        if self.config.ny_open <= current_time <= self.config.ny_close:
            return 'new_york'
        return 'transition'

    def is_in_killzone(self, current_time: time) -> bool:
        in_am = self.config.killzone_am_start <= current_time <= self.config.killzone_am_end
        in_pm = self.config.killzone_pm_start <= current_time <= self.config.killzone_pm_end
        return in_am or in_pm

    def get_session_score(self, session: str) -> float:
        return {
            'london': 0.8,
            'new_york': 1.0,
            'asian': 0.6,
            'transition': 0.4
        }.get(session, 0.4)

    def get_killzone_score(self, in_killzone: bool) -> float:
        return 0.15 if in_killzone else 0.0


# ==============================================================================
# TECHNICAL ANALYZER
# ==============================================================================

class ICTTechnicalAnalyzer:
    """Technical analysis using ICT methodology"""

    def __init__(self, config: ICTConfig):
        self.config = config

    def calculate_rsi(self, prices: pd.Series, period: int = None) -> float:
        """Calculate RSI with ICT parameters (single value)."""
        if period is None:
            period = self.config.rsi_period

        if len(prices) < period + 1:
            return 50.0

        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        value = rsi.iloc[-1]
        return float(value) if not pd.isna(value) else 50.0

    def calculate_momentum_score(self, df: pd.DataFrame, lookback: int = 10) -> float:
        """
        Calculate a momentum score in roughly [-1, 1].
        Positive = bullish momentum, negative = bearish.
        """
        if len(df) < lookback + 5:
            return 0.0

        recent_data = df.tail(lookback + 5)
        price_changes = recent_data["close"].pct_change()

        # Last `lookback` bars vs earlier 5 bars
        recent_momentum = price_changes.tail(lookback).mean()
        previous_momentum = price_changes.tail(lookback + 5).head(5).mean()

        if previous_momentum is None or np.isnan(previous_momentum) or previous_momentum == 0:
            return 0.0

        ratio = recent_momentum / abs(previous_momentum)
        # Clamp to [-1, 1]
        return float(max(-1.0, min(1.0, ratio)))

    def analyze_volume_profile(self, df: pd.DataFrame, lookback: int = 20) -> Dict:
        """
        Analyze volume profile for confluence (VWAP/high-/low-volume zones).
        Returns a dict with vwap, high_volume_nodes, low_volume_nodes.
        """
        if len(df) < lookback or "volume" not in df.columns:
            return {"vwap": None, "high_volume_nodes": [], "low_volume_nodes": []}

        recent_data = df.tail(lookback)

        typical_price = (recent_data["high"] + recent_data["low"] + recent_data["close"]) / 3.0
        vwap = float((typical_price * recent_data["volume"]).sum() / recent_data["volume"].sum())

        price_levels = recent_data["close"].values
        high_vol_threshold = float(np.percentile(price_levels, 75))
        low_vol_threshold = float(np.percentile(price_levels, 25))

        return {
            "vwap": vwap,
            "high_volume_nodes": [high_vol_threshold, float(np.percentile(price_levels, 90))],
            "low_volume_nodes": [low_vol_threshold, float(np.percentile(price_levels, 10))],
        }

    def get_volume_confluence(
        self,
        df: pd.DataFrame,
        signal_time: Optional[datetime] = None,
    ) -> float:
        """
        Compute a volume confluence score (0-1) for the signal bar.

        df: window of data up to and including the signal bar.
        signal_time: optional timestamp of the signal bar. If None,
                     the last row of df is treated as the signal bar.
        """
        # If no volume data or too few bars, neutral
        if "volume" not in df.columns or len(df) < 5:
            return 0.5

        # --- Find the signal bar safely ---
        if signal_time is not None:
            try:
                sel = df.loc[signal_time]
            except KeyError:
                # Fallback: last row
                signal_bar = df.iloc[-1]
            else:
                # sel can be a Series (single row) or DataFrame (multiple rows)
                if isinstance(sel, pd.DataFrame):
                    signal_bar = sel.iloc[-1]
                else:
                    signal_bar = sel
        else:
            signal_bar = df.iloc[-1]

        # --- Get current_volume as a scalar ---
        vol_value = signal_bar["volume"]
        if isinstance(vol_value, (pd.Series, np.ndarray, list)):
            # Take the last element if it's a collection
            vol_array = np.array(vol_value)
            if vol_array.size == 0:
                return 0.5
            current_volume = float(vol_array[-1])
        else:
            try:
                current_volume = float(vol_value)
            except (TypeError, ValueError):
                return 0.5

        # --- Compute average volume over last N bars ---
        lookback = min(20, len(df))
        avg_volume = df["volume"].tail(lookback).mean()

        if avg_volume <= 0 or pd.isna(avg_volume) or pd.isna(current_volume):
            return 0.5

        volume_ratio = current_volume / avg_volume

        # Map volume ratio to [0,1], centered around min_volume_ratio
        base = self.config.min_volume_ratio
        if base <= 0:
            base = 1.0

        if volume_ratio >= base:
            score = min(1.0, volume_ratio / (2.0 * base))
        else:
            score = max(0.0, volume_ratio / (2.0 * base))

        return float(score)




# ==============================================================================
# CONFIDENCE SCORER (baseline, before ML)
# ==============================================================================

class ICTConfidenceScorer:
    """Rule-based confidence score (can be overridden by ML)."""

    def __init__(self, config: ICTConfig):
        self.config = config

    def calculate_signal_confidence(self, ctx: Dict) -> float:
        c = self.config.base_confidence

        # Market structure (30%)
        c += self._score_market_structure(ctx) * 0.30

        # OB confluence (25%)
        ob_score = float(ctx.get('ob_confluence', 0.0))
        c += ob_score * 0.25

        # Session timing (20%)
        sess = float(ctx.get('session_score', 0.5))
        kz = float(ctx.get('killzone_score', 0.0))
        c += (sess + kz) * 0.20

        # Technicals (15%)
        rsi_score = self._score_rsi(float(ctx.get('rsi', 50.0)), ctx.get('bias', 'neutral'))
        mom_score = (float(ctx.get('momentum_score', 0.0)) + 1.0) / 2.0
        c += ((rsi_score + mom_score) / 2.0) * 0.15

        # Volume (10%)
        vol_score = float(ctx.get('volume_confluence', 0.5))
        c += vol_score * 0.10

        c = max(self.config.min_confidence, min(self.config.max_confidence, c))
        return float(c)

    def _score_market_structure(self, ctx: Dict) -> float:
        bias = ctx.get('bias', 'neutral')
        s_type = ctx.get('signal_type', 'long')
        if ((bias in ['strong_bullish', 'bullish'] and s_type == 'long') or
            (bias in ['strong_bearish', 'bearish'] and s_type == 'short')):
            return 1.0
        if bias in ['bullish', 'bearish']:
            return 0.6
        return 0.2

    def _score_rsi(self, rsi: float, bias: str) -> float:
        if bias in ['strong_bullish', 'bullish']:
            if rsi < 30:
                return 1.0
            if rsi < 50:
                return 0.7
            if rsi < 70:
                return 0.4
            return 0.1
        if bias in ['strong_bearish', 'bearish']:
            if rsi > 70:
                return 1.0
            if rsi > 50:
                return 0.7
            if rsi > 30:
                return 0.4
            return 0.1
        return 0.5


# ==============================================================================
# MAIN ICT STRATEGY (feature factory)
# ==============================================================================

class ICTStrategy:
    """
    Complete ICT strategy used as a *feature factory*.

    - prepare_data(df) -> df with ATR, EMAs, RSI
    - generate_signals(df) -> List[ICTSignal] (each with full context)
    """

    def __init__(self, config: Optional[ICTConfig] = None):
        self.config = config or ICTConfig()

        self.market_analyzer = ICTMarketStructureAnalyzer(self.config)
        self.ob_analyzer = ICTOrderBlockAnalyzer(self.config)
        self.liquidity_analyzer = ICTLiquidityAnalyzer(self.config)
        self.session_manager = ICTSessionManager(self.config)
        self.technical_analyzer = ICTTechnicalAnalyzer(self.config)
        self.confidence_scorer = ICTConfidenceScorer(self.config)

    # ---------- Data prep ----------

    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        high, low, close = df['high'], df['low'], df['close']
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr'] = tr.rolling(self.config.atr_period).mean()

        df['ema_fast'] = close.ewm(span=self.config.ema_fast_period).mean()
        df['ema_slow'] = close.ewm(span=self.config.ema_slow_period).mean()
        df['ema_trend'] = close.ewm(span=self.config.ema_trend_period).mean()

        # RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.config.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.config.rsi_period).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        return df

    # ---------- Signal generation ----------

    def generate_signals(self, df: pd.DataFrame) -> List[ICTSignal]:
        df = self.prepare_data(df)
        signals: List[ICTSignal] = []

        min_bars = max(self.config.ema_trend_period, self.config.ob_lookback) + 50
        if len(df) < min_bars:
            logger.warning(f"[ICT] Not enough data for signals. Need {min_bars}, have {len(df)}.")
            return signals

        for i in range(min_bars, len(df)):
            if not self._check_ema_crossover(df, i):
                continue

            historical = df.iloc[:i+1]
            current_bar = df.iloc[i]
            signal = self._create_ict_signal(historical, current_bar)
            if signal:
                signals.append(signal)
                logger.info(f"[ICT] {signal.signal_type.upper()} @ {signal.entry_price:.2f} {signal.timestamp}")

        return signals

    def _check_ema_crossover(self, df: pd.DataFrame, idx: int) -> bool:
        if idx < 2:
            return False
        cf, cs = df['ema_fast'].iloc[idx], df['ema_slow'].iloc[idx]
        pf, ps = df['ema_fast'].iloc[idx-1], df['ema_slow'].iloc[idx-1]
        if any(pd.isna(v) for v in [cf, cs, pf, ps]):
            return False
        golden = (pf <= ps and cf > cs)
        death = (pf >= ps and cf < cs)
        return golden or death

    def _create_ict_signal(self, df: pd.DataFrame, current_bar: pd.Series
                           ) -> Optional[ICTSignal]:
        cf, cs = current_bar['ema_fast'], current_bar['ema_slow']
        prev_fast = df['ema_fast'].iloc[-2]
        prev_slow = df['ema_slow'].iloc[-2]

        golden = (prev_fast <= prev_slow and cf > cs)
        signal_type = 'long' if golden else 'short'

        entry_price = float(current_bar['close'])
        atr = float(current_bar['atr']) if not pd.isna(current_bar['atr']) else np.nan
        if pd.isna(atr) or atr < self.config.min_atr:
            return None

        if signal_type == 'long':
            stop_loss = entry_price - atr * 1.5
            risk = entry_price - stop_loss
            take_profit = entry_price + risk * self.config.target_rr_ratio
        else:
            stop_loss = entry_price + atr * 1.5
            risk = stop_loss - entry_price
            take_profit = entry_price - risk * self.config.target_rr_ratio

        if risk <= 0:
            return None

        # Market structure
        bias = self.market_analyzer.determine_market_bias(df.tail(200))
        key_levels = self.market_analyzer.find_support_resistance_levels(df.tail(100))

        # OBs
        obs = self.ob_analyzer.find_order_blocks(df.tail(self.config.ob_lookback), bias)
        ob_confluence = self.ob_analyzer.calculate_ob_confluence(entry_price, obs, atr)

        # Liquidity
        liq_levels = self.liquidity_analyzer.find_liquidity_pools(df.tail(60))
        imbalances = self.liquidity_analyzer.detect_fair_value_gaps(df.tail(40))
        sweep = self.liquidity_analyzer.detect_liquidity_sweep(df.tail(40))

        # Session / killzone
        ts = current_bar.name
        current_time = ts.time()
        session = self.session_manager.get_current_session(current_time)
        in_killzone = self.session_manager.is_in_killzone(current_time)
        session_score = self.session_manager.get_session_score(session)
        killzone_score = self.session_manager.get_killzone_score(in_killzone)

        # Technical
        rsi_val = float(current_bar['rsi']) if not pd.isna(current_bar['rsi']) else 50.0
        momentum_score = self.technical_analyzer.calculate_momentum_score(df.tail(60))
        volume_conf = self.technical_analyzer.get_volume_confluence(df, ts)

        # Higher TF bias via EMAs
        higher_tf_bias = self._get_higher_timeframe_bias(df)

        # Context for confidence
        ctx = {
            'bias': bias,
            'signal_type': signal_type,
            'ob_confluence': ob_confluence,
            'session_score': session_score,
            'killzone_score': killzone_score,
            'rsi': rsi_val,
            'momentum_score': momentum_score,
            'volume_confluence': volume_conf
        }
        confidence = self.confidence_scorer.calculate_signal_confidence(ctx)

        return ICTSignal(
            timestamp=ts,
            signal_type=signal_type,
            entry_price=entry_price,
            stop_loss=float(stop_loss),
            take_profit=float(take_profit),
            confidence=float(confidence),

            order_block_info={
                'confluence': float(ob_confluence),
                'blocks': obs[:2]
            },
            market_structure={
                'bias': bias,
                'key_levels': key_levels
            },
            session_info={
                'session': session,
                'in_killzone': in_killzone,
                'session_score': float(session_score),
                'killzone_score': float(killzone_score)
            },
            liquidity_context={
                'levels': liq_levels,
                'imbalances': imbalances,
                'sweep_detected': sweep
            },

            rsi=rsi_val,
            atr=atr,
            volume_ratio=volume_conf,

            higher_tf_bias=higher_tf_bias,
            momentum_score=momentum_score
        )

    def _get_higher_timeframe_bias(self, df: pd.DataFrame) -> str:
        if len(df) < 200:
            return 'neutral'
        ema50 = df['close'].ewm(span=50).mean().iloc[-1]
        ema200 = df['close'].ewm(span=200).mean().iloc[-1]
        price = df['close'].iloc[-1]
        if price > ema50 > ema200:
            return 'bullish'
        if price < ema50 < ema200:
            return 'bearish'
        return 'neutral'


# ==============================================================================
# FEATURE BUILDER (for ML)
# ==============================================================================

class ICTFeatureBuilder:
    """
    Turns ICTSignals + OHLCV history into numeric feature dicts.
    """

    def __init__(self, config: ICTConfig):
        self.config = config

    def build_features_for_signal(self, df: pd.DataFrame, signal: ICTSignal) -> Dict[str, float]:
        """Build a flat feature dict for one ICTSignal."""
        ts = signal.timestamp
        try:
            idx = df.index.get_loc(ts)
        except KeyError:
            return {}

        bar = df.iloc[idx]
        price = float(bar['close'])
        atr = float(bar['atr']) if 'atr' in bar and not pd.isna(bar['atr']) else 1.0
        if atr <= 0:
            atr = 1.0

        # --- Market structure ---
        bias = signal.market_structure.get('bias', 'neutral')
        levels = signal.market_structure.get('key_levels', {})
        supports = levels.get('support', [])
        resistances = levels.get('resistance', [])

        nearest_support = min(supports, key=lambda x: abs(price - x)) if supports else None
        nearest_resistance = min(resistances, key=lambda x: abs(price - x)) if resistances else None

        dist_support_atr = abs(price - nearest_support) / atr if nearest_support is not None else 10.0
        dist_res_atr = abs(price - nearest_resistance) / atr if nearest_resistance is not None else 10.0

        # --- Order block context ---
        ob_conf = float(signal.order_block_info.get('confluence', 0.0))
        blocks = signal.order_block_info.get('blocks', []) or []
        num_obs = len(blocks)
        nearest_ob_dist_atr = 10.0
        ob_type_one_hot = {'ob_bullish': 0.0, 'ob_bearish': 0.0}

        for ob in blocks:
            center = (ob['zone_high'] + ob['zone_low']) / 2.0
            d = abs(price - center) / atr
            if d < nearest_ob_dist_atr:
                nearest_ob_dist_atr = d
            if ob.get('type') == 'bullish':
                ob_type_one_hot['ob_bullish'] = 1.0
            if ob.get('type') == 'bearish':
                ob_type_one_hot['ob_bearish'] = 1.0

        # --- Liquidity context ---
        liq = signal.liquidity_context.get('levels', {})
        highs_above = liq.get('above', []) or []
        lows_below = liq.get('below', []) or []

        num_highs_above = float(len(highs_above))
        num_lows_below = float(len(lows_below))

        fvgs = signal.liquidity_context.get('imbalances', []) or []
        has_bull_fvg = any(x.get('type') == 'bullish' for x in fvgs)
        has_bear_fvg = any(x.get('type') == 'bearish' for x in fvgs)
        num_fvgs = float(len(fvgs))
        sweep_flag = float(bool(signal.liquidity_context.get('sweep_detected', False)))

        # --- Session / Killzone ---
        sess_info = signal.session_info
        session = sess_info.get('session', 'unknown')
        in_kz = bool(sess_info.get('in_killzone', False))
        session_one_hot = {
            'session_london': float(session == 'london'),
            'session_new_york': float(session == 'new_york'),
            'session_asian': float(session == 'asian'),
            'session_transition': float(session == 'transition')
        }

        # --- Technicals ---
        ema_trend = float(bar.get('ema_trend', price))
        dist_ema_trend_atr = abs(price - ema_trend) / atr

        rr_ratio = abs(signal.take_profit - signal.entry_price) / max(
            abs(signal.entry_price - signal.stop_loss), 1e-6
        )

        features: Dict[str, float] = {
            # Identity / direction
            'signal_long': float(signal.signal_type == 'long'),
            'signal_short': float(signal.signal_type == 'short'),

            # Bias one-hot
            'bias_strong_bullish': float(bias == 'strong_bullish'),
            'bias_bullish': float(bias == 'bullish'),
            'bias_strong_bearish': float(bias == 'strong_bearish'),
            'bias_bearish': float(bias == 'bearish'),
            'bias_neutral': float(bias == 'neutral'),

            # Support/resistance
            'dist_support_atr': float(dist_support_atr),
            'dist_resistance_atr': float(dist_res_atr),

            # Order block
            'ob_confluence': float(ob_conf),
            'num_order_blocks': float(num_obs),
            'dist_nearest_ob_atr': float(nearest_ob_dist_atr),
            **ob_type_one_hot,

            # Liquidity
            'liquidity_highs_above': num_highs_above,
            'liquidity_lows_below': num_lows_below,
            'num_fvgs': num_fvgs,
            'has_bullish_fvg': float(has_bull_fvg),
            'has_bearish_fvg': float(has_bear_fvg),
            'sweep_detected': sweep_flag,

            # Session
            **session_one_hot,
            'in_killzone': float(in_kz),

            # Technical base
            'rsi': float(signal.rsi),
            'atr': float(signal.atr),
            'volume_ratio': float(signal.volume_ratio),
            'momentum_score': float(signal.momentum_score),
            'dist_ema_trend_atr': float(dist_ema_trend_atr),

            # Higher TF bias
            'htf_bullish': float(signal.higher_tf_bias == 'bullish'),
            'htf_bearish': float(signal.higher_tf_bias == 'bearish'),
            'htf_neutral': float(signal.higher_tf_bias not in ['bullish', 'bearish']),

            # Trade geometry
            'rr_ratio': float(rr_ratio),
        }

        return features


# ==============================================================================
# ML TRAINER
# ==============================================================================

class ICTMLTrainer:
    """
    Converts ICT signals into a labeled dataset and trains a classifier:

    Label: did +2R happen before -1R within N bars?
    (config.label_rr_win, config.label_rr_loss, config.label_horizon_bars)
    """

    def __init__(self, config: Optional[ICTConfig] = None):
        self.config = config or ICTConfig()
        self.feature_builder = ICTFeatureBuilder(self.config)
        self.model = None
        self.feature_names: List[str] = []

    # ---------- Labeling ----------

    def _label_signal(self, df: pd.DataFrame, signal: ICTSignal) -> Optional[int]:
        """
        1 = +RR_win hit before -RR_loss
        0 = -RR_loss hit before +RR_win
        None = neither hit within horizon
        """
        ts = signal.timestamp
        try:
            idx = df.index.get_loc(ts)
        except KeyError:
            return None

        horizon = self.config.label_horizon_bars
        future = df.iloc[idx+1: idx+1+horizon]
        if future.empty:
            return None

        entry = signal.entry_price
        risk = abs(signal.entry_price - signal.stop_loss)
        if risk <= 0:
            return None

        rr_win = self.config.label_rr_win
        rr_loss = self.config.label_rr_loss

        if signal.signal_type == 'long':
            win_level = entry + rr_win * risk
            loss_level = entry - rr_loss * risk

            for _, row in future.iterrows():
                low = row['low']
                high = row['high']
                if low <= loss_level:
                    return 0
                if high >= win_level:
                    return 1

        else:  # short
            win_level = entry - rr_win * risk
            loss_level = entry + rr_loss * risk

            for _, row in future.iterrows():
                low = row['low']
                high = row['high']
                if high >= loss_level:
                    return 0
                if low <= win_level:
                    return 1

        return None

    def build_labeled_dataframe(
        self,
        df: pd.DataFrame,
        signals: List[ICTSignal]
    ) -> pd.DataFrame:
        """
        Build a DataFrame with features + `label` column from signals.
        """
        rows: List[Dict[str, Any]] = []
        for sig in signals:
            label = self._label_signal(df, sig)
            if label is None:
                continue
            feats = self.feature_builder.build_features_for_signal(df, sig)
            if not feats:
                continue
            feats['label'] = int(label)
            feats['timestamp'] = sig.timestamp
            rows.append(feats)

        if not rows:
            logger.warning("[ICT-ML] No labeled rows produced. Check config/horizon.")
            return pd.DataFrame()

        df_out = pd.DataFrame(rows)
        return df_out

    # ---------- Training ----------

    def train_model(
        self,
        data: pd.DataFrame,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """
        Train a classifier (LightGBM if available, else sklearn GradientBoosting).
        """
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import roc_auc_score, f1_score, classification_report

        if data.empty:
            raise ValueError("No data to train on.")

        X = data.drop(columns=['label', 'timestamp'])
        y = data['label'].astype(int)
        self.feature_names = list(X.columns)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # Try LightGBM first
        model = None
        model_type = "sklearn_gb"
        try:
            import lightgbm as lgb
            model = lgb.LGBMClassifier(
                n_estimators=300,
                max_depth=-1,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="binary",
                random_state=random_state
            )
            model_type = "lightgbm"
            logger.info("[ICT-ML] Using LightGBM.")
        except ImportError:
            from sklearn.ensemble import GradientBoostingClassifier
            model = GradientBoostingClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=3,
                random_state=random_state
            )
            logger.info("[ICT-ML] LightGBM not found, using sklearn GradientBoosting.")

        model.fit(X_train, y_train)
        self.model = model

        # Metrics
        proba_test = model.predict_proba(X_test)[:, 1]
        pred_test = (proba_test >= 0.5).astype(int)

        auc = roc_auc_score(y_test, proba_test)
        f1 = f1_score(y_test, pred_test)
        report = classification_report(y_test, pred_test, output_dict=True)

        logger.info(f"[ICT-ML] AUC={auc:.3f}, F1={f1:.3f}")

        return {
            'model_type': model_type,
            'auc': float(auc),
            'f1': float(f1),
            'report': report,
            'n_train': int(len(X_train)),
            'n_test': int(len(X_test))
        }

    def save_model(self, path: str) -> None:
        """Persist model + metadata to disk as a joblib pickle."""
        if self.model is None:
            raise ValueError("No model to save.")
        import joblib

        payload = {
            'model': self.model,
            'feature_names': self.feature_names,
            'config': self.config,
        }
        joblib.dump(payload, path)
        logger.info(f"[ICT-ML] Saved model to {path}")
