from __future__ import annotations
import asyncio
import pandas as pd
import numpy as np
import pandas_ta as ta  # uses pandas_ta for generic indicators
from datetime import datetime, timedelta, time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from loguru import logger

from src.utils.risk_manager import RiskManager
from config.config import config
from src.utils.database import DatabaseManager
from src.indicators.custom import CustomIndicators
from src.core.models import Signal




# -------------------------------------------------
# ICT MASTER CONFIG
# -------------------------------------------------

@dataclass
class ICTMasterConfig:
    min_atr: float = 6.0
    displacement_atr_mult: float = 1.0
    max_poi_distance_atr: float = 3.0
    min_rr: float = 2.0
    swing_strength: int = 3
    lookback_bars: int = 120
    killzone_am_start: time = time(9, 30)
    killzone_am_end: time = time(11, 30)
    killzone_pm_start: time = time(13, 0)
    killzone_pm_end: time = time(15, 30)


class ICTMasterStrategy:
    """
    Simplified ICT-style master strategy:

    - Bias: close vs EMA(200)
    - Impulse leg: displacement in direction of bias
    - POI: last opposite candle before impulse
    - Trade: when price trades back into POI
    - Supports modes: conservative / normal / aggressive
    """

    def __init__(self, cfg_dict: Optional[Dict] = None):
        cfg_dict = cfg_dict or {}

        # ---- MODE HANDLING (matches config.py) ----
        mode = cfg_dict.pop("mode", "normal")

        base_min_atr = cfg_dict.pop("base_min_atr", 6.0)
        base_disp = cfg_dict.pop("base_displacement_atr_mult", 1.0)
        base_poi = cfg_dict.pop("base_max_poi_distance_atr", 3.0)
        base_rr = cfg_dict.pop("base_min_rr", 2.0)

        if mode == "aggressive":
            min_atr = base_min_atr * 0.5
            disp = base_disp * 0.5
            poi = base_poi * 2.0
            rr = base_rr * 0.7
        elif mode == "conservative":
            min_atr = base_min_atr * 1.5
            disp = base_disp * 1.5
            poi = base_poi * 0.8
            rr = base_rr * 1.2
        else:  # "normal"
            min_atr = base_min_atr
            disp = base_disp
            poi = base_poi
            rr = base_rr

        self.config = ICTMasterConfig(
            min_atr=min_atr,
            displacement_atr_mult=disp,
            max_poi_distance_atr=poi,
            min_rr=rr,
            swing_strength=cfg_dict.get("swing_strength", 3),
            lookback_bars=cfg_dict.get("lookback_bars", 120),
        )

    # ----------------- PUBLIC ----------------- #

    def generate_signals(self, df: pd.DataFrame, symbol: str) -> List[Signal]:
        signals: List[Signal] = []

        if df.empty or len(df) < max(self.config.lookback_bars, 50):
            return signals

        df = df.copy()
        self._ensure_atr(df)

        current = df.iloc[-1]
        ts = self._get_bar_time(current)
        atr_val = float(current.get("atr", 0.0))
        close = float(current["close"])

        # 1) Killzone filter
        if not self._in_killzone(ts.time()):
            logger.debug("[ICTMaster] SKIP: outside killzone.")
            return signals

        # 2) ATR filter
        if atr_val < self.config.min_atr:
            logger.debug(
                f"[ICTMaster] SKIP: ATR {atr_val:.2f} < min {self.config.min_atr:.2f}"
            )
            return signals

        # 3) Bias
        bias = self._get_bias(df)
        logger.info(f"[ICTMaster] Bias for {symbol}: {bias}")
        if bias == "neutral":
            logger.debug("[ICTMaster] SKIP: neutral bias.")
            return signals

        # 4) Impulse leg
        leg = self._detect_impulse_leg(df, bias)
        if leg is None:
            logger.debug("[ICTMaster] SKIP: no valid impulse leg found.")
            return signals

        leg_start, leg_end = leg
        logger.debug(f"[ICTMaster] impulse leg {leg_start}->{leg_end} ({bias})")

        # 5) POI from impulse
        poi = self._build_poi_from_leg(df, bias, leg_start, leg_end)
        if poi is None:
            logger.debug("[ICTMaster] SKIP: no POI derived from impulse.")
            return signals

        logger.debug(
            f"[ICTMaster] POI zone [{poi['zone_low']:.1f}, {poi['zone_high']:.1f}] "
            f"@ index {poi['index']}"
        )

        # 6) Check price vs POI
        if not self._price_in_poi(current, poi, atr_val):
            logger.debug(
                f"[ICTMaster] SKIP: price {close:.1f} not in/near POI zone."
            )
            return signals

        # 7) Build trade
        trade = self._build_trade_from_poi(current, poi, bias, symbol)
        if trade is None:
            logger.debug("[ICTMaster] SKIP: trade did not meet RR/structure rules.")
            return signals

        logger.info(
            f"[ICTMaster] Emitting {trade.signal_type} signal for {symbol} "
            f"@ {trade.entry_price:.2f} | SL={trade.stop_loss:.2f} "
            f"TP={trade.take_profit:.2f} RR={trade.risk_reward:.2f}"
        )
        signals.append(trade)
        return signals

    # ----------------- HELPERS ----------------- #

    def _ensure_atr(self, df: pd.DataFrame, period: int = 14):
        if "atr" in df.columns:
            return
        high = df["high"]
        low = df["low"]
        close = df["close"]
        prev_close = close.shift(1)

        tr = pd.concat(
            [
                (high - low),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)

        df["atr"] = tr.rolling(period).mean()

    def _get_bar_time(self, row: pd.Series) -> datetime:
        ts = row.get("timestamp")
        if isinstance(ts, datetime):
            return ts
        if isinstance(ts, str):
            try:
                return datetime.fromisoformat(ts)
            except Exception:
                pass
        idx = row.name
        if isinstance(idx, datetime):
            return idx
        raise ValueError("Cannot infer bar timestamp for ICTMasterStrategy.")

    def _in_killzone(self, t: time) -> bool:
        c = self.config
        in_am = c.killzone_am_start <= t <= c.killzone_am_end
        in_pm = c.killzone_pm_start <= t <= c.killzone_pm_end
        return in_am or in_pm

    def _get_bias(self, df: pd.DataFrame) -> str:
        """Simple bias: close vs EMA(200)."""
        cfg = self.config
        sub = df.tail(cfg.lookback_bars).copy()
        sub["ema200"] = sub["close"].ewm(span=200, adjust=False).mean()

        ema_last = float(sub["ema200"].iloc[-1])
        price_last = float(sub["close"].iloc[-1])

        if price_last > ema_last:
            return "bullish"
        elif price_last < ema_last:
            return "bearish"
        else:
            return "neutral"

    def _detect_impulse_leg(self, df: pd.DataFrame, bias: str) -> Optional[Tuple[int, int]]:
        """Find a simple displacement leg in direction of bias."""
        cfg = self.config
        sub = df.tail(cfg.lookback_bars).copy()
        atr = sub["atr"].fillna(method="ffill")

        best_leg = None
        best_size = 0.0

        for end in range(len(sub) - 1, 5, -1):
            for length in range(3, 20):
                start = end - length
                if start < 0:
                    break

                start_price = float(sub["close"].iloc[start])
                end_price = float(sub["close"].iloc[end])
                leg_atr = float(atr.iloc[start:end + 1].mean() or 1.0)

                diff = end_price - start_price
                size = abs(diff)

                if bias == "bullish" and diff <= 0:
                    continue
                if bias == "bearish" and diff >= 0:
                    continue

                if size < cfg.displacement_atr_mult * leg_atr:
                    continue

                if size > best_size:
                    best_size = size
                    abs_start = len(df) - len(sub) + start
                    abs_end = len(df) - len(sub) + end
                    best_leg = (abs_start, abs_end)

        return best_leg

    def _build_poi_from_leg(
        self, df: pd.DataFrame, bias: str, leg_start: int, leg_end: int
    ) -> Optional[Dict]:
        """Define a simple order block POI from last opposite candle before impulse."""
        sub = df.iloc[: leg_start + 1]
        if sub.empty:
            return None

        if bias == "bullish":
            mask = sub["close"] < sub["open"]
        else:
            mask = sub["close"] > sub["open"]

        candidate_idx = np.where(mask.values)[0]
        if len(candidate_idx) == 0:
            return None

        ob_idx = int(candidate_idx[-1])
        ob_row = df.iloc[ob_idx]

        zone_low = float(min(ob_row["open"], ob_row["close"], ob_row["low"]))
        zone_high = float(max(ob_row["open"], ob_row["close"], ob_row["high"]))

        return {
            "index": ob_idx,
            "bias": bias,
            "zone_low": zone_low,
            "zone_high": zone_high,
        }

    def _price_in_poi(self, current: pd.Series, poi: Dict, atr_value: float) -> bool:
        price = float(current["close"])
        z_low = float(poi["zone_low"])
        z_high = float(poi["zone_high"])

        if z_low <= price <= z_high:
            return True

        buf = 0.2 * (atr_value or 1.0)
        if poi["bias"] == "bullish":
            return (price >= z_low - buf) and (price <= z_high + buf)
        else:
            return (price <= z_high + buf) and (price >= z_low - buf)

    def _build_trade_from_poi(
        self, current: pd.Series, poi: Dict, bias: str, symbol: str
    ) -> Optional[Signal]:
        cfg = self.config

        entry = float(current["close"])
        atr_value = float(current.get("atr", 0.0))
        z_low = float(poi["zone_low"])
        z_high = float(poi["zone_high"])

        distance_atr = abs(entry - (z_high if bias == "bearish" else z_low)) / max(
            atr_value, 1e-6
        )
        if distance_atr > cfg.max_poi_distance_atr:
            logger.debug(
                f"[ICTMaster] SKIP trade: POI distance {distance_atr:.2f} ATR > "
                f"{cfg.max_poi_distance_atr:.2f}"
            )
            return None

        if bias == "bullish":
            stop_loss = z_low - 0.1 * atr_value
            risk = entry - stop_loss
            if risk <= 0:
                return None
            take_profit = entry + cfg.min_rr * risk
            signal_type = "long"
        else:
            stop_loss = z_high + 0.1 * atr_value
            risk = stop_loss - entry
            if risk <= 0:
                return None
            take_profit = entry - cfg.min_rr * risk
            signal_type = "short"

        rr = abs(take_profit - entry) / abs(entry - stop_loss)
        if rr < cfg.min_rr:
            logger.debug(
                f"[ICTMaster] SKIP trade: RR {rr:.2f} < min {cfg.min_rr:.2f}"
            )
            return None

        confidence = min(0.55 + 0.05 * (rr - cfg.min_rr), 0.9)

        return Signal(
            timestamp=current.get("timestamp", datetime.now()),
            strategy="ict_master",
            signal_type=signal_type,
            symbol=symbol,
            entry_price=float(entry),
            stop_loss=float(stop_loss),
            take_profit=float(take_profit),
            confidence=float(confidence),
            risk_reward=float(rr),
            atr=float(atr_value),
            volume=int(current.get("volume", 0) or 0),
            metadata={
                "poi_index": poi["index"],
                "poi_zone_low": z_low,
                "poi_zone_high": z_high,
                "distance_atr": distance_atr,
                "bias": bias,
            },
        )


# =========================
# ORIGINAL ENGINE
# =========================

@dataclass
class SignalGenerationEngine:
    """Main engine for generating HIGH-CONFIDENCE trading signals"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.risk_manager = RiskManager()
        self.custom_indicators = CustomIndicators()
        
        # Initialize strategies
        self.strategies = {}
        self._init_strategies()
        
        # Market data cache
        self.data_cache = {}
        self.last_analysis_time = {}
        
        # Track recent signals to avoid duplicates
        self.recent_signals = []
        
        logger.info("SignalGenerationEngine initialized with HIGH-CONFIDENCE mode")
    
    def _init_strategies(self):
        """Initialize strategies.
        
        If ict_master is enabled, we ONLY use the master strategy.
        Otherwise, fall back to the legacy strategy set.
        """
        from src.strategies.divergence import DivergenceStrategy
        from src.strategies.ema_pullback import EMAPullbackStrategy
        from src.strategies.htf_supertrend import HTFSupertrendStrategy
        from src.strategies.reversal_breakout import ReversalBreakoutStrategy
        
        strategy_configs = config.STRATEGIES

        # ✅ Master strategy takes precedence
        if "ict_master" in strategy_configs and strategy_configs["ict_master"].get("enabled"):
            conf = strategy_configs["ict_master"].copy()
            conf.pop("enabled", None)
            self.strategies["ict_master"] = ICTMasterStrategy(conf)
            logger.info("ICT Master strategy initialized (single master mode).")
            return
        
        # Legacy strategies (fallback if master is disabled)
        if strategy_configs['ema_pullback']['enabled']:
            conf = strategy_configs['ema_pullback'].copy()
            conf.pop('enabled', None)
            self.strategies['ema_pullback'] = EMAPullbackStrategy(conf)
            logger.info("EMA Pullback strategy initialized")
        
        if strategy_configs['htf_supertrend']['enabled']:
            conf = strategy_configs['htf_supertrend'].copy()
            conf.pop('enabled', None)
            self.strategies['htf_supertrend'] = HTFSupertrendStrategy(conf)
            logger.info("HTF Supertrend strategy initialized")
        
        if strategy_configs['reversal_breakout']['enabled']:
            conf = strategy_configs['reversal_breakout'].copy()
            conf.pop('enabled', None)
            self.strategies['reversal_breakout'] = ReversalBreakoutStrategy(conf)
            logger.info("Reversal-Breakout strategy initialized")
    
    async def process_new_data(self, market_data: Dict) -> List[Signal]:
        """Process new market data and generate signals.

        DEBUG VERSION:
        - No confidence filter (we keep EVERY signal).
        - Logs raw signal counts so we can see if strategies are firing.
        """
        try:
            symbol = market_data["symbol"]
            timeframe = market_data["timeframe"]

            # Store bar in DB
            self.db_manager.insert_market_data(market_data)

            # Update in-memory cache
            cache_key = f"{symbol}_{timeframe}"
            if cache_key not in self.data_cache:
                self.data_cache[cache_key] = []
            self.data_cache[cache_key].append(market_data)

            # keep last 500 bars only
            if len(self.data_cache[cache_key]) > 500:
                self.data_cache[cache_key] = self.data_cache[cache_key][-500:]

            # ----- generate signals -----
            signals = await self._generate_signals(symbol, timeframe)

            if not signals:
                # nothing from any strategy for this bar
                return []

            logger.info(
                f"[Engine] RAW signals this bar: "
                f"{len(signals)} total -> "
                + ", ".join(
                    f"{s.strategy}:{s.signal_type}@{s.entry_price:.1f}"
                    for s in signals
                )
            )

            # NO confidence filter for now – we want to see everything
            filtered_signals = signals

            # remove near-duplicates
            unique_signals = self._remove_duplicate_signals(filtered_signals)

            # persist to DB
            for signal in unique_signals:
                self.db_manager.insert_signal(
                    {
                        "timestamp": signal.timestamp,
                        "strategy": signal.strategy,
                        "signal_type": signal.signal_type,
                        "symbol": signal.symbol,
                        "entry_price": signal.entry_price,
                        "stop_loss": signal.stop_loss,
                        "take_profit": signal.take_profit,
                        "confidence": signal.confidence,
                        "risk_reward": signal.risk_reward,
                        "atr": signal.atr,
                        "volume": signal.volume,
                        "metadata": signal.metadata,
                    }
                )

            if unique_signals:
                logger.info(
                    f"🎯 Emitting {len(unique_signals)} signals this bar "
                    f"(no confidence filter, RR/conf printed above)."
                )

            return unique_signals

        except Exception as e:
            logger.error(f"Error processing new data: {e}")
            return []

    
    async def _generate_signals(self, symbol: str, timeframe: str) -> List[Signal]:
        """Generate signals with market structure analysis."""
        signals: List[Signal] = []

        df = self._get_analysis_dataframe(symbol, timeframe)
        if df.empty or len(df) < 100:
            return signals

        df_with_indicators = self._calculate_indicators(df)

        market_condition = self._analyze_market_structure(df_with_indicators)
        logger.info(f"Market condition: {market_condition}")

        for strategy_name, strategy in self.strategies.items():
            try:
                strategy_signals = strategy.generate_signals(
                    df_with_indicators, symbol
                )

                logger.info(
                    f"[Engine] {strategy_name} produced "
                    f"{len(strategy_signals)} raw signals on this bar."
                )

                # (optional) market-condition based confidence boosts
                for signal in strategy_signals:
                    if market_condition == "strong_trend":
                        if strategy_name in ["ema_pullback", "htf_supertrend"]:
                            signal.confidence = min(signal.confidence + 0.15, 0.95)
                    elif market_condition == "reversal":
                        if strategy_name == "reversal_breakout":
                            signal.confidence = min(signal.confidence + 0.20, 0.95)

                signals.extend(strategy_signals)

            except Exception as e:
                logger.error(f"Error in {strategy_name} strategy: {e}")

        return signals

    
    def _get_analysis_dataframe(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Get market data as DataFrame for analysis"""
        cache_key = f"{symbol}_{timeframe}"
        if cache_key in self.data_cache and len(self.data_cache[cache_key]) > 0:
            df = pd.DataFrame(self.data_cache[cache_key])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        
        # Get from database
        df = self.db_manager.get_latest_data(symbol, timeframe, limit=200)
        return df
    
    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators with proper error handling"""
        try:
            # Core momentum indicators
            df['rsi'] = ta.rsi(df['close'], length=14)
            df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
            
            # EMAs for trend
            df['ema_21'] = ta.ema(df['close'], length=21)
            df['ema_50'] = ta.ema(df['close'], length=50)
            df['ema_200'] = ta.ema(df['close'], length=200)
            
            # MACD for momentum
            macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
            if macd is not None and not macd.empty:
                df['macd'] = macd['MACD_12_26_9']
                df['macd_signal'] = macd['MACDs_12_26_9']
                df['macd_histogram'] = macd['MACDh_12_26_9']
            
            # Bollinger Bands for volatility
            bb = ta.bbands(df['close'], length=20, std=2)
            if bb is not None and not bb.empty:
                df['bb_upper'] = bb.iloc[:, 0]
                df['bb_middle'] = bb.iloc[:, 1]
                df['bb_lower'] = bb.iloc[:, 2]
                df['bb_width'] = df['bb_upper'] - df['bb_lower']
            
            # ADX for trend strength
            adx = ta.adx(df['high'], df['low'], df['close'], length=14)
            if adx is not None and not adx.empty:
                df['adx'] = adx['ADX_14']
            
            # Volume indicators
            df['volume_sma'] = ta.sma(df['volume'], length=20)
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            # Custom VWAP
            df['vwap'] = self.custom_indicators.calculate_vwap(df)
            
            # VWAP bands
            if 'vwap' in df.columns and 'atr' in df.columns:
                df['vwap_upper_1'] = df['vwap'] + df['atr']
                df['vwap_lower_1'] = df['vwap'] - df['atr']
                df['vwap_upper_2'] = df['vwap'] + 2 * df['atr']
                df['vwap_lower_2'] = df['vwap'] - 2 * df['atr']
            
            # Supertrend
            df['supertrend'] = self.custom_indicators.calculate_supertrend(df)
            
            # Fill NaN values
            df = df.fillna(method='ffill').fillna(method='bfill')
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return df
    
    def _analyze_market_structure(self, df: pd.DataFrame) -> str:
        """Analyze market structure for better signal context"""
        if df.empty or len(df) < 50:
            return 'unknown'
        
        try:
            current_bar = df.iloc[-1]
            
            if 'adx' in current_bar and not pd.isna(current_bar['adx']):
                adx = current_bar['adx']
                
                if adx > 30:
                    if 'ema_21' in current_bar and 'ema_50' in current_bar:
                        return 'strong_trend'
                elif adx < 20:
                    return 'ranging'
            
            if 'rsi' in current_bar:
                rsi = current_bar['rsi']
                if not pd.isna(rsi):
                    if rsi < 25 or rsi > 75:
                        return 'reversal'
            
            if 'atr' in current_bar:
                atr = current_bar['atr']
                if not pd.isna(atr):
                    if atr > 20:
                        return 'high_volatility'
                    elif atr < 10:
                        return 'low_volatility'
            
            return 'normal'
            
        except Exception as e:
            logger.error(f"Error analyzing market structure: {e}")
            return 'unknown'
    
    def _remove_duplicate_signals(self, signals: List[Signal]) -> List[Signal]:
        """Remove duplicate or very similar signals"""
        if not signals:
            return signals
        
        unique_signals = []
        
        for signal in signals:
            is_duplicate = False
            
            for recent in self.recent_signals[-10:]:
                if (signal.strategy == recent.strategy and
                    signal.signal_type == recent.signal_type and
                    abs(signal.entry_price - recent.entry_price) < 5 and
                    (signal.timestamp - recent.timestamp).seconds < 300):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_signals.append(signal)
                self.recent_signals.append(signal)
        
        if len(self.recent_signals) > 20:
            self.recent_signals = self.recent_signals[-20:]
        
        return unique_signals
    
    def get_strategy_performance(self, strategy_name: str, days: int = 7) -> Dict:
        """Get recent performance metrics for a specific strategy"""
        try:
            signals = self.db_manager.get_recent_signals(strategy_name, limit=50)
            
            if signals.empty:
                return {'total_signals': 0, 'avg_confidence': 0}
            
            recent_date = datetime.now() - timedelta(days=days)
            recent_signals = signals[signals['timestamp'] > recent_date]
            
            return {
                'total_signals': len(recent_signals),
                'avg_confidence': recent_signals['confidence'].mean(),
                'avg_risk_reward': recent_signals['risk_reward'].mean(),
                'strategies': recent_signals['strategy'].value_counts().to_dict()
            }
            
        except Exception as e:
            logger.error(f"Error getting strategy performance: {e}")
            return {'total_signals': 0, 'avg_confidence': 0}
    
    def cleanup(self):
        """Clean up resources"""
        try:
            self.db_manager.cleanup_old_data(days_to_keep=7)
            logger.info("Database cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
