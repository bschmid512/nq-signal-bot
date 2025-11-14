# src/strategies/ict_complete_adapter.py

from typing import List
import numpy as np
from loguru import logger

from src.core.models import Signal
from dataclasses import dataclass
from typing import List, Dict, Any
from .ict_complete_strategy import ICTStrategy, ICTConfig, ICTSignal



class ICTCompleteAdapter:
    """
    Adapter to plug the big ICTStrategy into the existing SignalGenerationEngine.
    It:
      - Reuses all the ICT analyzers
      - Only emits signals for the *current* bar
      - Returns your project's Signal objects
    """

    def __init__(self, cfg: dict | None = None):
        ict_cfg = ICTConfig(**(cfg or {}))
        self.strategy = ICTStrategy(ict_cfg)

    def generate_signals(self, df, symbol: str) -> List[Signal]:
        if df.empty:
            return []

        # Let the internal strategy prepare indicators
        df = self.strategy.prepare_data(df)

        # Use only the last bar for decision
        current_bar = df.iloc[-1]
        historical_data = df  # full history for structure/OBs

        ict_signal: ICTSignal | None = self._make_signal_for_last_bar(
            historical_data, current_bar
        )
        if ict_signal is None:
            return []

        # Convert ICTSignal -> your Signal dataclass
        rr = abs(ict_signal.take_profit - ict_signal.entry_price) / max(
            abs(ict_signal.entry_price - ict_signal.stop_loss), 1e-6
        )

        logger.info(
            f"[ICTComplete] {ict_signal.signal_type.upper()} "
            f"{symbol} @{ict_signal.entry_price:.2f} "
            f"SL={ict_signal.stop_loss:.2f} TP={ict_signal.take_profit:.2f} "
            f"RR={rr:.2f} conf={ict_signal.confidence:.2f}"
        )

        return [
            Signal(
                timestamp=ict_signal.timestamp,
                strategy="ict_complete",
                signal_type=ict_signal.signal_type,
                symbol=symbol,
                entry_price=ict_signal.entry_price,
                stop_loss=ict_signal.stop_loss,
                take_profit=ict_signal.take_profit,
                confidence=ict_signal.confidence,
                risk_reward=rr,
                atr=float(ict_signal.atr),
                volume=0,  # we don't size by volume here
                metadata={
                    "order_block_info": ict_signal.order_block_info,
                    "market_structure": ict_signal.market_structure,
                    "session_info": ict_signal.session_info,
                    "liquidity_context": ict_signal.liquidity_context,
                    "rsi": ict_signal.rsi,
                    "momentum_score": ict_signal.momentum_score,
                    "higher_tf_bias": ict_signal.higher_tf_bias,
                },
            )
        ]

    def _make_signal_for_last_bar(self, df, current_bar):
        """
        This is basically ICTStrategy._create_ict_signal,
        but scoped to the last bar instead of scanning the whole history.
        """

        # Reuse logic from ICTStrategy._check_ema_crossover and _create_ict_signal
        idx = len(df) - 1
        if idx < 2:
            return None

        # EMA crossover check (same as _check_ema_crossover)
        current_fast = df["ema_fast"].iloc[idx]
        current_slow = df["ema_slow"].iloc[idx]
        prev_fast = df["ema_fast"].iloc[idx - 1]
        prev_slow = df["ema_slow"].iloc[idx - 1]

        if (
            np.isnan(current_fast)
            or np.isnan(current_slow)
            or np.isnan(prev_fast)
            or np.isnan(prev_slow)
        ):
            return None

        golden_cross = prev_fast <= prev_slow and current_fast > current_slow
        death_cross = prev_fast >= prev_slow and current_fast < current_slow

        if not (golden_cross or death_cross):
            return None

        signal_type = "long" if golden_cross else "short"
        entry_price = current_bar["close"]
        atr = current_bar["atr"]

        if np.isnan(atr) or atr < self.strategy.config.min_atr:
            return None

        # SL/TP
        if signal_type == "long":
            stop_loss = entry_price - atr * 1.5
            risk = entry_price - stop_loss
            take_profit = entry_price + risk * self.strategy.config.target_rr_ratio
        else:
            stop_loss = entry_price + atr * 1.5
            risk = stop_loss - entry_price
            take_profit = entry_price - risk * self.strategy.config.target_rr_ratio

        if risk <= 0:
            return None

        # === Reuse the rest of _create_ict_signal logic ===
        bias = self.strategy.market_analyzer.determine_market_bias(df.tail(100))
        key_levels = self.strategy.market_analyzer.find_support_resistance_levels(
            df.tail(50)
        )

        order_blocks = self.strategy.ob_analyzer.find_order_blocks(
            df.tail(self.strategy.config.ob_lookback), bias
        )
        ob_confluence = self.strategy.ob_analyzer.calculate_ob_confluence(
            entry_price, order_blocks, atr
        )

        liquidity_levels = self.strategy.liquidity_analyzer.find_liquidity_pools(
            df.tail(30)
        )
        imbalances = self.strategy.liquidity_analyzer.detect_fair_value_gaps(
            df.tail(10)
        )
        liquidity_sweep = self.strategy.liquidity_analyzer.detect_liquidity_sweep(
            df.tail(10)
        )

        current_time = current_bar.name.time()
        session = self.strategy.session_manager.get_current_session(current_time)
        in_killzone = self.strategy.session_manager.is_in_killzone(current_time)

        session_score = self.strategy.session_manager.get_session_score(session)
        killzone_score = self.strategy.session_manager.get_killzone_score(in_killzone)

        rsi = current_bar["rsi"] if not np.isnan(current_bar["rsi"]) else 50.0
        momentum_score = self.strategy.technical_analyzer.calculate_momentum_score(
            df.tail(20)
        )
        volume_confluence = self.strategy.technical_analyzer.get_volume_confluence(
            df, current_bar.name
        )

        higher_tf_bias = self.strategy._get_higher_timeframe_bias(df)

        signal_context = {
            "bias": bias,
            "signal_type": signal_type,
            "ob_confluence": ob_confluence,
            "session_score": session_score,
            "killzone_score": killzone_score,
            "rsi": rsi,
            "momentum_score": momentum_score,
            "volume_confluence": volume_confluence,
        }

        confidence = self.strategy.confidence_scorer.calculate_signal_confidence(
            signal_context
        )

        if confidence < self.strategy.config.min_confidence:
            return None

        return ICTSignal(
            timestamp=current_bar.name,
            signal_type=signal_type,
            entry_price=float(entry_price),
            stop_loss=float(stop_loss),
            take_profit=float(take_profit),
            confidence=float(confidence),
            order_block_info={
                "confluence": ob_confluence,
                "blocks": order_blocks[:2],
            },
            market_structure={
                "bias": bias,
                "key_levels": key_levels,
                "swing_structure": "valid",
            },
            session_info={
                "session": session,
                "in_killzone": in_killzone,
                "session_score": session_score,
                "killzone_score": killzone_score,
            },
            liquidity_context={
                "levels": liquidity_levels,
                "imbalances": imbalances,
                "sweep_detected": liquidity_sweep,
            },
            rsi=float(rsi),
            atr=float(atr),
            volume_ratio=float(volume_confluence),
            higher_tf_bias=higher_tf_bias,
            momentum_score=float(momentum_score),
        )
