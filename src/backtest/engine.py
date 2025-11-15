from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd
from loguru import logger

from config.config import config
from src.utils.database import DatabaseManager
from src.utils.risk_manager import RiskManager
from src.core.engine import SignalGenerationEngine
from src.core.models import Signal

# ------------------------------------------------------------
# Basic contract / cost model for MNQ
# ------------------------------------------------------------

TICK_SIZE: float = 0.25          # NQ/MNQ tick size in points
POINT_VALUE: float = 2.0         # $ per point for 1 MNQ contract
SLIPPAGE_TICKS: float = 1.0      # assume 1 tick adverse slippage on entry
COMMISSION_PER_SIDE: float = 0.50  # $ per side, per contract
MAX_HOLD_BARS: int = 24          # how many 5m bars we hold a trade at most
MAX_BARS_FROM_DB: int = 20000    # how many bars to pull for a backtest run


class BacktestEngine:
    """Backtesting engine for strategy validation.

    This implementation:
    - Uses real SignalGenerationEngine (ICT master + ML gate)
    - Feeds bars from DB into the engine via in‑memory cache only
      (no DB writes during backtest, so production DB stays clean)
    - Simulates fills with a simple MNQ cost model
    - Tracks an equity curve and computes *real* DD / Sharpe / avg R
    """

    def __init__(self, db_manager: DatabaseManager, risk_manager: RiskManager,
                 initial_equity: float = 10000.0) -> None:
        self.db_manager = db_manager
        self.risk_manager = risk_manager
        self.signal_engine = SignalGenerationEngine(db_manager)

        self.initial_equity: float = float(initial_equity)

        # Aggregate stats returned to main.py
        self.results: Dict[str, float] = {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "total_pnl": 0.0,
            "max_drawdown": 0.0,   # as *fraction* of equity, negative number
            "sharpe_ratio": 0.0,
            "avg_r_multiple": 0.0,
            "win_rate": 0.0,
            "tp_hits": 0,
            "sl_hits": 0,
            "long_trades": 0,
            "short_trades": 0,
        }

        # Per‑trade / per‑run state
        self.trade_pnls: List[float] = []
        self.trade_r: List[float] = []
        self.trade_timestamps: List[datetime] = []
        self.equity_curve: List[float] = [self.initial_equity]

        logger.info("BacktestEngine initialized")

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_recent_data(self, limit: int = MAX_BARS_FROM_DB) -> pd.DataFrame:
        """Load the last `limit` bars of real data from the DB.

        Uses DatabaseManager.get_latest_data, which prefers the cleaned
        market_data_clean table if it exists.
        """
        try:
            symbol = config.SYMBOL
            timeframe = config.PRIMARY_TIMEFRAME

            logger.info(
                f"[Backtest] Loading last {limit} bars from DB for {symbol} {timeframe}-min"
            )
            df = self.db_manager.get_latest_data(symbol, timeframe, limit=limit)

            if df is None or df.empty:
                logger.warning("[Backtest] No recent data found in DB.")
                return pd.DataFrame()

            if "timestamp" in df.columns:
                df = df.copy()
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df = df.sort_values("timestamp").reset_index(drop=True)

            return df

        except Exception as e:
            logger.error(f"[Backtest] Error loading recent data from DB: {e}")
            return pd.DataFrame()

    # ------------------------------------------------------------------
    # Core backtest loop
    # ------------------------------------------------------------------

    def run_backtest(self, start_date: str, end_date: str) -> Dict[str, float]:
        """Run backtest between start_date / end_date.

        NOTE: current implementation ignores the dates and always takes the
        last MAX_BARS_FROM_DB bars – this matches the existing CLI behaviour.
        """
        try:
            logger.info(
                f"Backtest Mode - using last {MAX_BARS_FROM_DB} bars from database (dates ignored)"
            )

            df = self.load_recent_data(limit=MAX_BARS_FROM_DB)
            if df.empty:
                logger.warning("[Backtest] No data to backtest on.")
                return self.results

            # Infer symbol / timeframe from data or config
            symbol = (
                str(df.get("symbol").iloc[0])
                if "symbol" in df.columns
                else config.SYMBOL
            )
            timeframe = (
                str(df.get("timeframe").iloc[0])
                if "timeframe" in df.columns
                else str(config.PRIMARY_TIMEFRAME)
            )

            cache_key = f"{symbol}_{timeframe}"
            self.signal_engine.data_cache[cache_key] = []

            # Main bar loop
            for i, row in df.iterrows():
                bar = {
                    "timestamp": row["timestamp"],
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": int(row.get("volume", 0) or 0),
                    "timeframe": str(row.get("timeframe", timeframe)),
                    "symbol": str(row.get("symbol", symbol)),
                }

                # Update in‑memory cache only (no DB writes)
                self.signal_engine.data_cache[cache_key].append(bar)
                if len(self.signal_engine.data_cache[cache_key]) > 500:
                    self.signal_engine.data_cache[cache_key] =                         self.signal_engine.data_cache[cache_key][-500:]

                # Generate signals off the current history
                signals = asyncio.run(
                    self.signal_engine._generate_signals(symbol, timeframe)
                )

                if not signals:
                    continue

                logger.info(
                    f"[Backtest] RAW signals this bar: "
                    f"{len(signals)} -> "
                    + ", ".join(
                        f"{s.strategy}:{s.signal_type}@{s.entry_price:.1f}"
                        for s in signals
                    )
                )

                # Apply RiskManager quality filter
                filtered = self.risk_manager.filter_signals(signals)
                if not filtered:
                    continue

                logger.info(
                    f"[Backtest] {len(filtered)} signals passed risk filter on this bar."
                )

                # Simulate each surviving signal independently
                for sig in filtered:
                    trade_result = self._simulate_trade(sig, df, i)
                    self._record_trade(sig, trade_result)

            # Compute equity‑based stats
            self._calculate_final_metrics()
            logger.info("Backtest completed successfully")

        except Exception as e:
            logger.error(f"[Backtest] Error during run_backtest: {e}")

        return self.results

    # ------------------------------------------------------------------
    # Trade simulation & accounting
    # ------------------------------------------------------------------

    def _simulate_trade(self, signal: Signal, df: pd.DataFrame, idx: int) -> Dict:
        """Simulate execution of a single Signal on the subsequent bars.

        - Applies a simple MNQ cost model
        - Exits on first TP/SL hit, or at max horizon (time_exit)
        - Returns per‑trade PnL in dollars and R multiple
        """
        direction = 1 if signal.signal_type == "long" else -1

        # Entry with deterministic slippage (adverse)
        slippage_points = SLIPPAGE_TICKS * TICK_SIZE
        if direction == 1:
            actual_entry = signal.entry_price + slippage_points
        else:
            actual_entry = signal.entry_price - slippage_points

        stop = float(signal.stop_loss)
        tp = float(signal.take_profit)

        # Look ahead up to MAX_HOLD_BARS
        future = df.iloc[idx + 1 : idx + 1 + MAX_HOLD_BARS]
        if future.empty:
            exit_price = df["close"].iloc[idx]
            exit_reason = "no_future_bars"
        else:
            exit_price = future["close"].iloc[-1]
            exit_reason = "time_exit"

            for _, bar in future.iterrows():
                high = float(bar["high"])
                low = float(bar["low"])

                if direction == 1:
                    sl_hit = low <= stop
                    tp_hit = high >= tp
                else:
                    sl_hit = high >= stop
                    tp_hit = low <= tp

                # Conservative assumption if both touched: SL first
                if sl_hit:
                    exit_price = stop
                    exit_reason = "sl"
                    break
                if tp_hit:
                    exit_price = tp
                    exit_reason = "tp"
                    break

        # PnL in points
        if direction == 1:
            points = exit_price - actual_entry
        else:
            points = actual_entry - exit_price

        risk_points = abs(signal.entry_price - signal.stop_loss)
        r_mult = points / risk_points if risk_points > 0 else 0.0

        gross_pnl = points * POINT_VALUE  # 1 MNQ contract
        round_turn_commission = 2.0 * COMMISSION_PER_SIDE
        net_pnl = gross_pnl - round_turn_commission

        return {
            "executed": True,
            "pnl": float(net_pnl),
            "r_multiple": float(r_mult),
            "exit_reason": exit_reason,
            "entry_price": float(actual_entry),
            "exit_price": float(exit_price),
            "direction": signal.signal_type,
        }

    def _record_trade(self, signal: Signal, result: Dict) -> None:
        if not result.get("executed", False):
            return

        pnl = float(result.get("pnl", 0.0))
        r_mult = float(result.get("r_multiple", 0.0))
        exit_reason = result.get("exit_reason", "")

        self.results["total_trades"] += 1
        self.results["total_pnl"] += pnl

        if pnl > 0:
            self.results["winning_trades"] += 1
        elif pnl < 0:
            self.results["losing_trades"] += 1

        if signal.signal_type == "long":
            self.results["long_trades"] += 1
        else:
            self.results["short_trades"] += 1

        if exit_reason == "tp":
            self.results["tp_hits"] += 1
        elif exit_reason == "sl":
            self.results["sl_hits"] += 1

        self.trade_pnls.append(pnl)
        self.trade_r.append(r_mult)
        self.trade_timestamps.append(signal.timestamp)

        # Update equity curve
        self.equity_curve.append(self.equity_curve[-1] + pnl)

    # ------------------------------------------------------------------
    # Final metrics
    # ------------------------------------------------------------------

    def _calculate_final_metrics(self) -> None:
        """Compute win‑rate, avg R, Sharpe, and max drawdown from equity."""
        try:
            total_trades = int(self.results["total_trades"])
            wins = int(self.results["winning_trades"])

            if total_trades > 0:
                self.results["win_rate"] = wins / total_trades

            if self.trade_r:
                self.results["avg_r_multiple"] = float(np.mean(self.trade_r))

            eq = np.asarray(self.equity_curve, dtype=float)
            if eq.size > 1:
                running_max = np.maximum.accumulate(eq)
                drawdowns = (eq - running_max) / running_max
                self.results["max_drawdown"] = float(drawdowns.min())  # negative number

                # Per‑trade returns for Sharpe (simple proxy)
                rets = np.diff(eq) / eq[:-1]
                if rets.size > 1 and np.std(rets) > 1e-8:
                    sharpe = (np.mean(rets) / np.std(rets)) * np.sqrt(252.0)
                    self.results["sharpe_ratio"] = float(sharpe)
                else:
                    self.results["sharpe_ratio"] = 0.0

        except Exception as e:
            logger.error(f"[Backtest] Error calculating final metrics: {e}")
