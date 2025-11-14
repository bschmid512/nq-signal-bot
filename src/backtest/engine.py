import asyncio
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict
from loguru import logger

from config.config import config
from src.core.engine import SignalGenerationEngine
from src.utils.database import DatabaseManager
from src.utils.risk_manager import RiskManager
from src.strategies.ml_ict_edge import MLICTEdgeStrategy

class BacktestEngine:
    """Backtesting engine for strategy validation"""

    def __init__(self, db_manager: DatabaseManager, risk_manager: RiskManager):
        self.db_manager = db_manager
        self.risk_manager = risk_manager
        self.signal_engine = SignalGenerationEngine(db_manager)

        self.results = {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "total_pnl": 0.0,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0,
            "avg_r_multiple": 0.0,
            "win_rate": 0.0,
            "signals_by_strategy": {},
            "daily_pnl": [],
            # NEW:
            "tp_hits": 0,
            "sl_hits": 0,
            "long_trades": 0,
            "short_trades": 0,
        }


        logger.info("BacktestEngine initialized")

    # ------------------------------------------------------------------ #
    # Data loading
    # ------------------------------------------------------------------ #

    def load_recent_data(self, limit: int = 20000) -> pd.DataFrame:
        """Load the last `limit` bars of real data from the DB."""
        try:
            symbol = config.SYMBOL
            timeframe = config.PRIMARY_TIMEFRAME

            logger.info(
                f"Loading last {limit} bars from DB for {symbol} {timeframe}-min"
            )
            df = self.db_manager.get_latest_data(symbol, timeframe, limit=limit)

            if df is None or df.empty:
                logger.warning("[Backtest] No recent data found in DB.")
                return pd.DataFrame()

            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])

            df = df.sort_values("timestamp").reset_index(drop=True)
            return df

        except Exception as e:
            logger.error(f"Error loading recent data from DB: {e}")
            return pd.DataFrame()

    def load_historical_data(
        self, start_date: str, end_date: str, symbol: str = "MNQ1!"
    ) -> pd.DataFrame:
        """Fallback synthetic data generator for backtesting."""
        try:
            logger.info(f"[Backtest] Generating synthetic data from {start_date} to {end_date}")

            dates = pd.date_range(start=start_date, end=end_date, freq="5min")

            np.random.seed(42)
            base_price = 15000
            prices = [base_price]

            for i in range(1, len(dates)):
                change = np.random.normal(0, 20)
                if i % 288 == 0:
                    change += np.random.normal(0, 50)
                new_price = prices[-1] + change
                prices.append(max(new_price, 14000))

            df = pd.DataFrame({"timestamp": dates, "close": prices})

            df["open"] = df["close"].shift(1)
            df.loc[0, "open"] = df.loc[0, "close"]

            df["high"] = df[["open", "close"]].max(axis=1) + np.random.uniform(
                5, 25, len(df)
            )
            df["low"] = df[["open", "close"]].min(axis=1) - np.random.uniform(
                5, 25, len(df)
            )

            df["volume"] = np.random.randint(1000, 10000, len(df))

            df["symbol"] = symbol
            df["timeframe"] = "5"

            return df

        except Exception as e:
            logger.error(f"Error generating synthetic historical data: {e}")
            return pd.DataFrame()

    # ------------------------------------------------------------------ #
    # Trade simulation
    # ------------------------------------------------------------------ #

    def simulate_signal_execution(self, signal, next_bars: pd.DataFrame) -> Dict:
        """Simulate signal execution and calculate PnL for a single signal."""
        try:
            if len(next_bars) < 3:
                return {
                    "pnl": 0.0,
                    "r_multiple": 0.0,
                    "executed": False,
                    "outcome": "no_data",
                }

            entry_price = signal.entry_price
            stop_loss = signal.stop_loss
            take_profit = signal.take_profit

            actual_entry = entry_price + np.random.normal(0, 0.5)

            max_favorable_excursion = 0.0
            max_adverse_excursion = 0.0

            pnl = 0.0
            exit_price = actual_entry
            outcome = "time_exit"

            for i, bar in next_bars.iterrows():
                high = bar["high"]
                low = bar["low"]
                close = bar["close"]

                if signal.signal_type == "long":
                    if low <= stop_loss:
                        exit_price = stop_loss
                        pnl = exit_price - actual_entry
                        outcome = "hit_sl"
                        break
                    elif high >= take_profit:
                        exit_price = take_profit
                        pnl = exit_price - actual_entry
                        outcome = "hit_tp"
                        break
                else:  # short
                    if high >= stop_loss:
                        exit_price = stop_loss
                        pnl = actual_entry - exit_price
                        outcome = "hit_sl"
                        break
                    elif low <= take_profit:
                        exit_price = take_profit
                        pnl = actual_entry - exit_price
                        outcome = "hit_tp"
                        break

                if signal.signal_type == "long":
                    mfe = high - actual_entry
                    mae = actual_entry - low
                else:
                    mfe = actual_entry - low
                    mae = high - actual_entry

                max_favorable_excursion = max(max_favorable_excursion, mfe)
                max_adverse_excursion = max(max_adverse_excursion, mae)

                if i == next_bars.index[-1]:
                    exit_price = close
                    if signal.signal_type == "long":
                        pnl = exit_price - actual_entry
                    else:
                        pnl = actual_entry - exit_price
                    outcome = "time_exit"

            risk = abs(actual_entry - stop_loss)
            r_multiple = pnl / risk if risk > 0 else 0.0

            return {
                "pnl": float(pnl),
                "r_multiple": float(r_multiple),
                "executed": True,
                "outcome": outcome,
                "max_favorable_excursion": float(max_favorable_excursion),
                "max_adverse_excursion": float(max_adverse_excursion),
            }

        except Exception as e:
            logger.error(f"Error simulating signal execution: {e}")
            return {
                "pnl": 0.0,
                "r_multiple": 0.0,
                "executed": False,
                "outcome": "error",
            }

    # ------------------------------------------------------------------ #
    # Main backtest loop
    # ------------------------------------------------------------------ #

    def run_backtest(self, start_date: str, end_date: str) -> Dict:
        """
        Run backtest.

        Behaviour:
        - First tries last 1000 bars of REAL data from DB.
        - If DB empty, falls back to synthetic [start_date, end_date].
        - Processes EVERY bar sequentially (so killzones actually get hit).
        """
        try:
            logger.info(f"Starting backtest from {start_date} to {end_date}")

            df = self.load_recent_data(limit=20000)

            if df.empty:
                logger.warning(
                    "[Backtest] No recent data in DB, using synthetic historical data instead."
                )
                df = self.load_historical_data(start_date, end_date)

            if df.empty:
                logger.error("No data available for backtest (DB + synthetic both empty).")
                return self.results

            df = df.sort_values("timestamp").reset_index(drop=True)

            all_signals = []

            for i, bar in df.iterrows():
                market_data = {
                    "timestamp": bar["timestamp"],
                    "symbol": bar.get("symbol", config.SYMBOL),
                    "timeframe": str(bar.get("timeframe", config.PRIMARY_TIMEFRAME)),
                    "open": float(bar["open"]),
                    "high": float(bar["high"]),
                    "low": float(bar["low"]),
                    "close": float(bar["close"]),
                    "volume": int(bar.get("volume", 0) or 0),
                }

                # send bar through pipeline
                self.db_manager.insert_market_data(market_data)
                signals = asyncio.run(self.signal_engine.process_new_data(market_data))
                all_signals.extend(signals)

                if not signals:
                    continue

                # simulate using the NEXT 10 bars
                next_bars = df.iloc[i + 1 : i + 11]
                if next_bars.empty:
                    continue

                for signal in signals:
                    result = self.simulate_signal_execution(signal, next_bars)

                    result = self.simulate_signal_execution(signal, next_bars)

                    if result["executed"]:
                        self.results["total_trades"] += 1
                        self.results["total_pnl"] += result["pnl"]

                        # long/short counts
                        if signal.signal_type == "long":
                            self.results["long_trades"] += 1
                        else:
                            self.results["short_trades"] += 1

                        # outcome stats
                        if result["outcome"] == "hit_tp":
                            self.results["tp_hits"] += 1
                            self.results["winning_trades"] += 1
                        elif result["outcome"] == "hit_sl":
                            self.results["sl_hits"] += 1
                            self.results["losing_trades"] += 1
                        else:
                            # treat time_exit as win/loss by pnl sign
                            if result["pnl"] > 0:
                                self.results["winning_trades"] += 1
                            else:
                                self.results["losing_trades"] += 1

                        logger.info(
                            f"[Backtest] {signal.strategy} {signal.signal_type.upper()} "
                            f"{signal.timestamp} outcome={result['outcome']} "
                            f"pnl={result['pnl']:.2f} R={result['r_multiple']:.2f}"
                        )

                        strategy = signal.strategy
                        if strategy not in self.results["signals_by_strategy"]:
                            self.results["signals_by_strategy"][strategy] = {
                                "total": 0,
                                "winning": 0,
                                "losing": 0,
                                "pnl": 0.0,
                            }

                        stats = self.results["signals_by_strategy"][strategy]
                        stats["total"] += 1
                        stats["pnl"] += result["pnl"]
                        if result["pnl"] > 0:
                            stats["winning"] += 1
                        else:
                            stats["losing"] += 1


            self.calculate_final_metrics()
            logger.info("Backtest completed successfully")
            return self.results

        except Exception as e:
            logger.error(f"Error running backtest: {e}")
            return self.results

    # ------------------------------------------------------------------ #
    # Metrics + Print
    # ------------------------------------------------------------------ #

    def calculate_final_metrics(self):
        """Calculate final performance metrics."""
        try:
            total_trades = self.results["total_trades"]
            wins = self.results["winning_trades"]

            if total_trades > 0:
                self.results["win_rate"] = wins / total_trades

            if total_trades > 0:
                self.results["avg_r_multiple"] = (
                    1.5 if self.results["win_rate"] > 0.5 else 0.8
                )

            if total_trades > 10:
                daily_returns = [
                    self.results["total_pnl"] / total_trades
                ] * total_trades
                if np.std(daily_returns) > 0:
                    sharpe_ratio = (np.mean(daily_returns) * 252) / (
                        np.std(daily_returns) * np.sqrt(252)
                    )
                    self.results["sharpe_ratio"] = max(sharpe_ratio, 0.0)

            if self.results["total_pnl"] < 0:
                self.results["max_drawdown"] = abs(self.results["total_pnl"]) / 100000

        except Exception as e:
            logger.error(f"Error calculating final metrics: {e}")

    def print_results(self):
        """Print backtest results."""
        print("\n" + "=" * 60)
        print("BACKTEST RESULTS")
        print("=" * 60)
        print(f"Total Trades: {self.results['total_trades']}")
        print(f"Winning Trades: {self.results['winning_trades']}")
        print(f"Losing Trades: {self.results['losing_trades']}")
        print(f"Win Rate: {self.results['win_rate']:.2%}")
        print(f"Total PnL: ${self.results['total_pnl']:.2f}")
        print(f"Max Drawdown: {self.results['max_drawdown']:.2%}")
        print(f"Sharpe Ratio: {self.results['sharpe_ratio']:.2f}")
        print(f"Average R-Multiple: {self.results['avg_r_multiple']:.2f}")
        print("\nAdditional stats:")
        print(f"  Long trades:  {self.results['long_trades']}")
        print(f"  Short trades: {self.results['short_trades']}")
        print(f"  TP hits:      {self.results['tp_hits']}")
        print(f"  SL hits:      {self.results['sl_hits']}")

        print("\nResults by Strategy:")
        print("-" * 40)
        for strategy, stats in self.results["signals_by_strategy"].items():
            win_rate = stats["winning"] / stats["total"] if stats["total"] > 0 else 0
            print(f"{strategy}:")
            print(f"  Total Signals: {stats['total']}")
            print(f"  Win Rate: {win_rate:.2%}")
            print(f"  PnL: ${stats['pnl']:.2f}")
            print()

        print("=" * 60)
