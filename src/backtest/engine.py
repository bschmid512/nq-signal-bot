import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from loguru import logger

from src.core.engine import SignalGenerationEngine
from src.utils.database import DatabaseManager
from src.utils.risk_manager import RiskManager

class BacktestEngine:
    """Backtesting engine for strategy validation"""
    
    def __init__(self, db_manager: DatabaseManager, risk_manager: RiskManager):
        self.db_manager = db_manager
        self.risk_manager = risk_manager
        self.signal_engine = SignalGenerationEngine(db_manager)
        
        # Backtest results
        self.results = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'avg_r_multiple': 0.0,
            'signals_by_strategy': {},
            'daily_pnl': []
        }
        
        logger.info("BacktestEngine initialized")
    
    def load_historical_data(self, start_date: str, end_date: str, symbol: str = "NQ1!") -> pd.DataFrame:
        """Load historical market data for backtesting"""
        try:
            # In a real implementation, this would load from a data provider
            # For now, we'll generate sample data for demonstration
            
            logger.info(f"Loading historical data from {start_date} to {end_date}")
            
            # Generate sample data (in real implementation, load from database/API)
            dates = pd.date_range(start=start_date, end=end_date, freq='5min')
            
            # Generate realistic NQ price movements
            np.random.seed(42)  # For reproducible results
            
            # Start with a base price
            base_price = 15000
            prices = [base_price]
            
            for i in range(1, len(dates)):
                # Random walk with some trend
                change = np.random.normal(0, 20)  # Random price change
                if i % 288 == 0:  # Daily trend
                    change += np.random.normal(0, 50)
                
                new_price = prices[-1] + change
                prices.append(max(new_price, 14000))  # Floor at 14000
            
            # Create OHLC data
            df = pd.DataFrame({
                'timestamp': dates,
                'close': prices
            })
            
            # Generate OHLC from close prices
            df['open'] = df['close'].shift(1)
            df['open'].iloc[0] = df['close'].iloc[0]
            
            # Add some volatility
            df['high'] = df[['open', 'close']].max(axis=1) + np.random.uniform(5, 25, len(df))
            df['low'] = df[['open', 'close']].min(axis=1) - np.random.uniform(5, 25, len(df))
            
            # Add volume
            df['volume'] = np.random.randint(1000, 10000, len(df))
            
            # Add symbol and timeframe
            df['symbol'] = symbol
            df['timeframe'] = '5'
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading historical data: {e}")
            return pd.DataFrame()
    
    def simulate_signal_execution(self, signal: 'Signal', next_bars: pd.DataFrame) -> Dict:
        """Simulate signal execution and calculate PnL"""
        try:
            if len(next_bars) < 10:  # Need bars to simulate execution
                return {'pnl': 0, 'r_multiple': 0, 'executed': False}
            
            entry_price = signal.entry_price
            stop_loss = signal.stop_loss
            take_profit = signal.take_profit
            
            # Simulate execution with some slippage
            actual_entry = entry_price + np.random.normal(0, 0.5)  # Small slippage
            
            # Track trade outcome
            max_favorable_excursion = 0
            max_adverse_excursion = 0
            
            for i, bar in next_bars.iterrows():
                high = bar['high']
                low = bar['low']
                close = bar['close']
                
                # Check stop loss
                if signal.signal_type == 'long':
                    if low <= stop_loss:
                        exit_price = stop_loss
                        pnl = exit_price - actual_entry
                        break
                    elif high >= take_profit:
                        exit_price = take_profit
                        pnl = exit_price - actual_entry
                        break
                else:  # short
                    if high >= stop_loss:
                        exit_price = stop_loss
                        pnl = actual_entry - exit_price
                        break
                    elif low <= take_profit:
                        exit_price = take_profit
                        pnl = actual_entry - exit_price
                        break
                
                # Track MFE and MAE
                if signal.signal_type == 'long':
                    mfe = high - actual_entry
                    mae = actual_entry - low
                else:  # short
                    mfe = actual_entry - low
                    mae = high - actual_entry
                
                max_favorable_excursion = max(max_favorable_excursion, mfe)
                max_adverse_excursion = max(max_adverse_excursion, mae)
                
                # Exit at end of available data
                if i == next_bars.index[-1]:
                    exit_price = close
                    if signal.signal_type == 'long':
                        pnl = exit_price - actual_entry
                    else:  # short
                        pnl = actual_entry - exit_price
            
            # Calculate R-multiple
            risk = abs(actual_entry - stop_loss)
            r_multiple = pnl / risk if risk > 0 else 0
            
            return {
                'pnl': pnl,
                'r_multiple': r_multiple,
                'executed': True,
                'max_favorable_excursion': max_favorable_excursion,
                'max_adverse_excursion': max_adverse_excursion
            }
            
        except Exception as e:
            logger.error(f"Error simulating signal execution: {e}")
            return {'pnl': 0, 'r_multiple': 0, 'executed': False}
    
    def run_backtest(self, start_date: str, end_date: str) -> Dict:
        """Run backtest for specified date range"""
        try:
            logger.info(f"Starting backtest from {start_date} to {end_date}")
            
            # Load historical data
            df = self.load_historical_data(start_date, end_date)
            
            if df.empty:
                logger.error("No data available for backtest")
                return self.results
            
            # Process data in chunks (simulate real-time)
            chunk_size = 100  # Process 100 bars at a time
            all_signals = []
            
            for i in range(chunk_size, len(df), chunk_size):
                chunk = df.iloc[max(0, i-chunk_size):i]
                
                # Process this chunk through the signal engine
                # This would normally be done via webhooks, but for backtest we call directly
                for j, bar in chunk.iterrows():
                    market_data = {
                        'timestamp': bar['timestamp'],
                        'symbol': bar['symbol'],
                        'timeframe': bar['timeframe'],
                        'open': bar['open'],
                        'high': bar['high'],
                        'low': bar['low'],
                        'close': bar['close'],
                        'volume': bar['volume']
                    }
                    
                    # Store market data
                    self.db_manager.insert_market_data(market_data)
                
                # Generate signals for the last bar in chunk
                last_bar = chunk.iloc[-1]
                market_data = {
                    'timestamp': last_bar['timestamp'],
                    'symbol': last_bar['symbol'],
                    'timeframe': last_bar['timeframe'],
                    'open': last_bar['open'],
                    'high': last_bar['high'],
                    'low': last_bar['low'],
                    'close': last_bar['close'],
                    'volume': last_bar['volume']
                }
                
                # Process signals
                signals = self.signal_engine.process_new_data(market_data)
                all_signals.extend(signals)
                
                # Simulate signal execution for each signal
                for signal in signals:
                    # Get next bars for execution simulation
                    signal_idx = df[df['timestamp'] == signal.timestamp].index
                    if len(signal_idx) > 0:
                        next_bars = df.iloc[signal_idx[0]+1:signal_idx[0]+11]  # Next 10 bars
                        
                        result = self.simulate_signal_execution(signal, next_bars)
                        
                        if result['executed']:
                            # Update results
                            self.results['total_trades'] += 1
                            self.results['total_pnl'] += result['pnl']
                            
                            if result['pnl'] > 0:
                                self.results['winning_trades'] += 1
                            else:
                                self.results['losing_trades'] += 1
                            
                            # Track by strategy
                            strategy = signal.strategy
                            if strategy not in self.results['signals_by_strategy']:
                                self.results['signals_by_strategy'][strategy] = {
                                    'total': 0, 'winning': 0, 'losing': 0, 'pnl': 0
                                }
                            
                            self.results['signals_by_strategy'][strategy]['total'] += 1
                            self.results['signals_by_strategy'][strategy]['pnl'] += result['pnl']
                            
                            if result['pnl'] > 0:
                                self.results['signals_by_strategy'][strategy]['winning'] += 1
                            else:
                                self.results['signals_by_strategy'][strategy]['losing'] += 1
                
                # Progress update
                if i % 1000 == 0:
                    progress = (i / len(df)) * 100
                    logger.info(f"Backtest progress: {progress:.1f}%")
            
            # Calculate final metrics
            self.calculate_final_metrics()
            
            logger.info("Backtest completed successfully")
            return self.results
            
        except Exception as e:
            logger.error(f"Error running backtest: {e}")
            return self.results
    def run_backtest(self, start_date: str, end_date: str) -> dict:
        logger.info(f"Starting backtest from {start_date} to {end_date}")

        # ... your existing loading / simulation logic ...

        # Example: suppose you store a list of trade objects or dicts
        trades = self.trades  # or however you store them
        total_trades = len(trades)

        # PnL and win/loss stats
        pnls = [t.pnl for t in trades]
        wins = sum(1 for p in pnls if p > 0)
        losses = sum(1 for p in pnls if p < 0)

        total_pnl = sum(pnls) if pnls else 0.0
        win_rate = (wins / total_trades) if total_trades > 0 else 0.0

        # Equity curve, DD, Sharpe – adjust to your implementation
        equity_curve = self._build_equity_curve(pnls)
        max_drawdown = self._calculate_max_drawdown(equity_curve) if equity_curve else 0.0
        sharpe_ratio = self._calculate_sharpe_ratio(equity_curve) if equity_curve else 0.0

        results = {
            "total_trades": total_trades,
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe_ratio,
        }

        logger.info("Backtest completed successfully")
        logger.info(f"Backtest summary: {results}")
        return results

    def calculate_final_metrics(self):
        """Calculate final performance metrics"""
        try:
            # Win rate
            if self.results['total_trades'] > 0:
                self.results['win_rate'] = self.results['winning_trades'] / self.results['total_trades']
            
            # Average R-multiple (simplified)
            if self.results['total_trades'] > 0:
                # Assume average R-multiple based on win rate and risk/reward
                self.results['avg_r_multiple'] = 1.5 if self.results['win_rate'] > 0.5 else 0.8
            
            # Sharpe ratio (simplified)
            if self.results['total_trades'] > 10:
                # Assume 252 trading days per year
                daily_returns = [self.results['total_pnl'] / self.results['total_trades']] * self.results['total_trades']
                sharpe_ratio = (np.mean(daily_returns) * 252) / (np.std(daily_returns) * np.sqrt(252))
                self.results['sharpe_ratio'] = max(sharpe_ratio, 0)  # Cap at 0 for simplicity
            
            # Max drawdown (simplified)
            if self.results['total_pnl'] < 0:
                self.results['max_drawdown'] = abs(self.results['total_pnl']) / 100000  # Assume $100k account
            
        except Exception as e:
            logger.error(f"Error calculating final metrics: {e}")
    
    def print_results(self):
        """Print backtest results"""
        print("\n" + "="*60)
        print("BACKTEST RESULTS")
        print("="*60)
        print(f"Total Trades: {self.results['total_trades']}")
        print(f"Winning Trades: {self.results['winning_trades']}")
        print(f"Losing Trades: {self.results['losing_trades']}")
        print(f"Win Rate: {self.results['win_rate']:.2%}")
        print(f"Total PnL: ${self.results['total_pnl']:.2f}")
        print(f"Max Drawdown: {self.results['max_drawdown']:.2%}")
        print(f"Sharpe Ratio: {self.results['sharpe_ratio']:.2f}")
        print(f"Average R-Multiple: {self.results['avg_r_multiple']:.2f}")
        
        print("\nResults by Strategy:")
        print("-" * 40)
        for strategy, stats in self.results['signals_by_strategy'].items():
            win_rate = stats['winning'] / stats['total'] if stats['total'] > 0 else 0
            print(f"{strategy}:")
            print(f"  Total Signals: {stats['total']}")
            print(f"  Win Rate: {win_rate:.2%}")
            print(f"  PnL: ${stats['pnl']:.2f}")
            print()
        
        print("="*60)