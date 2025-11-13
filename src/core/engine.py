from __future__ import annotations
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass  # <-- MAKE SURE THIS LINE EXISTS
from loguru import logger
from src.utils.risk_manager import RiskManager
from config.config import config
from src.utils.database import DatabaseManager
from src.indicators.custom import CustomIndicators

@dataclass
class SignalGenerationEngine:
    """Main engine for generating trading signals"""
    
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
        
        logger.info("SignalGenerationEngine initialized")
    
    def _init_strategies(self):
        """Initialize all trading strategies"""
        
        # --- ADD THESE IMPORTS HERE ---
        from src.strategies.divergence import DivergenceStrategy
        from src.strategies.ema_pullback import EMAPullbackStrategy
        from src.strategies.htf_supertrend import HTFSupertrendStrategy
        from src.strategies.supply_demand import SupplyDemandStrategy
        from src.strategies.vwap import VWAPStrategy
        from src.strategies.orb import ORBStrategy
        # --- END OF ADDED IMPORTS ---
        strategy_configs = config.STRATEGIES
        
        if strategy_configs['divergence']['enabled']:
            # Create a copy and remove the 'enabled' key
            conf = strategy_configs['divergence'].copy()
            conf.pop('enabled', None)
            self.strategies['divergence'] = DivergenceStrategy(conf)
            logger.info("Divergence strategy initialized")
        
        if strategy_configs['ema_pullback']['enabled']:
            # Create a copy and remove the 'enabled' key
            conf = strategy_configs['ema_pullback'].copy()
            conf.pop('enabled', None)
            self.strategies['ema_pullback'] = EMAPullbackStrategy(conf)
            logger.info("EMA Pullback strategy initialized")
        
        if strategy_configs['htf_supertrend']['enabled']:
            # Create a copy and remove the 'enabled' key
            conf = strategy_configs['htf_supertrend'].copy()
            conf.pop('enabled', None)
            self.strategies['htf_supertrend'] = HTFSupertrendStrategy(conf)
            logger.info("HTF Supertrend strategy initialized")
        
        if strategy_configs['supply_demand']['enabled']:
            # Create a copy and remove the 'enabled' key
            conf = strategy_configs['supply_demand'].copy()
            conf.pop('enabled', None)
            self.strategies['supply_demand'] = SupplyDemandStrategy(conf)
            logger.info("Supply/Demand strategy initialized")
        
        if strategy_configs['vwap']['enabled']:
            # Create a copy and remove the 'enabled' key
            conf = strategy_configs['vwap'].copy()
            conf.pop('enabled', None)
            self.strategies['vwap'] = VWAPStrategy(conf)
            logger.info("VWAP strategy initialized")
        
        if strategy_configs['orb']['enabled']:
            # Create a copy and remove the 'enabled' key
            conf = strategy_configs['orb'].copy()
            conf.pop('enabled', None)
            self.strategies['orb'] = ORBStrategy(conf)
            logger.info("ORB strategy initialized")
    
    async def process_new_data(self, market_data: Dict) -> List[Signal]:
        """Process new market data and generate signals"""
        try:
            symbol = market_data['symbol']
            timeframe = market_data['timeframe']
            
            # Store market data
            self.db_manager.insert_market_data(market_data)
            
            # Update data cache
            cache_key = f"{symbol}_{timeframe}"
            if cache_key not in self.data_cache:
                self.data_cache[cache_key] = []
            
            self.data_cache[cache_key].append(market_data)
            
            # Keep only last 1000 bars in cache
            if len(self.data_cache[cache_key]) > 1000:
                self.data_cache[cache_key] = self.data_cache[cache_key][-1000:]
            
            # Check if we should analyze (avoid over-analysis on 1-minute data)
            current_time = datetime.now()
            last_analysis = self.last_analysis_time.get(cache_key, datetime.min)
            
            if timeframe == '1' and (current_time - last_analysis).seconds < 5:
                return []
            
            self.last_analysis_time[cache_key] = current_time
            
            # Generate signals
            signals = await self._generate_signals(symbol, timeframe)
            
            # Apply risk management
            filtered_signals = self.risk_manager.filter_signals(signals)
            
            # Store signals
            for signal in filtered_signals:
                self.db_manager.insert_signal({
                    'timestamp': signal.timestamp,
                    'strategy': signal.strategy,
                    'signal_type': signal.signal_type,
                    'symbol': signal.symbol,
                    'entry_price': signal.entry_price,
                    'stop_loss': signal.stop_loss,
                    'take_profit': signal.take_profit,
                    'confidence': signal.confidence,
                    'risk_reward': signal.risk_reward,
                    'atr': signal.atr,
                    'volume': signal.volume,
                    'metadata': signal.metadata
                })
            
            return filtered_signals
            
        except Exception as e:
            logger.error(f"Error processing new data: {e}")
            return []
    
    async def _generate_signals(self, symbol: str, timeframe: str) -> List[Signal]:
        """Generate signals from all enabled strategies"""
        signals = []
        
        # Get historical data for analysis
        df = self._get_analysis_dataframe(symbol, timeframe)
        
        if df.empty or len(df) < 50:  # Need minimum data for analysis
            return signals
        
        # Calculate common indicators
        df_with_indicators = self._calculate_indicators(df)
        
        # Generate signals from each strategy
        for strategy_name, strategy in self.strategies.items():
            try:
                strategy_signals = strategy.generate_signals(df_with_indicators, symbol)
                signals.extend(strategy_signals)
            except Exception as e:
                logger.error(f"Error in {strategy_name} strategy: {e}")
        
        return signals
    
    def _get_analysis_dataframe(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Get market data as DataFrame for analysis"""
        # Try to get from cache first
        cache_key = f"{symbol}_{timeframe}"
        if cache_key in self.data_cache and len(self.data_cache[cache_key]) > 0:
            df = pd.DataFrame(self.data_cache[cache_key])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        
        # Get from database
        df = self.db_manager.get_latest_data(symbol, timeframe, limit=200)
        return df
    
    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators"""
        try:
            import pandas_ta as ta
            
            # Basic indicators
            df['rsi'] = ta.rsi(df['close'], length=14)
            df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
            
            # EMAs
            df['ema_21'] = ta.ema(df['close'], length=21)
            df['ema_50'] = ta.ema(df['close'], length=50)
            df['ema_200'] = ta.ema(df['close'], length=200)
            
            # MACD
            macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
            if macd is not None:
                df['macd'] = macd['MACD_12_26_9']
                df['macd_signal'] = macd['MACDs_12_26_9']
                df['macd_histogram'] = macd['MACDh_12_26_9']
            
            # Bollinger Bands
            # Bollinger Bands - manual calculation
            df['bb_middle'] = df['close'].rolling(window=20).mean()
            bb_std = df['close'].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2.0)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2.0)
            df['bb_width'] = df['bb_upper'] - df['bb_lower']
            
            # ADX
            adx = ta.adx(df['high'], df['low'], df['close'], length=14)
            if adx is not None:
                df['adx'] = adx['ADX_14']
            
            # Custom indicators
            df['vwap'] = self.custom_indicators.calculate_vwap(df)
            vwap_bands = self.custom_indicators.calculate_vwap_bands(df)
            df['vwap_upper_1'] = vwap_bands['upper_band_1.0']
            df['vwap_lower_1'] = vwap_bands['lower_band_1.0']
            df['vwap_upper_2'] = vwap_bands['upper_band_2.0']
            df['vwap_lower_2'] = vwap_bands['lower_band_2.0']
            
            # Supertrend
            # Supertrend - using pandas-ta (more reliable)
            try:
                st = ta.supertrend(df['high'], df['low'], df['close'], length=10, multiplier=3.0)
                if st is not None and not st.empty:
                    # Find supertrend column (name varies by pandas-ta version)
                    st_cols = [c for c in st.columns if 'SUPERT_' in c and 'SUPERTd' not in c and 'SUPERTl' not in c and 'SUPERTs' not in c]
                    df['supertrend'] = st[st_cols[0]] if st_cols else df['close']
                else:
                    df['supertrend'] = df['close']
            except:
                df['supertrend'] = df['close']

            # Ensure no NaN values
            df['supertrend'] = df['supertrend'].fillna(df['close'])
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return df
    
    def get_strategy_performance(self, strategy_name: str, days: int = 30) -> Dict:
        """Get performance metrics for a specific strategy"""
        try:
            signals = self.db_manager.get_recent_signals(strategy_name, limit=100)
            
            if signals.empty:
                return {'total_signals': 0, 'win_rate': 0, 'avg_r_multiple': 0}
            
            executed_signals = signals[signals['executed'] == True]
            
            if executed_signals.empty:
                return {
                    'total_signals': len(signals),
                    'executed_signals': 0,
                    'win_rate': 0,
                    'avg_r_multiple': 0,
                    'total_pnl': 0
                }
            
            # Calculate metrics
            winning_signals = executed_signals[executed_signals['pnl'] > 0]
            win_rate = len(winning_signals) / len(executed_signals) if len(executed_signals) > 0 else 0
            
            # Calculate R-multiples
            executed_signals['r_multiple'] = executed_signals['pnl'] / (executed_signals['entry_price'] - executed_signals['stop_loss']).abs()
            avg_r_multiple = executed_signals['r_multiple'].mean()
            
            total_pnl = executed_signals['pnl'].sum()
            
            return {
                'total_signals': len(signals),
                'executed_signals': len(executed_signals),
                'win_rate': win_rate,
                'avg_r_multiple': avg_r_multiple,
                'total_pnl': total_pnl
            }
            
        except Exception as e:
            logger.error(f"Error getting strategy performance: {e}")
            return {'total_signals': 0, 'win_rate': 0, 'avg_r_multiple': 0}
    
    def cleanup(self):
        """Clean up resources"""
        try:
            self.db_manager.cleanup_old_data(days_to_keep=30)
            logger.info("Database cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")