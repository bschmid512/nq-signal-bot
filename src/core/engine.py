from __future__ import annotations
import asyncio
import pandas as pd
import numpy as np
import pandas_ta as ta  # FIX: Added missing import
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from loguru import logger

from src.utils.risk_manager import RiskManager
from config.config import config
from src.utils.database import DatabaseManager
from src.indicators.custom import CustomIndicators
from src.core.models import Signal

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
        """Initialize only HIGH-PERFORMING strategies"""
        from src.strategies.divergence import DivergenceStrategy
        from src.strategies.ema_pullback import EMAPullbackStrategy
        from src.strategies.htf_supertrend import HTFSupertrendStrategy
        from src.strategies.reversal_breakout import ReversalBreakoutStrategy
        
        strategy_configs = config.STRATEGIES
        
        # Only enable best performing strategies
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
        """Process new market data and generate HIGH-CONFIDENCE signals"""
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
            
            # Keep only last 500 bars in cache (reduced for performance)
            if len(self.data_cache[cache_key]) > 500:
                self.data_cache[cache_key] = self.data_cache[cache_key][-500:]
            
            # Generate signals
            signals = await self._generate_signals(symbol, timeframe)
            
            # Filter for HIGH CONFIDENCE only (>= 0.65)
            high_confidence_signals = [s for s in signals if s.confidence >= 0.65]
            
            # Remove duplicates
            unique_signals = self._remove_duplicate_signals(high_confidence_signals)
            
            # Store signals
            for signal in unique_signals:
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
            
            if unique_signals:
                logger.info(f"🎯 Generated {len(unique_signals)} HIGH-CONFIDENCE signals (min conf: {min(s.confidence for s in unique_signals):.2f})")
                for sig in unique_signals:
                    logger.info(f"  📊 {sig.strategy} {sig.signal_type.upper()} @ {sig.entry_price:.2f} (Conf: {sig.confidence:.2f}, RR: {sig.risk_reward:.1f})")
            
            return unique_signals
            
        except Exception as e:
            logger.error(f"Error processing new data: {e}")
            return []
    
    async def _generate_signals(self, symbol: str, timeframe: str) -> List[Signal]:
        """Generate signals with market structure analysis"""
        signals = []
        
        # Get historical data for analysis
        df = self._get_analysis_dataframe(symbol, timeframe)
        
        if df.empty or len(df) < 100:  # Need more data for proper analysis
            return signals
        
        # Calculate comprehensive indicators
        df_with_indicators = self._calculate_indicators(df)
        
        # Analyze market structure
        market_condition = self._analyze_market_structure(df_with_indicators)
        logger.info(f"Market condition: {market_condition}")
        
        # Generate signals from strategies
        for strategy_name, strategy in self.strategies.items():
            try:
                strategy_signals = strategy.generate_signals(df_with_indicators, symbol)
                
                # Apply market structure boost
                for signal in strategy_signals:
                    if market_condition == 'strong_trend':
                        if strategy_name in ['ema_pullback', 'htf_supertrend']:
                            signal.confidence += 0.15  # Boost trend strategies in trends
                    elif market_condition == 'reversal':
                        if strategy_name == 'reversal_breakout':
                            signal.confidence += 0.20  # Boost reversal strategy at turns
                    
                    # Cap at 0.95
                    signal.confidence = min(signal.confidence, 0.95)
                
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
                df['bb_upper'] = bb.iloc[:, 0]  # Upper band
                df['bb_middle'] = bb.iloc[:, 1]  # Middle band
                df['bb_lower'] = bb.iloc[:, 2]  # Lower band
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
            if 'vwap' in df.columns:
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
            # Return df with whatever indicators succeeded
            return df
    
    def _analyze_market_structure(self, df: pd.DataFrame) -> str:
        """Analyze market structure for better signal context"""
        if df.empty or len(df) < 50:
            return 'unknown'
        
        try:
            current_bar = df.iloc[-1]
            
            # Check trend strength
            if 'adx' in current_bar and not pd.isna(current_bar['adx']):
                adx = current_bar['adx']
                
                # Strong trend
                if adx > 30:
                    # Check direction
                    if 'ema_21' in current_bar and 'ema_50' in current_bar:
                        if current_bar['ema_21'] > current_bar['ema_50']:
                            return 'strong_trend'
                        else:
                            return 'strong_trend'
                
                # Ranging/Choppy
                elif adx < 20:
                    return 'ranging'
            
            # Check for potential reversal
            if 'rsi' in current_bar:
                rsi = current_bar['rsi']
                if not pd.isna(rsi):
                    if rsi < 25 or rsi > 75:
                        return 'reversal'
            
            # Check volatility
            if 'atr' in current_bar:
                atr = current_bar['atr']
                if not pd.isna(atr):
                    # High volatility environment
                    if atr > 20:
                        return 'high_volatility'
                    # Low volatility (potential breakout)
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
            
            # Check against recent signals
            for recent in self.recent_signals[-10:]:  # Last 10 signals
                if (signal.strategy == recent.strategy and
                    signal.signal_type == recent.signal_type and
                    abs(signal.entry_price - recent.entry_price) < 5 and  # Within 5 points
                    (signal.timestamp - recent.timestamp).seconds < 300):  # Within 5 minutes
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_signals.append(signal)
                self.recent_signals.append(signal)
        
        # Keep only last 20 in memory
        if len(self.recent_signals) > 20:
            self.recent_signals = self.recent_signals[-20:]
        
        return unique_signals
    
    def get_strategy_performance(self, strategy_name: str, days: int = 7) -> Dict:
        """Get recent performance metrics for a specific strategy"""
        try:
            signals = self.db_manager.get_recent_signals(strategy_name, limit=50)
            
            if signals.empty:
                return {'total_signals': 0, 'avg_confidence': 0}
            
            # Get recent signals only
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
            self.db_manager.cleanup_old_data(days_to_keep=7)  # Keep less data
            logger.info("Database cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")