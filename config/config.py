import os
from dataclasses import dataclass, field
from typing import Dict, List
from dotenv import load_dotenv

load_dotenv()

@dataclass
class TradingConfig:
    # Market Configuration (NQ-Specific)
    SYMBOL: str = "MNQ1!"
    PRIMARY_TIMEFRAME: str = "5"  # Focus on 5-min for better signals
    EXECUTION_TIMEFRAME: str = "1"
    HTF_TIMEFRAME: str = "15"
    
    # Risk Management (Increased for NQ volatility)
    MAX_CONCURRENT_POSITIONS: int = 2  # Critical: was 1, now allows diversification
    MAX_DAILY_LOSS_R: float = 3.0
    MAX_DAILY_LOSS_DOLLAR: float = 1500.0
    MIN_ATR_FOR_TRADING: float = 8.0  # Skip dead periods
    
    # Database
    DATABASE_PATH: str = "data/nq_signals.db"
    
    # Webhook Server
    WEBHOOK_HOST: str = "0.0.0.0"
    WEBHOOK_PORT: int = 8000
    WEBHOOK_SECRET: str = os.getenv("WEBHOOK_SECRET", "your_webhook_secret_here")
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/nq_bot.log"
    
    # Strategy Configuration (Optimized for NQ)
    STRATEGIES: Dict[str, Dict] = field(default_factory=lambda: {
        "divergence": {
            "enabled": True,
            "base_confidence": 0.55,
            "rsi_period": 14,
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
            "min_divergence_points": 4,
            "pivot_strength": 5,
            "ema_period": 21,
            "atr_period": 14,
            "stop_loss_atr": 1.2,  # Wider for NQ
            "take_profit_atr": 2.5  # Minimum 2:1 ratio
        },
        "ema_pullback": {
            "enabled": True,
            "base_confidence": 0.6,
            "fast_ema": 21,
            "medium_ema": 50,
            "slow_ema": 200,
            "min_slope": 0.1,
            "volume_multiplier": 1.2,
            "atr_period": 14,
            "stop_loss_atr": 1.0,
            "take_profit_r_mult": 2.0  # Dynamic R-mult instead of fixed
        },
        "htf_supertrend": {
            "enabled": True,
            "base_confidence": 0.65,
            "htf_ema": 200,
            "supertrend_atr": 10,
            "supertrend_multiplier": 3.0,
            "news_buffer_minutes": 2,
            "max_stop_atr": 1.5,
            "take_profit_r_mult": 2.5
        },
        "supply_demand": {
            "enabled": False,  # Disabled: too subjective for NQ intraday
            "base_confidence": 0.6,
            "pivot_strength": 8,
            "min_impulse_atr": 1.5,
            "zone_buffer": 2.0,
            "max_touches": 2,
            "stop_buffer_atr": 0.5
        },
        "vwap": {
            "enabled": False,  # Disabled until VWAP calculation is fixed
            "base_confidence": 0.55,
            "session_start": "09:30",
            "band_std_dev": [1.0, 2.0],
            "adx_period": 14,
            "adx_threshold": 18,
            "bb_period": 60
        },
        "orb": {
            "enabled": False,  # Disabled initially: high variance
            "base_confidence": 0.7,
            "range_minutes": 5,  # Faster for NQ
            "buffer_atr_factor": 0.15,
            "min_buffer_points": 3.0,
            "volume_confirm": 1.5,
            "max_gap_atr": 0.7,
            "alternative_range": 5
        },
        # In config.py, add to STRATEGIES dict:
        "reversal_breakout": {
            "enabled": True,
            "base_confidence": 0.65,
            "rsi_oversold": 25,
            "rsi_overbought": 75,
            "stop_loss_atr": 1.0,
            "take_profit_r_mult": 2.5
        }
    })

# Global configuration instance
config = TradingConfig()