import os
from dataclasses import dataclass, field
from typing import Dict, List
from dotenv import load_dotenv

load_dotenv()

@dataclass
class TradingConfig:
    # Market Configuration
    SYMBOL: str = "MNQ1!"
    PRIMARY_TIMEFRAME: str = "5"
    EXECUTION_TIMEFRAME: str = "1"
    HTF_TIMEFRAME: str = "15"
    
    # Risk Management
    MAX_CONCURRENT_POSITIONS: int = 1
    MAX_DAILY_LOSS_R: float = 3.0
    MAX_DAILY_LOSS_DOLLAR: float = 1500.0
    MIN_ATR_FOR_TRADING: float = 0.0
    
    # Database
    DATABASE_PATH: str = "data/nq_signals.db"
    
    # Webhook Server
    WEBHOOK_HOST: str = "0.0.0.0"
    WEBHOOK_PORT: int = 8000
    WEBHOOK_SECRET: str = os.getenv("WEBHOOK_SECRET", "your_webhook_secret_here")
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/nq_bot.log"
    
    # Strategy Configuration
# Strategy Configuration
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
            "stop_loss_atr": 1.0,
            "take_profit_atr": 1.5
        },
        "ema_pullback": {
            "enabled": True,
            "base_confidence": 0.6,
            "fast_ema": 21,
            "medium_ema": 50,
            "slow_ema": 200,
            "min_slope": 0.1,  # % per bar
            "volume_multiplier": 1.2,
            "atr_period": 14
        },
        "htf_supertrend": {
            "enabled": True,
            "base_confidence": 0.65,
            "htf_ema": 50,
            "supertrend_atr": 10,
            "supertrend_multiplier": 3.0,
            "news_buffer_minutes": 2,
            "max_stop_atr": 1.2
        },
        "supply_demand": {
            "enabled": True,
            "base_confidence": 0.6,
            "pivot_strength": 8,
            "min_impulse_atr": 1.5,
            "zone_buffer": 2.0,
            "max_touches": 2,
            "stop_buffer_atr": 0.5
        },
        "vwap": {
            "enabled": True,
            "base_confidence": 0.55,
            "session_start": "09:30",
            "band_std_dev": [1.0, 2.0],
            "adx_period": 14,
            "adx_threshold": 18,
            "bb_period": 60
        },
        "orb": {
            "enabled": True,
            "base_confidence": 0.7,
            "range_minutes": 15,
            "buffer_atr_factor": 0.15,
            "min_buffer_points": 3,
            "volume_confirm": 1.5,
            "max_gap_atr": 0.7,
            "alternative_range": 5
        }
    })

# Global configuration instance
config = TradingConfig()