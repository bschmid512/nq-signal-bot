import os
from dataclasses import dataclass, field
from typing import Dict, List
from dotenv import load_dotenv

load_dotenv()

@dataclass
class TradingConfig:
    # Market Configuration (NQ-Specific)
    SYMBOL: str = "MNQ1!"
    PRIMARY_TIMEFRAME: str = "5"   # 5-min for signal generation
    EXECUTION_TIMEFRAME: str = "1" # 1-min for precise entries
    HTF_TIMEFRAME: str = "15"
    ML_EDGE_MODEL_PATH: str = "models/mnq_ict_edge_xgb.pkl"
    ML_EDGE_THRESHOLD: float = 0.45
    # Risk Management (SIMPLIFIED - Quality over Quantity)
    MAX_CONCURRENT_POSITIONS: int = 3  # Allow more flexibility
    MIN_ATR_FOR_TRADING: float = 5.0   # Lower threshold for NQ
    
    # Database
    DATABASE_PATH: str = "data/nq_signals.db"
    
    # Webhook Server
    WEBHOOK_HOST: str = "0.0.0.0"
    WEBHOOK_PORT: int = 8000
    WEBHOOK_SECRET: str = os.getenv("WEBHOOK_SECRET", "your_webhook_secret_here")
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/nq_bot.log"
    
    # Strategy Configuration (HIGH-CONFIDENCE SETUPS ONLY)
    STRATEGIES: Dict[str, Dict] = field(default_factory=lambda: {
        "divergence": {
            "enabled": False,  # Disabled - too many false signals
            "base_confidence": 0.50,
            "rsi_period": 14,
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
            "min_divergence_points": 4,
            "pivot_strength": 5,
            "ema_period": 21,
            "atr_period": 14,
            "stop_loss_atr": 1.2,
            "take_profit_atr": 3.0  # Higher targets
        },
        "ema_pullback": {
            "enabled": False,  # disabled in favor of master strategy
            "base_confidence": 0.70,  # Increased base confidence
            "fast_ema": 21,
            "medium_ema": 50,
            "slow_ema": 200,
            "min_slope": 0.15,  # Stronger trend requirement
            "volume_multiplier": 1.3,  # Higher volume requirement
            "atr_period": 14,
            "stop_loss_atr": 0.8,  # Tighter stop
            "take_profit_r_mult": 2.5  # Better R:R
        },
        "htf_supertrend": {
            "enabled": False,  # disabled in favor of master strategy
            "base_confidence": 0.75,  # High base confidence
            "htf_ema": 200,
            "supertrend_atr": 10,
            "supertrend_multiplier": 2.5,  # Tighter bands
            "news_buffer_minutes": 2,
            "max_stop_atr": 1.2,
            "take_profit_r_mult": 3.0  # High reward targets
        },
        "supply_demand": {
            "enabled": False,  # Disabled - needs manual zone identification
            "base_confidence": 0.60,
            "pivot_strength": 8,
            "min_impulse_atr": 1.5,
            "zone_buffer": 2.0,
            "max_touches": 2,
            "stop_buffer_atr": 0.5
        },
        "vwap": {
            "enabled": False,  # Disabled - better for ranging markets
            "base_confidence": 0.55,
            "session_start": "09:30",
            "band_std_dev": [1.0, 2.0],
            "adx_period": 14,
            "adx_threshold": 18,
            "bb_period": 60
        },
        "orb": {
            "enabled": False,  # Disabled - high variance
            "base_confidence": 0.70,
            "range_minutes": 5,
            "buffer_atr_factor": 0.15,
            "min_buffer_points": 3.0,
            "volume_confirm": 1.5,
            "max_gap_atr": 0.7,
            "alternative_range": 5
        },
        "reversal_breakout": {
            "enabled": False,  # disabled in favor of master strategy
            "base_confidence": 0.70,  # Increased confidence
            "rsi_oversold": 20,  # More extreme levels
            "rsi_overbought": 80,
            "stop_loss_atr": 0.8,
            "take_profit_r_mult": 3.0  # High reward for reversals
        },
        "momentum_surge": {
            "enabled": False,  # disabled in favor of master strategy
            "base_confidence": 0.75,
            "volume_surge": 2.0,  # 2x average volume
            "atr_surge": 1.5,     # 1.5x average ATR
            "rsi_momentum": 60,   # Strong momentum threshold
            "stop_loss_atr": 1.0,
            "take_profit_r_mult": 2.5
        },

        # --------------------------------------------------------------
        # ICT MASTER STRATEGY
        # Modes supported in ICTMasterStrategy.__init__:
        #   - "conservative": fewer, higher-quality trades
        #   - "normal": balanced
        #   - "aggressive": more trades (good for testing / research)
        # --------------------------------------------------------------

    # ... existing strategies ...
    "ict_master": {
        "enabled": True,
        "risk_per_trade": 0.01,
        "max_portfolio_risk": 0.03,
        "min_rr_ratio": 1.5,
        "target_rr_ratio": 2.0,
        "min_atr": 5.0,
        "base_confidence": 0.6,
        "min_confidence": 0.65,   # ML probability threshold
        "max_confidence": 0.99,
        "model_path": "models/ict_edge_model.pkl",  # after you train + save
    },

        "ml_ict_edge": {
            "enabled": False,
            "model_path": "models/mnq_ict_edge_xgb.pkl",
            "min_prob": 0.65,
            "stop_atr_mult": 1.0,
            "rr_target": 2.0,
            "min_atr": 3.0,
        }
    })

# Global configuration instance
config = TradingConfig()
