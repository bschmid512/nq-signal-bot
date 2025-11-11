from dataclasses import dataclass
from datetime import datetime
from typing import Dict

@dataclass
class Signal:
    """Trading signal data structure"""
    timestamp: datetime
    strategy: str
    signal_type: str  # 'long' or 'short'
    symbol: str
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: float
    risk_reward: float
    atr: float
    volume: int = 0
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}