
"""
Stub ML ICT Edge strategy.

Originally we planned to use this as a standalone strategy, but the
current design pushes the ML edge gating into SignalGenerationEngine
itself (using ICTStrategy + ICTMLEdgeModel).

We keep this stub so that any legacy imports like
`from src.strategies.ml_ict_edge import MLICTEdgeStrategy`
won't crash the app, but this class does NOT generate signals.
"""

from dataclasses import dataclass
from typing import List, Dict, Any
import pandas as pd
from loguru import logger

from src.core.models import Signal


@dataclass
class MLICTEdgeStrategy:
    """No-op strategy: all ML gating is in SignalGenerationEngine now."""

    config: Dict[str, Any] = None

    def __post_init__(self):
        if self.config is None:
            self.config = {}
        logger.warning(
            "[MLICTEdgeStrategy] Stub in use. "
            "ML edge is applied inside SignalGenerationEngine; "
            "this strategy returns no signals."
        )

    def generate_signals(self, df: pd.DataFrame, symbol: str) -> List[Signal]:
        """
        Legacy API: called like other strategies but always returns [].
        """
        return []
