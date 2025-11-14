from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from loguru import logger

from src.core.models import Signal
from src.strategies.ict_complete_strategy import ICTStrategy, ICTConfig

from src.ml.ict_feature_pipeline import build_features_for_bar


@dataclass
class MLICTEdgeConfig:
    model_path: str = "models/mnq_ict_edge_xgb.pkl"
    min_prob: float = 0.65          # gate: only take trades if P(win) > this
    stop_atr_mult: float = 1.0
    rr_target: float = 2.0
    min_atr: float = 3.0            # don't trade when ATR is tiny
    max_spread: float = 2.0         # reserved if you add spread later


class MLICTEdgeStrategy:
    """
    Live strategy that:
      1) Uses ICT analyzers to build a feature vector on each bar
      2) Feeds it into the trained ML model
      3) Emits a Signal only if P(win) > min_prob
    """

    def __init__(self, config: Optional[Dict] = None):
        cfg = config or {}
        self.config = MLICTEdgeConfig(**cfg)

        self.ict_config = ICTConfig()
        self.ict_strategy = ICTStrategy(self.ict_config)

        self.model = None
        self.feature_columns: List[str] = []
        self._load_model()

    # ------------------------------------------------------------------ #
    # Model loading
    # ------------------------------------------------------------------ #

    def _load_model(self):
        try:
            payload = joblib.load(self.config.model_path)
        except FileNotFoundError:
            logger.error(
                f"[MLICT] Model not found at {self.config.model_path}. "
                f"Train it with src/ml/train_mnq_ict_edge_from_db.py first."
            )
            return

        self.model = payload["model"]
        self.feature_columns = payload["feature_columns"]
        logger.info(
            f"[MLICT] Loaded ICT edge model from {self.config.model_path} "
            f"with {len(self.feature_columns)} features."
        )

    # ------------------------------------------------------------------ #
    # Strategy API (called by SignalGenerationEngine)
    # ------------------------------------------------------------------ #

    def generate_signals(self, df: pd.DataFrame, symbol: str) -> List[Signal]:
        if self.model is None:
            # Keep engine alive even if model isn't ready
            return []

        if df is None or df.empty:
            return []

        # Ensure datetime index
        if "timestamp" in df.columns:
            df = df.copy()
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df = df.set_index("timestamp")

        # Let ICTStrategy add its indicators (ATR / EMAs / RSI)
        df = self.ict_strategy.prepare_data(df)

        idx = len(df) - 1
        bar = df.iloc[idx]
        ts = df.index[idx]

        atr = float(bar.get("atr", np.nan))
        if not np.isfinite(atr) or atr < self.config.min_atr:
            return []

        # Build the same features we used in training
        feats = build_features_for_bar(df, self.ict_strategy, idx)
        if feats is None:
            return []

        side = "long" if feats["side"] == 1 else "short"

        X = pd.DataFrame([feats])

        # Align feature columns with what the model was trained on
        for col in list(X.columns):
            if col not in self.feature_columns:
                X = X.drop(columns=[col])

        for col in self.feature_columns:
            if col not in X.columns:
                X[col] = 0.0

        X = X[self.feature_columns]

        proba = float(self.model.predict_proba(X)[0, 1])
        if proba < self.config.min_prob:
            logger.info(
                f"[MLICT] p_win={proba:.2f} below threshold "
                f"{self.config.min_prob:.2f} – no trade."
            )
            return []

        # Build trade params consistent with label definition
        entry = float(bar["close"])
        risk = self.config.stop_atr_mult * atr

        if side == "long":
            stop_loss = entry - risk
            take_profit = entry + risk * self.config.rr_target
        else:
            stop_loss = entry + risk
            take_profit = entry - risk * self.config.rr_target

        rr = self.config.rr_target

        signal = Signal(
            timestamp=ts.to_pydatetime(),
            strategy="ml_ict_edge",
            signal_type=side,
            symbol=symbol,
            entry_price=entry,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=proba,   # now confidence == P(win)
            risk_reward=rr,
            atr=atr,
            volume=0,
            metadata={
                "p_win": float(proba),
                "bias": int(feats["bias"]),
                "higher_tf_bias": int(feats["higher_tf_bias"]),
                "session_score": float(feats["session_score"]),
                "in_killzone": int(feats["in_killzone"]),
            },
        )

        logger.info(
            f"[MLICT] {side.upper()} {symbol} @ {entry:.2f} "
            f"SL={stop_loss:.2f} TP={take_profit:.2f} "
            f"p_win={proba:.2f} RR={rr:.1f}"
        )

        return [signal]
