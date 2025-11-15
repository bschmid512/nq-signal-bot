from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import joblib
import numpy as np
import pandas as pd
from loguru import logger

from src.ml.ict_feature_pipeline import build_features_for_bar
from src.strategies.ict_complete_strategy import ICTStrategy, ICTConfig


@dataclass
class ICTMLEdgeModel:
    """
    Thin wrapper around the trained ICT edge model.

    It:
      - loads a joblib model payload from model_path
      - stores feature column order used at training time
      - can score a single bar (using build_features_for_bar) and return P(win)
    """

    # This must match how SignalGenerationEngine constructs it
    model_path: str = "models/mnq_ict_edge_xgb.pkl"

    # Internal state
    _model: Any = None
    _feature_cols: Optional[Sequence[str]] = None

    def __post_init__(self) -> None:
        # Use loguru's logger by default
        self.logger = logger
        self._load_model()

    # ------------------------------------------------------------------ #
    # Model loading
    # ------------------------------------------------------------------ #
    def _load_model(self) -> None:
        path = Path(self.model_path)
        if not path.exists():
            self.logger.warning(
                f"[ICTMLEdgeModel] Model file not found at {path}. "
                "ICT ML edge filter will be disabled."
            )
            self._model = None
            self._feature_cols = None
            return

        loaded = joblib.load(path)
        self.logger.info(
            f"[ICTMLEdgeModel] Loaded object of type {type(loaded)} from {path}"
        )

        model = None
        feature_cols: Optional[Sequence[str]] = None

        # Training script saves: {"model": clf, "feature_columns": [...]}
        if isinstance(loaded, dict):
            model = loaded.get("model", None)
            feature_cols = loaded.get("feature_columns") or loaded.get(
                "feature_cols"
            )
        else:
            model = loaded
            if hasattr(model, "feature_names_in_"):
                feature_cols = list(model.feature_names_in_)

        if model is None:
            self.logger.error(
                "[ICTMLEdgeModel] Could not find an estimator in the loaded payload. "
                "ML edge will be disabled."
            )
            self._model = None
            self._feature_cols = None
            return

        if not hasattr(model, "predict_proba"):
            self.logger.error(
                "[ICTMLEdgeModel] Loaded estimator does not implement predict_proba. "
                "ML edge will be disabled."
            )
            self._model = None
            self._feature_cols = None
            return

        self._model = model
        self._feature_cols = feature_cols

        if self._feature_cols is not None:
            self.logger.info(
                f"[ICTMLEdgeModel] Using {len(self._feature_cols)} feature columns."
            )
        else:
            self.logger.warning(
                "[ICTMLEdgeModel] No feature column metadata found; "
                "will fall back to numeric columns at scoring time."
            )

    def is_ready(self) -> bool:
        """Return True if a usable model is loaded."""
        return self._model is not None

    def set_feature_cols(self, cols: Sequence[str]) -> None:
        """Optional helper to override feature columns after loading."""
        self._feature_cols = list(cols) if cols is not None else None

    # ------------------------------------------------------------------ #
    # Internal prediction helper
    # ------------------------------------------------------------------ #
    def _predict_from_features(
        self, feat_row: Dict | pd.Series | pd.DataFrame
    ) -> Optional[float]:
        """
        Take a single feature row and return P(win) from the model.
        """

        if self._model is None:
            self.logger.debug("[ICTMLEdgeModel] No model loaded; returning None.")
            return None

        # Normalize input into a 1-row DataFrame
        if isinstance(feat_row, dict):
            df = pd.DataFrame([feat_row])
        elif isinstance(feat_row, pd.Series):
            df = feat_row.to_frame().T
        elif isinstance(feat_row, pd.DataFrame):
            if len(feat_row) == 0:
                self.logger.warning(
                    "[ICTMLEdgeModel] Empty DataFrame passed to _predict_from_features."
                )
                return None
            df = df.tail(1).copy()
        else:
            self.logger.error(
                "[ICTMLEdgeModel] Unsupported feat_row type: %s", type(feat_row)
            )
            return None

        # Align columns with training-time feature set
        if not self._feature_cols:
            # Fallback: numeric cols only
            X = df.select_dtypes(include=[np.number])
        else:
            # Ensure all expected columns exist; fill missing with 0.0
            for col in self._feature_cols:
                if col not in df.columns:
                    df[col] = 0.0

            try:
                X = df[list(self._feature_cols)]
            except KeyError as e:
                self.logger.error(
                    "[ICTMLEdgeModel] Column alignment failed even after "
                    "filling missing features: %s",
                    e,
                    exc_info=True,
                )
                return None

        # Final safety: only numeric
        X = X.select_dtypes(include=[np.number])

        try:
            proba = float(self._model.predict_proba(X)[0, 1])
            return proba
        except Exception as e:
            self.logger.error(
                "[ICTMLEdgeModel] Error predicting probability: %s",
                e,
                exc_info=True,
            )
            return None

    # ------------------------------------------------------------------ #
    # Public API used by SignalGenerationEngine
    # ------------------------------------------------------------------ #
    def score_bar(
        self,
        df: pd.DataFrame,
        strategy: ICTStrategy,
        idx: int,
    ) -> Optional[float]:
        """
        Build ICT features for df.iloc[idx] and return P(win) from the ML model.

        This matches how SignalGenerationEngine currently calls it.
        """
        if self._model is None:
            return None

        feats: Optional[Dict] = build_features_for_bar(df, strategy, idx)
        if feats is None:
            # e.g. too little history, ATR invalid, neutral bias, etc.
            return None

        return self._predict_from_features(feats)
