from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Any, Dict

import numpy as np
import pandas as pd
from loguru import logger
import joblib

from src.ml.ict_feature_pipeline import build_features_for_bar
from src.strategies.ict_complete_strategy import ICTStrategy, ICTConfig


@dataclass
class ICTMLEdgeModel:
    """
    Thin wrapper around the trained ICT edge model.

    It:
      - loads models/mnq_ict_edge_xgb.pkl
      - knows which feature columns it was trained on
      - can score a single bar: P(+2R before -1R | features)
    """
    model_path: str = "models/mnq_ict_edge_xgb.pkl"
    _model: Any = None
    _feature_cols: Optional[Sequence[str]] = None

    def __post_init__(self) -> None:
        self._load_model()

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #
    def _load_model(self) -> None:
        path = Path(self.model_path)
        if not path.exists():
            logger.warning(
                f"[ICTMLEdgeModel] Model file not found at {path}. "
                "ICT ML edge filter will be disabled."
            )
            return

        loaded = joblib.load(path)
        logger.info(f"[ICTMLEdgeModel] Loaded object of type {type(loaded)} from {path}")

        # Case 1: training script saved a dict like {"model": clf, "feature_cols": [...]}
        if isinstance(loaded, dict):
            # Extract the underlying estimator
            if "model" in loaded:
                self._model = loaded["model"]
            elif "estimator" in loaded:
                self._model = loaded["estimator"]
            else:
                # Fallback: if there is only one estimator-like object in the dict
                est_keys = [
                    k for k, v in loaded.items()
                    if hasattr(v, "predict_proba")
                ]
                if est_keys:
                    key = est_keys[0]
                    self._model = loaded[key]
                    logger.warning(
                        f"[ICTMLEdgeModel] Using estimator from dict key '{key}'."
                    )
                else:
                    logger.error(
                        "[ICTMLEdgeModel] Loaded dict does not contain a model / "
                        "estimator entry with predict_proba. "
                        "ML edge will be disabled."
                    )
                    self._model = None
                    return

            # Extract feature columns if present
            for feat_key in ("feature_cols", "feature_columns", "feature_names", "features"):
                if feat_key in loaded and loaded[feat_key] is not None:
                    self._feature_cols = list(loaded[feat_key])
                    logger.info(
                        f"[ICTMLEdgeModel] Using {len(self._feature_cols)} "
                        f"feature columns from saved dict key '{feat_key}'."
                    )
                    break

        else:
            # Case 2: joblib.dump(model) without wrapping dict
            self._model = loaded

        # If the estimator itself knows its input feature names, use them
        if self._model is not None and self._feature_cols is None:
            if hasattr(self._model, "feature_names_in_"):
                self._feature_cols = list(self._model.feature_names_in_)
                logger.info(
                    f"[ICTMLEdgeModel] Inferred {len(self._feature_cols)} feature "
                    "columns from model.feature_names_in_."
                )

        if self._model is None:
            logger.error(
                "[ICTMLEdgeModel] Model could not be initialized; "
                "ML edge will be disabled."
            )
        elif not hasattr(self._model, "predict_proba"):
            logger.error(
                "[ICTMLEdgeModel] Loaded object does not implement predict_proba; "
                "ML edge will be disabled."
            )
            self._model = None
        else:
            logger.info("[ICTMLEdgeModel] Model ready for ML edge scoring.")

    def is_ready(self) -> bool:
        return self._model is not None

    # ------------------------------------------------------------------ #
    # Scoring
    # ------------------------------------------------------------------ #
    def score_bar(
        self,
        df: pd.DataFrame,
        strategy: ICTStrategy,
        bar_index: Any,
    ) -> Optional[float]:
        """
        Build ICT feature vector for a single bar and return
        P(+2R before -1R) from the trained model.

        Parameters
        ----------
        df : pd.DataFrame
            Historical OHLCV (5m MNQ) including the bar to score.
        strategy : ICTStrategy
            The same ICT engine instance you used during training.
        bar_index : int or Timestamp
            The index of the bar inside df you want to score.
            - If df is indexed by timestamp, you can pass df.index[-1]
            - If it's positional, you can pass len(df) - 1

        Returns
        -------
        Optional[float]
            model probability in [0, 1], or None if something is missing
        """
        if not self.is_ready():
            return None

        # 1) Build features for this bar (same as during training)
        try:
            feats_dict: Dict[str, Any] = build_features_for_bar(df, strategy, bar_index)
        except Exception as e:
            logger.error(f"[ICTMLEdgeModel] Error building features for bar {bar_index}: {e}")
            return None

        feat_series = pd.Series(feats_dict)

        # 2) Align with the columns used at training time
        if self._feature_cols is not None:
            missing = [c for c in self._feature_cols if c not in feat_series.index]
            if missing:
                logger.error(
                    "[ICTMLEdgeModel] Missing feature columns in live data: "
                    f"{missing}"
                )
                return None
            X = feat_series[self._feature_cols].values.reshape(1, -1)
        else:
            # Fallback: use whatever order we have (only safe if training
            # also used a raw numpy array without explicit column names).
            X = feat_series.values.reshape(1, -1)

        # 3) Predict probability
        try:
            proba = float(self._model.predict_proba(X)[0, 1])
            return proba
        except Exception as e:
            logger.error(f"[ICTMLEdgeModel] Error predicting probability: {e}")
            return None
