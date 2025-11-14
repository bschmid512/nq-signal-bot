from __future__ import annotations

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

try:
    from xgboost import XGBClassifier
    HAVE_XGB = True
except ImportError:
    HAVE_XGB = False
    from sklearn.ensemble import GradientBoostingClassifier as FallbackGBM


FEATURE_EXCLUDE = {"timestamp", "label"}


def train_ict_edge_model(
    df_features: pd.DataFrame,
    model_dir: str = "models",
    model_name: str = "mnq_ict_edge_xgb.pkl",
) -> str:
    """
    Train classifier on ICT features and save to disk.
    Returns: path to saved model.
    """
    if df_features.empty:
        raise ValueError("Empty feature DataFrame passed to train_ict_edge_model")

    X = df_features.drop(columns=[c for c in FEATURE_EXCLUDE if c in df_features.columns])
    y = df_features["label"].astype(int)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, shuffle=False  # time-aware split
    )

    if HAVE_XGB:
        logger.info("Training XGBClassifier for ICT edge model...")
        model = XGBClassifier(
            max_depth=4,
            n_estimators=250,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            tree_method="hist",
        )
    else:
        logger.warning("xgboost not installed, using GradientBoostingClassifier fallback.")
        model = FallbackGBM(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
        )

    model.fit(X_train, y_train)

    # ---- Evaluation ----
    y_pred_val = model.predict(X_val)
    if hasattr(model, "predict_proba"):
        proba_val = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, proba_val)
    else:
        proba_val = None
        auc = None

    logger.info("Validation report:\n" + classification_report(y_val, y_pred_val))
    if auc is not None:
        logger.info(f"Validation ROC-AUC: {auc:.3f}")

    Path(model_dir).mkdir(parents=True, exist_ok=True)
    model_path = os.path.join(model_dir, model_name)
    payload = {
        "model": model,
        "feature_columns": list(X.columns),
    }
    joblib.dump(payload, model_path)
    logger.info(f"Saved ICT edge model to {model_path}")
    return model_path


if __name__ == "__main__":
    print("Import this function from a training script, don't run directly.")
