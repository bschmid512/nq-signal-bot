from __future__ import annotations

from loguru import logger

from config.config import config
from src.utils.database import DatabaseManager
from src.strategies.ict_complete_strategy import ICTStrategy, ICTConfig
from src.ml.ict_feature_pipeline import LabelParams, build_ict_feature_dataset
from src.ml.train_ict_edge_model import train_ict_edge_model


def main():
    logger.info("Loading MNQ data from DB for ICT ML training...")

    db = DatabaseManager(config.DATABASE_PATH)

    df = db.get_latest_data(
        symbol=config.SYMBOL,
        timeframe=config.PRIMARY_TIMEFRAME,
        limit=50000,   # adjust if you want more
    )

    if df is None or df.empty:
        logger.error("No market data in DB – load CSVs or run live collector first.")
        return

    logger.info(f"Loaded {len(df)} rows for {config.SYMBOL} TF={config.PRIMARY_TIMEFRAME}.")

    ict_cfg = ICTConfig()
    ict_strategy = ICTStrategy(ict_cfg)

    label_params = LabelParams(
        horizon_bars=24,
        stop_atr_mult=1.0,
        rr_target=2.0,
        min_atr=ict_cfg.min_atr,
    )

    df_features = build_ict_feature_dataset(df, ict_strategy, label_params)
    if df_features.empty:
        logger.error("Feature dataset is empty – check label params / data length.")
        return

    logger.info(f"Feature dataset shape: {df_features.shape}")

    model_path = train_ict_edge_model(df_features)
    logger.info(f"Finished training ICT edge model: {model_path}")


if __name__ == "__main__":
    main()
