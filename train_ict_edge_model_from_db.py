"""
Train ICT "edge model" from MNQ 5-minute data in nq_signals.db.

Pipeline:
1. Load OHLCV from market_data (symbol=MNQ1!, timeframe='5').
2. Run ICTStrategy to generate raw ICT signals with full context.
3. Build labeled dataset (+2R before -1R within N bars).
4. Train classifier to estimate P(win | features).
5. Save to ict_edge_model.pkl.

Later:
- Load this model inside your SignalGenerationEngine.
- For each ICT signal, build features, get p = model.predict_proba, and
  only trade when p >= threshold (e.g. 0.65).
"""

import sqlite3
import pandas as pd
from loguru import logger

from src.ict.ict_complete_strategy import (
    ICTConfig,
    ICTStrategy,
    ICTMLTrainer,
)


DB_PATH = "nq_signals.db"      # adjust if needed
SYMBOL = "MNQ1!"
TIMEFRAME = "5"
MODEL_PATH = "ict_edge_model.pkl"


def load_mnq_from_db(
    db_path: str,
    symbol: str,
    timeframe: str
) -> pd.DataFrame:
    """Load MNQ OHLCV from SQLite."""
    logger.info(f"Loading {symbol} {timeframe}m data from {db_path}...")
    conn = sqlite3.connect(db_path)

    query = """
        SELECT timestamp, open, high, low, close, volume
        FROM market_data
        WHERE symbol = ? AND timeframe = ?
        ORDER BY timestamp ASC
    """
    df = pd.read_sql_query(
        query,
        conn,
        params=(symbol, timeframe),
        parse_dates=["timestamp"]
    )
    conn.close()

    if df.empty:
        raise ValueError("No rows returned from market_data. Check symbol/timeframe.")

    df.set_index("timestamp", inplace=True)
    logger.info(f"Loaded {len(df)} bars from {df.index.min()} to {df.index.max()}")
    return df


def main():
    config = ICTConfig()
    strategy = ICTStrategy(config)
    trainer = ICTMLTrainer(config)

    # 1) Load data
    df = load_mnq_from_db(DB_PATH, SYMBOL, TIMEFRAME)

    # 2) Prepare + generate ICT signals
    logger.info("Preparing data & generating ICT signals...")
    df_prep = strategy.prepare_data(df)
    signals = strategy.generate_signals(df_prep)
    logger.info(f"Generated {len(signals)} ICT signals.")

    if not signals:
        logger.error("No ICT signals generated. Cannot train model.")
        return

    # 3) Build labeled dataset
    logger.info("Building labeled dataset (+2R vs -1R)...")
    dataset = trainer.build_labeled_dataframe(df_prep, signals)
    if dataset.empty:
        logger.error("Labeled dataset is empty (no +2R/-1R outcomes within horizon).")
        return

    logger.info(f"Labeled rows: {len(dataset)}")

    # 4) Train model
    logger.info("Training edge model...")
    metrics = trainer.train_model(dataset)
    logger.info(f"Training complete. AUC={metrics['auc']:.3f}, F1={metrics['f1']:.3f}")

    # 5) Save model
    trainer.save_model(MODEL_PATH)
    logger.info(f"Saved edge model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
