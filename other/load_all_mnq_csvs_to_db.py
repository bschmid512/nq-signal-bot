# load_all_mnq_csvs_to_db.py

import pandas as pd
from pathlib import Path
from datetime import datetime
from loguru import logger

from config.config import config
from src.utils.database import DatabaseManager


DATA_DIR = Path("data/mnq_csv")  # folder containing all your MNQ CSVs


def normalize_timestamp(ts_str: str) -> str:
    """
    Parse TradingView-style ISO time with timezone and store as naive local string.
    Example input: '2025-10-08T09:05:00-04:00'
    """
    ts = pd.to_datetime(ts_str)  # parses the -04:00 / -05:00 offset
    # Convert to New York time and drop tz
    if ts.tzinfo is not None:
        ts = ts.tz_convert("America/New_York").tz_localize(None)
    return ts.isoformat(sep=" ")


def main():
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"{DATA_DIR} does not exist. Put your CSVs there.")

    csv_files = sorted(DATA_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {DATA_DIR}")

    logger.info(f"Found {len(csv_files)} CSV files:")
    for f in csv_files:
        logger.info(f"  - {f.name}")

    frames = []
    for f in csv_files:
        logger.info(f"Reading {f.name}")
        df = pd.read_csv(f)

        required_cols = ["time", "open", "high", "low", "close"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"{f.name} is missing required columns: {missing}")

        # Keep only what we need + remember source file (optional)
        sub = df[["time", "open", "high", "low", "close"]].copy()
        sub["source_file"] = f.name
        frames.append(sub)

    combined = pd.concat(frames, ignore_index=True)
    logger.info(f"Combined rows before cleaning: {len(combined)}")

    # Normalize timestamps and clean
    combined["timestamp"] = combined["time"].apply(normalize_timestamp)

    # Sort by time
    combined = combined.sort_values("timestamp").reset_index(drop=True)

    # Drop duplicate bars on timestamp (keep last)
    before = len(combined)
    combined = combined.drop_duplicates(subset=["timestamp"], keep="last")
    after = len(combined)
    logger.info(f"Dropped {before - after} duplicate timestamps; {after} rows remain.")

    # Insert into DB
    db = DatabaseManager(config.DATABASE_PATH)
    symbol = config.SYMBOL           # e.g. "MNQ1!"
    timeframe = config.PRIMARY_TIMEFRAME  # e.g. "5"

    inserted = 0
    for _, row in combined.iterrows():
        try:
            data = {
                "timestamp": row["timestamp"],     # already a string
                "symbol": symbol,
                "timeframe": str(timeframe),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": 0,  # CSVs have no volume; use 0
            }
            if db.insert_market_data(data):
                inserted += 1
        except Exception as e:
            logger.error(f"Error inserting row: {e} | row={row.to_dict()}")

    logger.info(f"Finished loading CSVs into DB: inserted {inserted} bars.")


if __name__ == "__main__":
    main()
