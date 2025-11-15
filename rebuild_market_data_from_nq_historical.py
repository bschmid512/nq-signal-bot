#!/usr/bin/env python

"""
Rebuild market_data from nq_historical.csv (1-minute NQ data).

- Optionally deletes the existing nq_signals.db file ("clean slate")
- Ensures schema is created via DatabaseManager
- Loads nq_historical.csv
- (Optionally) stores both 1-min and 5-min candles in market_data
"""

from pathlib import Path
import sqlite3
import pandas as pd
from config.config import config  

from src.utils.database import DatabaseManager

# === CONFIG ===

# Where nq_historical.csv lives (adjust if needed)
CSV_PATH = Path("data/nq_historical.csv")

# Which timeframes to load into market_data
LOAD_1MIN = True
LOAD_5MIN = True

# If True, delete the DB file first (hard reset of all tables)
RESET_DB_FILE = True  # <- flip to True if you want a brand new nq_signals.db

# Optional date range filter (strings "YYYY-MM-DD" or None for full history)
START_DATE = None      # e.g. "2018-01-01"
END_DATE = None        # e.g. "2025-01-01"


def load_and_prepare():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found at {CSV_PATH}")

    print(f"Reading {CSV_PATH} ...")
    usecols = ["Date", "Time", "Open", "High", "Low", "Close", "Volume"]
    df = pd.read_csv(CSV_PATH, usecols=usecols)

    # Combine Date + Time into a single datetime
    df["dt"] = pd.to_datetime(df["Date"] + " " + df["Time"])

    # Optional filter by date range
    if START_DATE is not None:
        df = df[df["dt"] >= pd.to_datetime(START_DATE)]
    if END_DATE is not None:
        df = df[df["dt"] <= pd.to_datetime(END_DATE)]

    df = df.sort_values("dt").reset_index(drop=True)

    print(f"Loaded {len(df)} 1-minute bars "
          f"from {df['dt'].min()} to {df['dt'].max()}")

    return df


def make_1min_df(df: pd.DataFrame) -> pd.DataFrame:
    """Build DB-ready 1-minute dataframe (timeframe='1')."""
    d1 = pd.DataFrame({
        "timestamp": df["dt"].dt.strftime("%Y-%m-%d %H:%M:%S"),
        "open": df["Open"].astype(float),
        "high": df["High"].astype(float),
        "low": df["Low"].astype(float),
        "close": df["Close"].astype(float),
        "volume": df["Volume"].fillna(0).astype(int),
        "timeframe": "1",
        "symbol": config.SYMBOL,   # e.g. "MNQ1!"
    })
    return d1


def make_5min_df(df: pd.DataFrame) -> pd.DataFrame:
    """Resample 1-min data to 5-min and build DB-ready dataframe (timeframe='5')."""
    df = df.set_index("dt")

    agg = {
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    }

    df_5 = df.resample("5min").agg(agg).dropna(subset=["Open", "High", "Low", "Close"])

    d5 = pd.DataFrame({
        "timestamp": df_5.index.strftime("%Y-%m-%d %H:%M:%S"),
        "open": df_5["Open"].astype(float),
        "high": df_5["High"].astype(float),
        "low": df_5["Low"].astype(float),
        "close": df_5["Close"].astype(float),
        "volume": df_5["Volume"].fillna(0).astype(int),
        "timeframe": config.PRIMARY_TIMEFRAME,  # "5"
        "symbol": config.SYMBOL,
    })

    print(f"Resampled to {len(d5)} 5-minute bars.")
    return d5


def insert_in_chunks(df: pd.DataFrame, conn: sqlite3.Connection, table: str):
    """Write a large dataframe to SQLite in chunks so we don't blow memory."""
    if df.empty:
        return

    chunksize = 50_000
    total = len(df)
    print(f"Inserting {total} rows into {table} ...")

    for start in range(0, total, chunksize):
        end = start + chunksize
        chunk = df.iloc[start:end]
        chunk.to_sql(table, conn, if_exists="append", index=False)


def main():
    db_path = Path(config.DATABASE_PATH)
    print(f"Using DB: {db_path}")

    # Optional: hard reset of DB file
    if RESET_DB_FILE and db_path.exists():
        print(f"RESET_DB_FILE=True -> deleting existing DB: {db_path}")
        db_path.unlink()

    # Ensure tables exist
    _ = DatabaseManager(str(db_path))

    # Load raw 1-minute data
    df = load_and_prepare()

    # Build timeframes
    frames = []

    if LOAD_1MIN:
        d1 = make_1min_df(df)
        frames.append(d1)

    if LOAD_5MIN:
        d5 = make_5min_df(df)
        frames.append(d5)

    if not frames:
        print("Nothing to insert (both LOAD_1MIN and LOAD_5MIN are False).")
        return

    # Combine all timeframes and insert
    df_all = pd.concat(frames, ignore_index=True)

    conn = sqlite3.connect(db_path)
    with conn:
        print("Clearing existing market_data and market_data_clean ...")
        conn.execute("DELETE FROM market_data")
        conn.execute("DROP TABLE IF EXISTS market_data_clean")

        insert_in_chunks(df_all, conn, "market_data")

    conn.close()
    print("Done. New market_data loaded.")
    print(f"Total rows inserted: {len(df_all)}")


if __name__ == "__main__":
    main()
