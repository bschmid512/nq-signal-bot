import sqlite3
import json
from pathlib import Path

import numpy as np
import pandas as pd

DB_PATH = Path("nq_signals.db")  # adjust if needed

# Set this to around when you kicked off the backtest
RUN_START = "2025-11-15 10:46:00"  # <- change for each run

def load_ict_signals(db_path=DB_PATH, run_start=RUN_START):
    conn = sqlite3.connect(db_path)
    query = """
        SELECT *
        FROM signals
        WHERE strategy = 'ict_master'
          AND created_at >= ?
        ORDER BY timestamp
    """
    df = pd.read_sql_query(query, conn, params=(run_start,))
    conn.close()

    if df.empty:
        print("No ict_master signals found for this run window.")
        return df

    # Parse metadata JSON
    def parse_meta(x):
        if x is None:
            return {}
        if isinstance(x, dict):
            return x
        try:
            return json.loads(x)
        except Exception:
            return {}

    meta = df["metadata"].apply(parse_meta)

    # Extract useful ML-related fields (if present)
    df["ml_edge_p"] = meta.apply(lambda m: m.get("ml_edge_p"))
    df["pre_ml_confidence"] = meta.apply(lambda m: m.get("pre_ml_confidence"))
    df["bias"] = meta.apply(lambda m: m.get("bias"))
    df["killzone"] = meta.apply(lambda m: m.get("killzone"))
    df["regime"] = meta.apply(lambda m: m.get("regime"))

    return df


def basic_summary(df: pd.DataFrame):
    print(f"Total ict_master signals in window: {len(df)}")
    print(df[["ml_edge_p", "pre_ml_confidence"]].describe())

    # How many signals survive ML threshold?
    if "ml_edge_p" in df.columns:
        for thr in [0.5, 0.6, 0.65, 0.7, 0.75]:
            cnt = (df["ml_edge_p"] >= thr).sum()
            print(f"ml_edge_p >= {thr:.2f}: {cnt} signals")

    # Quick view by bias / killzone
    if "bias" in df.columns:
        print("\nCounts by bias:")
        print(df["bias"].value_counts(dropna=False))

    if "killzone" in df.columns:
        print("\nCounts by killzone:")
        print(df["killzone"].value_counts(dropna=False))


if __name__ == "__main__":
    df_signals = load_ict_signals()
    if not df_signals.empty:
        basic_summary(df_signals)
        # Optional: df_signals.to_csv("ict_signals_last_run.csv", index=False)
