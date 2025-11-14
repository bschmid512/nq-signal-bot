import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, List, Dict
import json
from loguru import logger
import math  # ✅ add this

def _normalize_timestamp(ts):
    """
    Ensure timestamp is safe for sqlite:
    - pandas.Timestamp -> naive ISO string
    - datetime -> naive ISO string
    - string -> passed through
    - anything else -> str()
    """
    if isinstance(ts, pd.Timestamp):
        ts = ts.to_pydatetime()
    if isinstance(ts, datetime):
        # drop tz if present and use a consistent format
        if ts.tzinfo is not None:
            ts = ts.astimezone(tz=None).replace(tzinfo=None)
        return ts.isoformat(sep=" ")
    if isinstance(ts, str):
        return ts
    return str(ts)



def _is_missing(v) -> bool:
    """Treat None and NaN as missing."""
    if v is None:
        return True
    if isinstance(v, float) and math.isnan(v):
        return True
    return False




class DatabaseManager:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        with sqlite3.connect(self.db_path) as conn:
            # Market data table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS market_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume INTEGER NOT NULL,
                    timeframe TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Signals table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    strategy TEXT NOT NULL,
                    signal_type TEXT NOT NULL,  -- 'long' or 'short'
                    symbol TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    stop_loss REAL NOT NULL,
                    take_profit REAL NOT NULL,
                    confidence REAL NOT NULL,
                    risk_reward REAL NOT NULL,
                    atr REAL NOT NULL,
                    volume INTEGER,
                    metadata TEXT,  -- JSON string for additional data
                    executed BOOLEAN DEFAULT FALSE,
                    pnl REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Strategy performance table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS strategy_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date DATE NOT NULL,
                    strategy TEXT NOT NULL,
                    signals_generated INTEGER DEFAULT 0,
                    signals_executed INTEGER DEFAULT 0,
                    total_pnl REAL DEFAULT 0,
                    win_rate REAL DEFAULT 0,
                    avg_r_multiple REAL DEFAULT 0,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Risk management table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS risk_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date DATE NOT NULL,
                    daily_pnl REAL DEFAULT 0,
                    max_drawdown REAL DEFAULT 0,
                    consecutive_losses INTEGER DEFAULT 0,
                    total_trades INTEGER DEFAULT 0,
                    risk_per_trade REAL DEFAULT 0,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
    
    def insert_market_data(self, data: Dict) -> bool:
        """Insert market data into database (robust to None/NaN fields)."""
        try:
            ts = _normalize_timestamp(data.get("timestamp"))

            # Core prices
            close = data.get("close")
            open_ = data.get("open", close)
            high = data.get("high")
            low = data.get("low")

            # Fix NaN/None using _is_missing
            if _is_missing(close):
                # if we don't even have a close, this row is trash
                logger.error(f"[DB] Skipping market_data insert (missing close): {data}")
                return False

            if _is_missing(open_):
                open_ = close

            if _is_missing(high):
                high = max(open_, close)

            if _is_missing(low):
                low = min(open_, close)

            # Final safety: if anything still missing, skip insert
            if any(_is_missing(v) for v in (open_, high, low, close)):
                logger.error(f"[DB] Skipping market_data insert (still missing price): {data}")
                return False

            volume = int(data.get("volume", 0) or 0)
            timeframe = data.get("timeframe", "")
            symbol = data.get("symbol", "")

            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO market_data (
                        timestamp, open, high, low, close, volume, timeframe, symbol
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        ts,
                        float(open_),
                        float(high),
                        float(low),
                        float(close),
                        volume,
                        timeframe,
                        symbol,
                    ),
                )
            return True

        except Exception as e:
            logger.error(f"Error inserting market data: {e} | data={data}")
            return False


    def _sanitize_metadata(self, obj):
        """
        Make metadata JSON-serializable:
        - convert numpy / pandas scalars to Python ints/floats
        - recurse through dicts/lists/tuples
        - fallback to str() for unknown types
        """
        if obj is None:
            return None

        # numpy / pandas scalar
        if isinstance(obj, (np.generic,)):
            return obj.item()

        # plain scalar
        if isinstance(obj, (int, float, str, bool)):
            return obj

        # dict
        if isinstance(obj, dict):
            return {str(k): self._sanitize_metadata(v) for k, v in obj.items()}

        # list / tuple
        if isinstance(obj, (list, tuple)):
            return [self._sanitize_metadata(v) for v in obj]

        # datetime-like
        if isinstance(obj, datetime):
            return obj.isoformat()

        # anything else – stringify
        return str(obj)


        
    def insert_signal(self, signal: Dict) -> bool:
        """Insert trading signal into database."""
        try:
            ts = _normalize_timestamp(signal["timestamp"])

            metadata = signal.get("metadata", {})
            metadata_clean = self._sanitize_metadata(metadata)

            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO signals (
                        timestamp, strategy, signal_type, symbol, entry_price,
                        stop_loss, take_profit, confidence, risk_reward, atr,
                        volume, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        ts,
                        signal["strategy"],
                        signal["signal_type"],
                        signal["symbol"],
                        float(signal["entry_price"]),
                        float(signal["stop_loss"]),
                        float(signal["take_profit"]),
                        float(signal["confidence"]),
                        float(signal["risk_reward"]),
                        float(signal["atr"]),
                        int(signal.get("volume", 0) or 0),
                        json.dumps(metadata_clean),
                    ),
                )
            return True
        except Exception as e:
            logger.error(f"Error inserting signal: {e} | signal={signal}")
            return False



    
    def get_latest_data(self, symbol: str, timeframe: str, limit: int = 100) -> pd.DataFrame:
        """Get latest market data for analysis"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = '''
                    SELECT * FROM market_data 
                    WHERE symbol = ? AND timeframe = ? 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                '''
                df = pd.read_sql_query(query, conn, params=(symbol, timeframe, limit))
                # Around line 150 in get_latest_data method:
                if not df.empty:
                    df = df.sort_values('timestamp')
                    # Handle mixed timestamp formats
                    try:
                        df['timestamp'] = pd.to_datetime(df['timestamp'], format='ISO8601', utc=True)
                    except Exception:
                        # Fallback for mixed formats
                        df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', utc=True)
                return df
        except Exception as e:
            logger.error(f"Error fetching latest data: {e}")
            return pd.DataFrame()
    
    def get_recent_signals(self, strategy: Optional[str] = None, limit: int = 50) -> pd.DataFrame:
        """Get recent trading signals"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                if strategy:
                    query = '''
                        SELECT * FROM signals 
                        WHERE strategy = ? 
                        ORDER BY timestamp DESC 
                        LIMIT ?
                    '''
                    df = pd.read_sql_query(query, conn, params=(strategy, limit))
                else:
                    query = '''
                        SELECT * FROM signals 
                        ORDER BY timestamp DESC 
                        LIMIT ?
                    '''
                    df = pd.read_sql_query(query, conn, params=(limit,))
                
                if not df.empty:
                    df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', utc=True)
                    df['metadata'] = df['metadata'].apply(lambda x: json.loads(x) if x else {})
                return df
        except Exception as e:
            logger.error(f"Error fetching recent signals: {e}")
            return pd.DataFrame()
    
    def update_signal_execution(self, signal_id: int, executed: bool = True, pnl: Optional[float] = None):
        """Update signal execution status"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                if pnl is not None:
                    conn.execute('''
                        UPDATE signals 
                        SET executed = ?, pnl = ? 
                        WHERE id = ?
                    ''', (executed, pnl, signal_id))
                else:
                    conn.execute('''
                        UPDATE signals 
                        SET executed = ? 
                        WHERE id = ?
                    ''', (executed, signal_id))
        except Exception as e:
            logger.error(f"Error updating signal execution: {e}")
    
    def get_strategy_performance(self, days: int = 30) -> pd.DataFrame:
        """Get strategy performance metrics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = '''
                    SELECT 
                        date,
                        strategy,
                        signals_generated,
                        signals_executed,
                        total_pnl,
                        win_rate,
                        avg_r_multiple
                    FROM strategy_performance 
                    WHERE date >= date('now', '-{} days')
                    ORDER BY date DESC
                '''.format(days)
                return pd.read_sql_query(query, conn)
        except Exception as e:
            logger.error(f"Error fetching strategy performance: {e}")
            return pd.DataFrame()
    
    def cleanup_old_data(self, days_to_keep: int = 30):
        """Clean up old market data to manage database size"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    DELETE FROM market_data 
                    WHERE timestamp < datetime('now', '-{} days')
                '''.format(days_to_keep))
                
                conn.execute('''
                    DELETE FROM signals 
                    WHERE timestamp < datetime('now', '-{} days')
                '''.format(days_to_keep))
                
                logger.info(f"Cleaned up data older than {days_to_keep} days")
        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")