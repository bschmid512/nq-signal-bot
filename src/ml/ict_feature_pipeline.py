from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from src.strategies.ict_complete_strategy import ICTStrategy, ICTConfig



@dataclass
class LabelParams:
    """Label rule: did +rr_target*R happen before -1R in next horizon_bars?"""
    horizon_bars: int = 24          # lookahead window (bars)
    stop_atr_mult: float = 1.0      # SL distance in ATRs
    rr_target: float = 2.0          # +2R vs -1R
    min_atr: float = 1.0            # skip dead markets


def _encode_bias(bias: str) -> int:
    mapping = {
        "strong_bullish": 2,
        "bullish": 1,
        "neutral": 0,
        "bearish": -1,
        "strong_bearish": -2,
    }
    return mapping.get(bias, 0)


def _session_to_one_hot(session: str) -> Dict[str, int]:
    return {
        "session_asian": int(session == "asian"),
        "session_london": int(session == "london"),
        "session_new_york": int(session == "new_york"),
        "session_transition": int(session == "transition"),
    }


def label_bar(
    df: pd.DataFrame,
    idx: int,
    side: str,
    params: LabelParams,
) -> Optional[int]:
    """
    1 = +rr_target*R happened before -1R in next N bars
    0 = SL hit first OR neither hit within window
    None = ambiguous (TP & SL in same bar) or not enough lookahead
    """
    if idx >= len(df) - 1:
        return None

    atr = df["atr"].iloc[idx]
    if pd.isna(atr) or atr <= 0 or atr < params.min_atr:
        return None

    entry = df["close"].iloc[idx]
    risk = params.stop_atr_mult * atr

    if side == "long":
        sl = entry - risk
        tp = entry + params.rr_target * risk
    else:
        sl = entry + risk
        tp = entry - params.rr_target * risk

    end_idx = min(len(df), idx + 1 + params.horizon_bars)
    future = df.iloc[idx + 1 : end_idx]
    if future.empty:
        return None

    for _, bar in future.iterrows():
        high = bar["high"]
        low = bar["low"]

        if side == "long":
            hit_tp = high >= tp
            hit_sl = low <= sl
        else:
            hit_tp = low <= tp
            hit_sl = high >= sl

        # Ambiguous inside one bar → drop this sample
        if hit_tp and hit_sl:
            return None
        if hit_tp:
            return 1
        if hit_sl:
            return 0

    # Neither TP nor SL within window → treat as "no +2R edge"
    return 0


def build_features_for_bar(
    df: pd.DataFrame,
    strategy: ICTStrategy,
    idx: int,
) -> Optional[Dict]:
    """
    Build ICT feature vector for a single bar.

    Returns dict or None if bar should be skipped (e.g. neutral bias, no ATR).
    """
    if idx < 100:  # need history for structure/OB/liquidity
        return None

    bar = df.iloc[idx]
    ts = df.index[idx]

    atr = bar.get("atr", np.nan)
    if pd.isna(atr) or atr <= 0:
        return None

    close = bar["close"]

    # Windows for analyzers (small, so it's not crazy slow)
    window_struct = df.iloc[max(0, idx - 200) : idx + 1]
    window_ob = df.iloc[max(0, idx - strategy.config.ob_lookback - 5) : idx + 1]
    window_liq = df.iloc[max(0, idx - 30) : idx + 1]
    window_fvg = df.iloc[max(0, idx - 20) : idx + 1]
    window_mom = df.iloc[max(0, idx - 30) : idx + 1]

    # ---- Market structure ----
    bias = strategy.market_analyzer.determine_market_bias(window_struct)
    bias_num = _encode_bias(bias)

    # Candidate trade direction = direction of bias
    if bias in ("strong_bullish", "bullish"):
        side = "long"
    elif bias in ("strong_bearish", "bearish"):
        side = "short"
    else:
        # skip neutral bars – we're not taking trades there anyway
        return None

    key_levels = strategy.market_analyzer.find_support_resistance_levels(
        window_struct, lookback=50
    )
    supports = key_levels.get("support", []) or []
    resistances = key_levels.get("resistance", []) or []

    dist_support_atr = 0.0
    if supports:
        nearest_sup = max(supports)
        dist_support_atr = (close - nearest_sup) / max(atr, 1e-6)

    dist_res_atr = 0.0
    if resistances:
        nearest_res = min(resistances)
        dist_res_atr = (nearest_res - close) / max(atr, 1e-6)

    # ---- Order blocks ----
    order_blocks = strategy.ob_analyzer.find_order_blocks(window_ob, bias)
    ob_confluence = strategy.ob_analyzer.calculate_ob_confluence(
        close, order_blocks, atr
    )

    nearest_ob_type = 0  # 1 = bullish, -1 = bearish, 0 = none
    nearest_ob_dist_atr = 0.0
    best_dist = None

    for ob in order_blocks:
        center = (ob["zone_high"] + ob["zone_low"]) / 2.0
        dist = (close - center) / max(atr, 1e-6)
        if best_dist is None or abs(dist) < abs(best_dist):
            best_dist = dist
            nearest_ob_dist_atr = dist
            nearest_ob_type = 1 if ob["type"] == "bullish" else -1

    # ---- Liquidity ----
    liq_levels = strategy.liquidity_analyzer.find_liquidity_pools(
        window_liq, lookback=30
    )
    num_liq_above = len(liq_levels.get("above", []))
    num_liq_below = len(liq_levels.get("below", []))
    liq_sweep = int(strategy.liquidity_analyzer.detect_liquidity_sweep(window_liq))

    # FVGs
    fvg_list = strategy.liquidity_analyzer.detect_fair_value_gaps(window_fvg)
    has_bullish_fvg = any(f["type"] == "bullish" for f in fvg_list)
    has_bearish_fvg = any(f["type"] == "bearish" for f in fvg_list)

    # ---- Session / killzone ----
    session = strategy.session_manager.get_current_session(ts.time())
    in_killzone = strategy.session_manager.is_in_killzone(ts.time())
    session_score = strategy.session_manager.get_session_score(session)
    killzone_score = strategy.session_manager.get_killzone_score(in_killzone)
    session_one_hot = _session_to_one_hot(session)

    # ---- Technical ----
    rsi_val = float(bar.get("rsi", 50.0))
    if pd.isna(rsi_val):
        rsi_val = 50.0

    momentum_score = strategy.technical_analyzer.calculate_momentum_score(window_mom)
    volume_conf = strategy.technical_analyzer.get_volume_confluence(window_struct, ts)

    ema_fast = bar.get("ema_fast", np.nan)
    ema_slow = bar.get("ema_slow", np.nan)
    ema_trend = bar.get("ema_trend", np.nan)

    dist_ema_fast = (close - ema_fast) / max(atr, 1e-6) if not pd.isna(ema_fast) else 0.0
    dist_ema_slow = (close - ema_slow) / max(atr, 1e-6) if not pd.isna(ema_slow) else 0.0
    dist_ema_trend = (close - ema_trend) / max(atr, 1e-6) if not pd.isna(ema_trend) else 0.0

    # Higher-TF bias
    higher_tf_bias = strategy._get_higher_timeframe_bias(window_struct)
    higher_tf_bias_num = _encode_bias(higher_tf_bias)

    features = {
        "timestamp": ts,
        "side": 1 if side == "long" else 0,  # 1=long, 0=short

        # Market structure
        "bias": bias_num,
        "higher_tf_bias": higher_tf_bias_num,
        "dist_support_atr": dist_support_atr,
        "dist_res_atr": dist_res_atr,

        # Order blocks
        "ob_confluence": float(ob_confluence),
        "nearest_ob_type": nearest_ob_type,
        "nearest_ob_dist_atr": nearest_ob_dist_atr,

        # Liquidity
        "num_liq_above": num_liq_above,
        "num_liq_below": num_liq_below,
        "liquidity_sweep": liq_sweep,
        "has_bullish_fvg": int(has_bullish_fvg),
        "has_bearish_fvg": int(has_bearish_fvg),

        # Session
        "session_score": session_score,
        "killzone_score": killzone_score,
        "in_killzone": int(in_killzone),
        **session_one_hot,

        # Technical
        "rsi": rsi_val,
        "momentum_score": float(momentum_score),
        "atr": float(atr),
        "dist_ema_fast": float(dist_ema_fast),
        "dist_ema_slow": float(dist_ema_slow),
        "dist_ema_trend": float(dist_ema_trend),
        "volume_confluence": float(volume_conf),
    }

    return features


def build_ict_feature_dataset(
    df: pd.DataFrame,
    strategy: ICTStrategy,
    label_params: LabelParams,
) -> pd.DataFrame:
    """
    Take raw OHLCV and produce:
      - ICT features
      - binary label: did +2R happen before -1R?
    """

    if df.empty:
        return pd.DataFrame()

    # Make sure we have a datetime index named timestamp
    if "timestamp" in df.columns:
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp")

    # Let ICTStrategy add ATR / EMAs / RSI columns
    df = strategy.prepare_data(df)

    rows: List[Dict] = []
    labels: List[int] = []

    max_idx = len(df) - label_params.horizon_bars - 1

    for idx in range(max_idx):
        feats = build_features_for_bar(df, strategy, idx)
        if feats is None:
            continue

        side = "long" if feats["side"] == 1 else "short"
        label = label_bar(df, idx, side, label_params)
        if label is None:
            continue

        rows.append(feats)
        labels.append(int(label))

    if not rows:
        logger.warning("No rows produced in ICT feature dataset.")
        return pd.DataFrame()

    X = pd.DataFrame(rows)
    X["label"] = labels

    # Clean up NaNs / infinities
    X = X.replace([np.inf, -np.inf], np.nan).dropna()

    logger.info(
        f"Built ICT feature dataset with {len(X)} rows "
        f"and {len(X.columns) - 2} features."
    )
    return X
