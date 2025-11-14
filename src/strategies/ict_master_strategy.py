from dataclasses import dataclass
from datetime import datetime, time
from typing import Dict, List, Optional, Tuple

# -------------------------------------------------
# ICT MASTER CONFIG
# -------------------------------------------------

@dataclass
class ICTMasterConfig:
    # volatility & quality
    min_atr: float = 0.0                 # VERY lenient for now
    displacement_atr_mult: float = 0.5   # low threshold for impulse leg
    max_poi_distance_atr: float = 10.0   # allow price far from POI (debug)

    # risk / reward
    min_rr: float = 0.5                  # easy to pass for now

    # structure
    swing_strength: int = 2
    lookback_bars: int = 80

    # killzones (NY time)
    killzone_am_start: time = time(9, 30)
    killzone_am_end: time = time(11, 30)
    killzone_pm_start: time = time(13, 0)
    killzone_pm_end: time = time(15, 30)


class ICTMasterStrategy:
    """
    Simplified ICT-style master strategy:

    - Bias: close vs EMA(200) (no fancy swing logic… yet)
    - Impulse leg: displacement in direction of bias over last N bars
    - POI: last opposite candle before impulse
    - Trade: only when price trades back into POI
    - Lots of logging so we can see WHY it skips.
    """

    def __init__(self, cfg_dict: Optional[Dict] = None):
        cfg_dict = cfg_dict or {}
        self.config = ICTMasterConfig(**cfg_dict)

    # ----------------- PUBLIC ----------------- #

    def generate_signals(self, df: pd.DataFrame, symbol: str) -> List["Signal"]:
        signals: List["Signal"] = []

        if df.empty or len(df) < max(self.config.lookback_bars, 50):
            return signals

        df = df.copy()
        self._ensure_atr(df)

        current = df.iloc[-1]
        ts = self._get_bar_time(current)
        atr_val = float(current.get("atr", 0.0))
        close = float(current["close"])

        logger.debug(
            f"[ICTMaster] bar {ts} | close={close:.1f} atr={atr_val:.2f}"
        )

        # 1) Killzone filter
        if not self._in_killzone(ts.time()):
            logger.debug("[ICTMaster] SKIP: outside killzone.")
            return signals

        # 2) ATR filter
        if atr_val < self.config.min_atr:
            logger.debug(
                f"[ICTMaster] SKIP: ATR {atr_val:.2f} < min {self.config.min_atr:.2f}"
            )
            return signals

        # 3) Bias
        bias = self._get_bias(df)
        logger.info(f"[ICTMaster] Bias={bias} at {ts.time()}")
        if bias == "neutral":
            logger.debug("[ICTMaster] SKIP: neutral bias.")
            return signals

        # 4) Impulse leg
        leg = self._detect_impulse_leg(df, bias)
        if leg is None:
            logger.debug("[ICTMaster] SKIP: no valid impulse leg found.")
            return signals

        leg_start, leg_end = leg
        logger.debug(f"[ICTMaster] impulse leg {leg_start}->{leg_end} ({bias})")

        # 5) POI from impulse
        poi = self._build_poi_from_leg(df, bias, leg_start, leg_end)
        if poi is None:
            logger.debug("[ICTMaster] SKIP: no POI derived from impulse.")
            return signals

        logger.debug(
            f"[ICTMaster] POI zone [{poi['zone_low']:.1f}, {poi['zone_high']:.1f}] "
            f"@ index {poi['index']}"
        )

        # 6) Check price vs POI
        if not self._price_in_poi(current, poi, atr_val):
            logger.debug(
                f"[ICTMaster] SKIP: price {close:.1f} not in/near POI zone."
            )
            return signals

        # 7) Build trade
        trade = self._build_trade_from_poi(current, poi, bias, symbol)
        if trade is None:
            logger.debug("[ICTMaster] SKIP: trade did not meet RR/structure rules.")
            return signals

        logger.info(
            f"[ICTMaster] EMIT {trade.signal_type.upper()} {symbol} "
            f"@{trade.entry_price:.1f} SL={trade.stop_loss:.1f} "
            f"TP={trade.take_profit:.1f} RR={trade.risk_reward:.2f} "
            f"(bias={bias})"
        )
        signals.append(trade)
        return signals

    # ----------------- HELPERS ----------------- #

    def _ensure_atr(self, df: pd.DataFrame, period: int = 14):
        if "atr" in df.columns:
            return
        high = df["high"]
        low = df["low"]
        close = df["close"]
        prev_close = close.shift(1)

        tr = pd.concat(
            [
                (high - low),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)

        df["atr"] = tr.rolling(period).mean()

    def _get_bar_time(self, row: pd.Series) -> datetime:
        ts = row.get("timestamp")
        if isinstance(ts, datetime):
            return ts
        if isinstance(ts, str):
            try:
                return datetime.fromisoformat(ts)
            except Exception:
                pass
        idx = row.name
        if isinstance(idx, datetime):
            return idx
        raise ValueError("Cannot infer bar timestamp for ICTMasterStrategy.")

    def _in_killzone(self, t: time) -> bool:
        c = self.config
        in_am = c.killzone_am_start <= t <= c.killzone_am_end
        in_pm = c.killzone_pm_start <= t <= c.killzone_pm_end
        return in_am or in_pm

    def _get_bias(self, df: pd.DataFrame) -> str:
        """Simple bias: close vs EMA(200)."""
        cfg = self.config
        sub = df.tail(cfg.lookback_bars).copy()
        sub["ema200"] = sub["close"].ewm(span=200, adjust=False).mean()

        ema_last = float(sub["ema200"].iloc[-1])
        price_last = float(sub["close"].iloc[-1])

        if price_last > ema_last:
            return "bullish"
        elif price_last < ema_last:
            return "bearish"
        else:
            return "neutral"

    def _detect_impulse_leg(self, df: pd.DataFrame, bias: str) -> Optional[Tuple[int, int]]:
        """Find a simple displacement leg in direction of bias."""
        cfg = self.config
        sub = df.tail(cfg.lookback_bars).copy()
        atr = sub["atr"].fillna(method="ffill")

        best_leg = None
        best_size = 0.0

        for end in range(len(sub) - 1, 5, -1):
            for length in range(3, 20):
                start = end - length
                if start < 0:
                    break

                start_price = float(sub["close"].iloc[start])
                end_price = float(sub["close"].iloc[end])
                leg_atr = float(atr.iloc[start:end + 1].mean() or 1.0)

                diff = end_price - start_price
                size = abs(diff)

                if bias == "bullish" and diff <= 0:
                    continue
                if bias == "bearish" and diff >= 0:
                    continue

                if size < cfg.displacement_atr_mult * leg_atr:
                    continue

                if size > best_size:
                    best_size = size
                    abs_start = len(df) - len(sub) + start
                    abs_end = len(df) - len(sub) + end
                    best_leg = (abs_start, abs_end)

        return best_leg

    def _build_poi_from_leg(
        self, df: pd.DataFrame, bias: str, leg_start: int, leg_end: int
    ) -> Optional[Dict]:
        """Define a simple order block POI from last opposite candle before impulse."""
        sub = df.iloc[: leg_start + 1]
        if sub.empty:
            return None

        if bias == "bullish":
            mask = sub["close"] < sub["open"]
        else:
            mask = sub["close"] > sub["open"]

        candidate_idx = np.where(mask.values)[0]
        if len(candidate_idx) == 0:
            return None

        ob_idx = int(candidate_idx[-1])
        ob_row = df.iloc[ob_idx]

        zone_low = float(min(ob_row["open"], ob_row["close"], ob_row["low"]))
        zone_high = float(max(ob_row["open"], ob_row["close"], ob_row["high"]))

        return {
            "index": ob_idx,
            "bias": bias,
            "zone_low": zone_low,
            "zone_high": zone_high,
        }

    def _price_in_poi(self, current: pd.Series, poi: Dict, atr_value: float) -> bool:
        price = float(current["close"])
        z_low = float(poi["zone_low"])
        z_high = float(poi["zone_high"])

        if z_low <= price <= z_high:
            return True

        buf = 0.2 * (atr_value or 1.0)
        if poi["bias"] == "bullish":
            return (price >= z_low - buf) and (price <= z_high + buf)
        else:
            return (price <= z_high + buf) and (price >= z_low - buf)

    def _build_trade_from_poi(
        self, current: pd.Series, poi: Dict, bias: str, symbol: str
    ) -> Optional["Signal"]:
        cfg = self.config

        entry = float(current["close"])
        atr_value = float(current.get("atr", 0.0))
        z_low = float(poi["zone_low"])
        z_high = float(poi["zone_high"])

        # distance from POI in ATRs
        distance_atr = abs(entry - (z_high if bias == "bearish" else z_low)) / max(
            atr_value, 1e-6
        )
        if distance_atr > cfg.max_poi_distance_atr:
            logger.debug(
                f"[ICTMaster] SKIP trade: POI distance {distance_atr:.2f} ATR > "
                f"{cfg.max_poi_distance_atr:.2f}"
            )
            return None

        if bias == "bullish":
            stop_loss = z_low - 0.1 * atr_value
            risk = entry - stop_loss
            if risk <= 0:
                return None
            take_profit = entry + cfg.min_rr * risk
            signal_type = "long"
        else:
            stop_loss = z_high + 0.1 * atr_value
            risk = stop_loss - entry
            if risk <= 0:
                return None
            take_profit = entry - cfg.min_rr * risk
            signal_type = "short"

        rr = abs(take_profit - entry) / abs(entry - stop_loss)
        if rr < cfg.min_rr:
            logger.debug(
                f"[ICTMaster] SKIP trade: RR {rr:.2f} < min {cfg.min_rr:.2f}"
            )
            return None

        # simple confidence heuristic
        confidence = min(0.55 + 0.05 * (rr - cfg.min_rr), 0.9)

        from src.core.models import Signal  # import here to avoid circular

        return Signal(
            timestamp=current.get("timestamp", datetime.now()),
            strategy="ict_master",
            signal_type=signal_type,
            symbol=symbol,
            entry_price=float(entry),
            stop_loss=float(stop_loss),
            take_profit=float(take_profit),
            confidence=float(confidence),
            risk_reward=float(rr),
            atr=float(atr_value),
            volume=int(current.get("volume", 0) or 0),
            metadata={
                "poi_index": poi["index"],
                "poi_zone_low": z_low,
                "poi_zone_high": z_high,
                "distance_atr": distance_atr,
                "bias": bias,
            },
        )
