# Strategy Configuration Examples

This document provides detailed examples of how to configure each trading strategy in the NQ Signal Bot.

## Strategy Configuration Structure

All strategies are configured in `config/config.py` under the `STRATEGIES` dictionary:

```python
STRATEGIES = {
    "strategy_name": {
        "enabled": True/False,
        "base_confidence": 0.5-0.9,
        # strategy-specific parameters
    }
}
```

## 1. Divergence Strategy

### Configuration Example:
```python
"divergence": {
    "enabled": True,
    "base_confidence": 0.55,
    "rsi_period": 14,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "min_divergence_points": 4,
    "pivot_strength": 5,
    "ema_period": 21,
    "atr_period": 14,
    "stop_loss_atr": 1.0,
    "take_profit_atr": 1.5
}
```

### Parameter Descriptions:
- **rsi_period**: RSI calculation period (default: 14)
- **macd_fast/slow/signal**: MACD parameters (default: 12, 26, 9)
- **min_divergence_points**: Minimum price difference for divergence (default: 4 points)
- **pivot_strength**: Strength of pivot points to identify (default: 5 bars each side)
- **ema_period**: EMA period for trend confirmation (default: 21)
- **stop_loss_atr**: Stop loss multiplier (default: 1.0 × ATR)
- **take_profit_atr**: Take profit multiplier (default: 1.5 × ATR)

### Signal Logic:
1. Detect RSI/MACD divergence with price
2. Wait for price to reclaim EMA(21)
3. Enter on confirmation candle
4. Confidence boosted by VWAP confluence and divergence strength

## 2. EMA Pullback Strategy

### Configuration Example:
```python
"ema_pullback": {
    "enabled": True,
    "base_confidence": 0.6,
    "fast_ema": 21,
    "medium_ema": 50,
    "slow_ema": 200,
    "min_slope": 0.1,
    "volume_multiplier": 1.2,
    "atr_period": 14
}
```

### Parameter Descriptions:
- **fast_ema**: Fast EMA period (default: 21)
- **medium_ema**: Medium EMA period (default: 50)
- **slow_ema**: Slow EMA period (default: 200)
- **min_slope**: Minimum EMA slope (% per bar, default: 0.1%)
- **volume_multiplier**: Volume confirmation multiplier (default: 1.2× average)

### Signal Logic:
1. Check EMA alignment (fast > medium > slow for bullish)
2. Identify pullback to EMA(21) or EMA(50)
3. Confirm with rejection candle and volume
4. Filter by VWAP/session bias
5. Target: 2R with trailing stop under EMA(21)

## 3. HTF Supertrend Strategy

### Configuration Example:
```python
"htf_supertrend": {
    "enabled": True,
    "base_confidence": 0.65,
    "htf_ema": 50,
    "supertrend_atr": 10,
    "supertrend_multiplier": 3.0,
    "news_buffer_minutes": 2,
    "max_stop_atr": 1.2
}
```

### Parameter Descriptions:
- **htf_ema**: Higher timeframe EMA for bias (default: 50)
- **supertrend_atr**: Supertrend ATR period (default: 10)
- **supertrend_multiplier**: Supertrend multiplier (default: 3.0)
- **news_buffer_minutes**: Minutes to avoid trading around news (default: 2)
- **max_stop_atr**: Maximum stop loss in ATR units (default: 1.2)

### Signal Logic:
1. Determine HTF bias using 15-min EMA(50)
2. Trade 1-min Supertrend in HTF direction
3. Enter on pullback to Supertrend line
4. Stop: Supertrend flip or max 1.2 × ATR
5. Target: Trail by Supertrend, hard TP at 2.5R

## 4. Supply/Demand Strategy

### Configuration Example:
```python
"supply_demand": {
    "enabled": True,
    "base_confidence": 0.6,
    "pivot_strength": 8,
    "min_impulse_atr": 1.5,
    "zone_buffer": 2.0,
    "max_touches": 2,
    "stop_buffer_atr": 0.5
}
```

### Parameter Descriptions:
- **pivot_strength**: Pivot strength for zone identification (default: 8)
- **min_impulse_atr**: Minimum impulse for zone creation (default: 1.5 × ATR)
- **zone_buffer**: Buffer around zone edges (default: 2.0 points)
- **max_touches**: Maximum touches before zone invalidates (default: 2)
- **stop_buffer_atr**: Stop loss buffer beyond zone (default: 0.5 × ATR)

### Signal Logic:
1. Identify supply/demand zones using ZigZag pivots
2. Require minimum 1.5 × ATR impulse for zone creation
3. Enter on rejection candle at zone edge
4. Fresh zones (0 touches) get confidence boost
5. Target: 2R to next trouble area

## 5. VWAP Strategy

### Configuration Example:
```python
"vwap": {
    "enabled": True,
    "base_confidence": 0.55,
    "session_start": "09:30",
    "band_std_dev": [1.0, 2.0],
    "adx_period": 14,
    "adx_threshold": 18,
    "bb_period": 60
}
```

### Parameter Descriptions:
- **session_start**: Session start time for VWAP reset (default: 09:30 ET)
- **band_std_dev**: VWAP band standard deviations (default: [1σ, 2σ])
- **adx_threshold**: ADX threshold for trending market (default: 18)
- **bb_period**: Bollinger Band period for regime detection (default: 60)

### Signal Logic:
- **Ranging Market** (ADX < 18): Fade 2σ band extremes, target VWAP
- **Trending Market** (ADX ≥ 18): Pullback to VWAP with trend direction
- **Stop**: 0.75 × ATR for mean reversion, 1.0 × ATR for trend following

## 6. Opening Range Breakout (ORB)

### Configuration Example:
```python
"orb": {
    "enabled": True,
    "base_confidence": 0.7,
    "range_minutes": 15,
    "buffer_atr_factor": 0.15,
    "min_buffer_points": 3.0,
    "volume_confirm": 1.5,
    "max_gap_atr": 0.7,
    "alternative_range": 5
}
```

### Parameter Descriptions:
- **range_minutes**: Opening range duration (default: 15 minutes)
- **buffer_atr_factor**: Buffer as ATR factor (default: 0.15)
- **min_buffer_points**: Minimum buffer in points (default: 3.0)
- **volume_confirm**: Volume confirmation multiplier (default: 1.5×)
- **max_gap_atr**: Maximum gap size before waiting for retest (default: 0.7 × ATR)
- **alternative_range**: Alternative range for high volatility (default: 5 minutes)

### Signal Logic:
1. Define opening range (09:30-09:45 ET)
2. Calculate ORH (high) and ORL (low)
3. Add buffer: max(0.15 × ATR, 3 pts)
4. Enter on breakout with volume confirmation
5. Stop: Opposite side of OR ± 0.5 × ATR
6. Target: 1R partial, trail by swing points

## Confidence Scoring

Each strategy has a base confidence that can be boosted by:

### Divergence:
- +0.1 for both RSI and MACD divergence
- +0.05 for strong divergence (>10% strength)
- +0.1 for VWAP confluence

### EMA Pullback:
- +0.1 for strong EMA slope (>0.2%/bar)
- +0.05 for high volume (>1.5× average)

### HTF Supertrend:
- +0.1 for strong HTF bias (>0.5% above/below EMA)
- +0.05 for high volume confirmation

### Supply/Demand:
- +0.1 for strong zones (>2.0 strength)
- +0.1 for fresh zones (0 touches)
- +0.1 for HTF confluence

### VWAP:
- +0.1 for strong rejection candles (wick > 2× body)
- +0.1 for strong trends (ADX > 25)

### ORB:
- +0.1 for strong volume (>2× average)
- +0.05 for clean breakouts (no immediate pullback)

## Risk Management Integration

All strategies are filtered through global risk management:

1. **Chop Guard**: Skip if ATR < 6 points (except ORB)
2. **Daily Loss Limit**: Stop if -$1,500 or -3R reached
3. **Position Limits**: Max 1 concurrent position
4. **Confidence Filter**: Minimum 40% confidence required
5. **Risk/Reward**: Minimum 1:1 required

## Customization Examples

### Conservative Settings:
```python
"divergence": {
    "enabled": True,
    "base_confidence": 0.7,  # Higher confidence required
    "min_divergence_points": 6,  # Stronger divergence
    # ... other parameters unchanged
}
```

### Aggressive Settings:
```python
"ema_pullback": {
    "enabled": True,
    "base_confidence": 0.5,  # Lower confidence threshold
    "min_slope": 0.05,  # Less strict slope requirement
    "volume_multiplier": 1.0,  # No volume requirement
    # ... other parameters unchanged
}
```

### Strategy-Specific Settings:
```python
"vwap": {
    "enabled": True,
    "base_confidence": 0.6,
    "adx_threshold": 15,  # Lower threshold for trending markets
    "band_std_dev": [0.5, 1.0, 2.0],  # More bands for mean reversion
    # ... other parameters unchanged
}
```

## Testing Changes

After modifying strategy configurations:

1. **Restart the bot** to load new settings
2. **Run in paper trading mode** to test changes
3. **Use backtest mode** for comprehensive testing
4. **Monitor performance** metrics for each strategy
5. **Adjust gradually** - change one parameter at a time

## Performance Monitoring

Track these metrics for each strategy:
- Win rate and average R-multiple
- Signal frequency and timing
- Performance in different market regimes
- Confidence score distribution
- Risk-adjusted returns

## Best Practices

1. **Start Conservative**: Begin with higher confidence thresholds
2. **Test Thoroughly**: Use paper trading before live deployment
3. **Monitor Regimes**: Different strategies work in different markets
4. **Adjust Gradually**: Make small changes and monitor results
5. **Keep Records**: Document configuration changes and results
6. **Review Regularly**: Monthly strategy performance reviews