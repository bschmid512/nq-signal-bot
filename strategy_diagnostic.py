#!/usr/bin/env python3
"""
Strategy Diagnostic - See why signals aren't generating
This will show you what each strategy is looking for and why it's rejecting signals
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime

print("="*80)
print(" STRATEGY DIAGNOSTIC - Why No Signals?")
print("="*80)

# Initialize
from src.utils.database import DatabaseManager
from src.core.engine import SignalGenerationEngine

db = DatabaseManager('data/nq_signals.db')
engine = SignalGenerationEngine(db)

# Get real market data
df = db.get_latest_data('MNQ1!', '5', limit=200)
print(f"\n✓ Loaded {len(df)} bars of market data")

# Calculate indicators
df_with_indicators = engine._calculate_indicators(df)

# Get latest data
latest = df_with_indicators.iloc[-1]
print(f"Latest close: {latest['close']:.2f}")
print(f"Latest ATR: {latest['atr']:.2f}")
print(f"Latest RSI: {latest['rsi']:.2f}")
print(f"Latest VWAP: {latest['vwap']:.2f}")

print("\n" + "="*80)
print(" CHECKING EACH STRATEGY")
print("="*80)

# ============================================================================
# 1. DIVERGENCE STRATEGY
# ============================================================================
print("\n[1] DIVERGENCE STRATEGY")
print("-" * 80)

divergence_strat = engine.strategies['divergence']
print("Looking for:")
print("  • RSI or MACD divergence with price")
print("  • Price reclaiming EMA(21) after divergence")
print("  • Confirmation candle")

# Check current conditions
ema_21 = latest['ema_21']
close = latest['close']
rsi = latest['rsi']

print(f"\nCurrent conditions:")
print(f"  Close: {close:.2f}")
print(f"  EMA(21): {ema_21:.2f}")
print(f"  Close vs EMA(21): {('ABOVE ✓' if close > ema_21 else 'BELOW ✗')}")
print(f"  RSI: {rsi:.2f}")

# Check for recent divergences
from src.indicators.custom import CustomIndicators
custom_ind = CustomIndicators()

recent_bars = df_with_indicators.tail(120)
rsi_divs = custom_ind.detect_divergence(recent_bars, recent_bars['rsi'], min_bars=5, max_bars=120)
macd_divs = custom_ind.detect_divergence(recent_bars, recent_bars['macd'], min_bars=5, max_bars=120)

print(f"\nDivergences detected:")
print(f"  RSI divergences: {len(rsi_divs)}")
print(f"  MACD divergences: {len(macd_divs)}")

if len(rsi_divs) == 0 and len(macd_divs) == 0:
    print("  ❌ NO DIVERGENCES FOUND - Strategy waiting for divergence to form")
else:
    print("  ✓ Divergences present, checking for entry trigger...")

# ============================================================================
# 2. EMA PULLBACK STRATEGY
# ============================================================================
print("\n[2] EMA PULLBACK STRATEGY")
print("-" * 80)

print("Looking for:")
print("  • Strong trend: EMA(21) > EMA(50) > EMA(200) for bullish")
print("  • Pullback to EMA(21) or EMA(50)")
print("  • Rejection candle with volume")
print("  • Price above VWAP")

ema_50 = latest['ema_50']
ema_200 = latest['ema_200']
vwap = latest['vwap']

print(f"\nCurrent conditions:")
print(f"  EMA(21): {ema_21:.2f}")
print(f"  EMA(50): {ema_50:.2f}")
print(f"  EMA(200): {ema_200:.2f}")

# Check bullish alignment
bullish_trend = ema_21 > ema_50 > ema_200
print(f"  Bullish trend (21>50>200): {bullish_trend} {'✓' if bullish_trend else '✗'}")

# Check bearish alignment
bearish_trend = ema_21 < ema_50 < ema_200
print(f"  Bearish trend (21<50<200): {bearish_trend} {'✓' if bearish_trend else '✗'}")

if not bullish_trend and not bearish_trend:
    print("  ❌ NO CLEAR TREND - EMAs not aligned, strategy waiting")

print(f"  Close vs VWAP: {close:.2f} vs {vwap:.2f} = {('ABOVE ✓' if close > vwap else 'BELOW ✗')}")

# ============================================================================
# 3. HTF SUPERTREND STRATEGY (ADAPTED FOR 5-MIN)
# ============================================================================
print("\n[3] HTF SUPERTREND STRATEGY (Adapted for 5-min)")
print("-" * 80)

print("Looking for:")
print("  • EMA(200) on 5-min determines HTF bias (proxy for longer timeframe)")
print("  • Supertrend on 5-min for pullback entries")
print("  • Price must be >0.3% away from EMA(200) for bias")

supertrend = latest['supertrend']
ema_200 = latest.get('ema_200', None)

print(f"\nCurrent conditions:")
print(f"  Close: {close:.2f}")

# Check Supertrend
if pd.isna(supertrend):
    print(f"  Supertrend: NaN ❌ (CALCULATION FAILED - needs fixing)")
else:
    print(f"  Supertrend: {supertrend:.2f}")
    if close > supertrend:
        print(f"  Close vs Supertrend: ABOVE (uptrend) ✓")
    else:
        print(f"  Close vs Supertrend: BELOW (downtrend) ✗")

# Check HTF Bias (EMA 200)
if ema_200 is None or pd.isna(ema_200):
    print(f"  EMA(200): Not calculated ❌")
    print(f"  HTF Bias: Cannot determine (EMA(200) missing)")
else:
    print(f"  EMA(200): {ema_200:.2f}")
    distance_pct = (close - ema_200) / ema_200 * 100
    print(f"  Distance from EMA(200): {distance_pct:+.2f}%")
    
    if distance_pct > 0.3:
        print(f"  HTF Bias: BULLISH ✓ (price >0.3% above EMA)")
    elif distance_pct < -0.3:
        print(f"  HTF Bias: BEARISH ✓ (price >0.3% below EMA)")
    else:
        print(f"  HTF Bias: NEUTRAL ✗ (price within ±0.3% of EMA - no clear trend)")

# Summary
print(f"\n  Strategy status:")
if pd.isna(supertrend):
    print(f"  ❌ Supertrend NaN - Apply Step 1 of Option B fix")
elif ema_200 is None or pd.isna(ema_200):
    print(f"  ❌ EMA(200) missing - Check indicator calculation")
elif abs(distance_pct) < 0.3:
    print(f"  ⚠️  No HTF bias - Price too close to EMA(200), waiting for trend")
else:
    print(f"  ✓ Strategy operational - Looking for pullbacks to Supertrend")

# ============================================================================
# 4. SUPPLY/DEMAND STRATEGY
# ============================================================================
print("\n[4] SUPPLY/DEMAND STRATEGY")
print("-" * 80)

print("Looking for:")
print("  • Valid supply/demand zones (ZigZag pivots + 1.5× ATR impulse)")
print("  • Price at zone edge")
print("  • Rejection candle")
print("  • Max 2 touches per zone")

# This strategy calculates zones internally - hard to diagnose without running it
print("\nThis strategy needs to calculate zones from pivots...")
print("If no signals: No valid zones detected or price not at zone edges")

# ============================================================================
# 5. VWAP STRATEGY
# ============================================================================
print("\n[5] VWAP STRATEGY")
print("-" * 80)

print("Looking for:")
print("  • ADX < 18: Fade 2σ band extremes (mean reversion)")
print("  • ADX ≥ 18: Trade pullbacks to VWAP (trend following)")

adx = latest['adx']
vwap_upper_2 = latest['vwap_upper_2']
vwap_lower_2 = latest['vwap_lower_2']

print(f"\nCurrent conditions:")
print(f"  ADX: {adx:.2f} → {'RANGING (fade extremes)' if adx < 18 else 'TRENDING (trade pullbacks)'}")
print(f"  Close: {close:.2f}")
print(f"  VWAP: {vwap:.2f}")
print(f"  Upper 2σ: {vwap_upper_2:.2f}")
print(f"  Lower 2σ: {vwap_lower_2:.2f}")

distance_to_vwap = abs(close - vwap)
at_upper_band = close >= vwap_upper_2
at_lower_band = close <= vwap_lower_2

print(f"\n  Distance to VWAP: {distance_to_vwap:.2f} points")
print(f"  At upper 2σ band: {at_upper_band} {'✓' if at_upper_band else '✗'}")
print(f"  At lower 2σ band: {at_lower_band} {'✓' if at_lower_band else '✗'}")

if not at_upper_band and not at_lower_band:
    if adx < 18:
        print("  ❌ Ranging market but not at 2σ extremes - waiting")
    else:
        if distance_to_vwap > latest['atr']:
            print(f"  ❌ Too far from VWAP ({distance_to_vwap:.2f} > ATR {latest['atr']:.2f}) - waiting")

# ============================================================================
# 6. ORB STRATEGY
# ============================================================================
print("\n[6] ORB (Opening Range Breakout) STRATEGY")
print("-" * 80)

print("Looking for:")
print("  • Opening range established (09:30-09:45 ET)")
print("  • Breakout above ORH or below ORL")
print("  • Volume confirmation")

print("\nCurrent time: After market hours")
print("ORB only trades during market hours (9:30 AM - 4:00 PM ET)")
print("\n  ⚠️  If you're testing after hours, ORB won't trigger")

# ============================================================================
# RISK MANAGEMENT FILTERS
# ============================================================================
print("\n" + "="*80)
print(" RISK MANAGEMENT FILTERS")
print("="*80)

print("\nEven if a strategy finds a setup, it must pass these filters:")
print(f"  • Confidence ≥ 40%")
print(f"  • Risk/Reward ≥ 1.0")
print(f"  • ATR ≥ {engine.risk_manager.min_atr} points")
print(f"  • Max positions: {engine.risk_manager.max_positions}")
print(f"  • Daily loss limit: ${engine.risk_manager.max_daily_loss}")

risk_summary = engine.risk_manager.get_risk_summary()
print(f"\nCurrent risk status:")
print(f"  Trading allowed: {risk_summary['trading_allowed']}")
print(f"  Daily PnL: ${risk_summary['daily_pnl']:.2f}")
print(f"  Current positions: {risk_summary['current_positions']}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print(" SUMMARY & RECOMMENDATIONS")
print("="*80)

print("\nMost likely reasons for no signals:")

issues = []

# Check divergence
if len(rsi_divs) == 0 and len(macd_divs) == 0:
    issues.append("1. DIVERGENCE: No divergences detected → Strategy waiting")

# Check EMA alignment
if not bullish_trend and not bearish_trend:
    issues.append("2. EMA PULLBACK: EMAs not aligned → No clear trend")

# Check HTF Supertrend
if pd.isna(latest.get('supertrend', np.nan)):
    issues.append("3. HTF SUPERTREND: Supertrend = NaN → Apply supertrend fix from Option B")
elif latest.get('ema_200') is None or pd.isna(latest.get('ema_200')):
    issues.append("3. HTF SUPERTREND: EMA(200) missing → Check indicator calculation")
else:
    ema_200_dist = abs((close - latest['ema_200']) / latest['ema_200'] * 100)
    if ema_200_dist < 0.3:
        issues.append("3. HTF SUPERTREND: Price within ±0.3% of EMA(200) → No HTF bias")

# Check VWAP position
if not at_upper_band and not at_lower_band and adx < 18:
    issues.append("3. VWAP: Price not at 2σ extremes in ranging market")
elif adx >= 18 and distance_to_vwap > latest['atr']:
    issues.append("3. VWAP: Too far from VWAP for pullback entry")

# Check time
current_hour = datetime.now().hour
if current_hour < 9 or current_hour >= 16:
    issues.append("4. ORB: After market hours → Strategy inactive")

if issues:
    for issue in issues:
        print(f"  • {issue}")
else:
    print("  • All conditions look reasonable... checking strategy logic")

print("\n" + "="*80)
print(" WHAT TO DO")
print("="*80)

print("""
If the market conditions above don't match strategy requirements:
  → This is NORMAL - strategies wait for specific setups
  → Quality over quantity is the goal
  → Check back during active market hours (9:30-4:00 ET)

If Supertrend shows NaN:
  → Apply the Supertrend fix from Option B (Step 1 in OPTION_B_INSTRUCTIONS.md)
  → This is required for HTF Supertrend to work

If HTF Supertrend shows EMA(200) missing:
  → Check that EMA(200) is being calculated in engine.py
  → Make sure htf_ema config is set to 200 (Step 2 in Option B)

If conditions look good but still no signals:
  → Add detailed logging to each strategy to see exact rejection reasons
  → Lower confidence thresholds temporarily to see if signals appear
  → Check if strategy-specific filters are too strict

Note on HTF Supertrend:
  → Now adapted for 5-min data (uses EMA(200) as HTF proxy)
  → If you applied Option B changes, this strategy should work
  → Requires strong trends (price >0.3% from EMA(200))

Recommended: Add this logging to engine.py _generate_signals:
  logger.info(f"[{strategy_name}] Checking for signals...")
  strategy_signals = strategy.generate_signals(df_with_indicators, symbol)
  logger.info(f"[{strategy_name}] → {len(strategy_signals)} signals found")
""")

print("="*80)