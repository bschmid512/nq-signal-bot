#!/usr/bin/env python3
"""
Quick Test - Manually test if signal generation is working
Run this in your project directory: python test_signals.py
"""

import sys
import sqlite3
from datetime import datetime

print("="*70)
print(" QUICK SIGNAL GENERATION TEST")
print("="*70)

# Test 1: Check database
print("\n[1] Checking database...")
try:
    conn = sqlite3.connect('data/nq_signals.db')
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM market_data WHERE symbol='MNQ1!' AND timeframe='5'")
    count = cursor.fetchone()[0]
    print(f"    ✓ Database has {count} bars of 5-min data")
    
    if count == 0:
        print("\n    ❌ PROBLEM FOUND: No data in database!")
        print("    → Webhooks are being received but data isn't being stored")
        print("    → Check db_manager.insert_market_data() method")
        conn.close()
        sys.exit(1)
    
    if count < 50:
        print(f"\n    ⚠️  Only {count} bars available (need 50+)")
        print(f"    → Wait for more data to accumulate")
        print(f"    → Need about {(50 - count) * 5} more minutes")
    
    # Get latest bar
    cursor.execute("""
        SELECT timestamp, close, volume 
        FROM market_data 
        WHERE symbol='MNQ1!' AND timeframe='5' 
        ORDER BY timestamp DESC LIMIT 1
    """)
    latest = cursor.fetchone()
    if latest:
        print(f"    Latest bar: {latest[0]} @ {latest[1]:.2f}")
    
    conn.close()
    
except Exception as e:
    print(f"    ❌ Database error: {e}")
    sys.exit(1)

# Test 2: Try to import and initialize engine
print("\n[2] Testing signal engine initialization...")
try:
    from src.utils.database import DatabaseManager
    from src.core.engine import SignalGenerationEngine
    
    db = DatabaseManager('data/nq_signals.db')
    engine = SignalGenerationEngine(db)
    
    print(f"    ✓ Engine initialized with {len(engine.strategies)} strategies")
    for name in engine.strategies.keys():
        print(f"      - {name}")
    
except Exception as e:
    print(f"    ❌ Engine initialization failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Try to generate signals manually
print("\n[3] Attempting manual signal generation...")
try:
    import asyncio
    
    # Create a fake webhook data point
    test_data = {
        'timestamp': datetime.now().isoformat(),
        'symbol': 'MNQ1!',
        'timeframe': '5',
        'open': 25660.0,
        'high': 25665.0,
        'low': 25655.0,
        'close': 25662.0,
        'volume': 1000
    }
    
    print(f"    Testing with fake data: {test_data['symbol']} @ {test_data['close']}")
    
    # Run the signal generation
    async def test():
        signals = await engine.process_new_data(test_data)
        return signals
    
    signals = asyncio.run(test())
    
    print(f"\n    Result: {len(signals)} signals generated")
    
    if signals:
        print("\n    ✓ SIGNALS FOUND!")
        for sig in signals:
            print(f"      → {sig.strategy} {sig.signal_type} @ {sig.entry_price:.2f}")
            print(f"        Confidence: {sig.confidence:.2%}, R/R: {sig.risk_reward:.2f}")
    else:
        print("\n    ℹ️  No signals (this might be normal given market conditions)")
        print("    → Market may not be presenting clear setups")
        print("    → Add logging to see which strategies were checked")
    
except Exception as e:
    print(f"\n    ❌ Signal generation failed: {e}")
    import traceback
    traceback.print_exc()
    print("\n    This is the root cause - signal generation is erroring out!")

# Test 4: Check if pandas-ta is working
print("\n[4] Testing indicator calculation...")
try:
    import pandas as pd
    import pandas_ta as ta
    
    # Get real data from database
    from src.utils.database import DatabaseManager
    db = DatabaseManager('data/nq_signals.db')
    df = db.get_latest_data('MNQ1!', '5', limit=50)
    
    if len(df) < 14:
        print(f"    ⚠️  Need more data ({len(df)} bars)")
    else:
        # Try to calculate ATR
        atr = ta.atr(df['high'], df['low'], df['close'], length=14)
        latest_atr = atr.iloc[-1]
        
        print(f"    ✓ ATR calculation works")
        print(f"    Latest ATR: {latest_atr:.2f} points")
        
        if latest_atr < 15:
            print(f"\n    ⚠️  ATR is LOW ({latest_atr:.2f} points)")
            print(f"    → Market is in low volatility / choppy")
            print(f"    → This explains why no signals are generating")
            print(f"    → Strategies are correctly waiting for better conditions")
        
except Exception as e:
    print(f"    ❌ Indicator calculation failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("TEST COMPLETE")
print("="*70)
print("""
If you see errors above, that's your problem!
If everything works but no signals, the market just doesn't have setups right now.

Next steps:
1. If database has 0 bars → Fix data storage
2. If engine fails to initialize → Fix imports/strategy init  
3. If signal generation errors → That's the bug to fix
4. If everything works but no signals → Market conditions not suitable
""")
print("="*70)
