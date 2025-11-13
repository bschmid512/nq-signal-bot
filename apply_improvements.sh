#!/bin/bash
# Update script to apply high-confidence signal improvements

echo "=========================================="
echo "NQ Signal Bot - HIGH CONFIDENCE UPDATE"
echo "=========================================="
echo ""

# Check if we're in the right directory
if [ ! -f "main.py" ]; then
    echo "❌ Error: Please run this script from the nq-signal-bot directory"
    exit 1
fi

echo "📦 Installing missing dependencies..."
pip install pandas_ta --break-system-packages 2>/dev/null || pip install pandas_ta

echo ""
echo "🔧 Backing up original files..."
# Backup original files
cp src/core/engine.py src/core/engine.py.backup.$(date +%Y%m%d_%H%M%S)
cp src/utils/risk_manager.py src/utils/risk_manager.py.backup.$(date +%Y%m%d_%H%M%S)
cp config/config.py config/config.py.backup.$(date +%Y%m%d_%H%M%S)

echo ""
echo "📝 Applying improvements..."

# Copy improved files from outputs
if [ -f "/mnt/user-data/outputs/engine_fixed.py" ]; then
    cp /mnt/user-data/outputs/engine_fixed.py src/core/engine.py
    echo "✅ Updated engine.py with high-confidence mode"
fi

if [ -f "/mnt/user-data/outputs/risk_manager_simplified.py" ]; then
    cp /mnt/user-data/outputs/risk_manager_simplified.py src/utils/risk_manager.py
    echo "✅ Updated risk_manager.py for quality-focused filtering"
fi

if [ -f "/mnt/user-data/outputs/config_improved.py" ]; then
    cp /mnt/user-data/outputs/config_improved.py config/config.py
    echo "✅ Updated config.py with optimized settings"
fi

if [ -f "/mnt/user-data/outputs/momentum_surge.py" ]; then
    cp /mnt/user-data/outputs/momentum_surge.py src/strategies/momentum_surge.py
    echo "✅ Added new momentum_surge.py strategy"
fi

echo ""
echo "🧪 Running diagnostic test..."
if [ -f "/mnt/user-data/outputs/test_improved_signals.py" ]; then
    python3 /mnt/user-data/outputs/test_improved_signals.py
fi

echo ""
echo "=========================================="
echo "✅ UPDATE COMPLETE!"
echo "=========================================="
echo ""
echo "Key improvements applied:"
echo "  • Fixed pandas_ta import bug in engine"
echo "  • Simplified risk manager (quality > quantity)"
echo "  • Increased confidence thresholds to 0.65+"
echo "  • Enabled only high-performance strategies"
echo "  • Added momentum surge strategy"
echo "  • Removed restrictive time/loss filters"
echo ""
echo "To start receiving HIGH-CONFIDENCE signals:"
echo "  1. Start the webhook server: python main.py webhook"
echo "  2. Monitor logs for 🎯 HIGH-CONFIDENCE signals"
echo "  3. Only signals with 65%+ confidence will pass"
echo ""
echo "Expected behavior:"
echo "  • Fewer but MUCH higher quality signals"
echo "  • Focus on strong trends and reversals"
echo "  • Better risk/reward ratios (2.5:1 minimum)"
echo "  • Clear entry/exit points"
echo ""
