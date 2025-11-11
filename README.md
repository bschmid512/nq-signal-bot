# NQ Intraday Momentum Signal Bot

A comprehensive Python-based signal generation system designed specifically for NQ (Nasdaq-100 futures) intraday trading. The bot employs six distinct trading strategies with advanced risk management and confidence scoring.

## Features

### Trading Strategies
1. **Divergence Detection** - RSI/MACD with swing pivots
2. **EMA Pullback** - Trend-following with EMA alignment
3. **HTF Supertrend** - Higher timeframe confirmation
4. **Supply/Demand Zones** - Zone-based trading
5. **VWAP Strategy** - Mean reversion and trend-following
6. **Opening Range Breakout** - ORB with multiple variations

### Risk Management
- Dynamic position sizing based on ATR
- Daily loss limits and drawdown controls
- Market environment assessment (chop detection)
- Confidence-based signal filtering
- Maximum concurrent position limits

### Technical Architecture
- **Real-time Data**: TradingView webhooks with SQLite3 persistence
- **Multi-strategy Engine**: Modular strategy design with standard interfaces
- **Risk Controls**: Global risk management with daily limits
- **Signal Output**: Actionable alerts with entry, stop, and target levels

## Installation

### Prerequisites
- Python 3.8+
- TradingView account with alert webhooks
- SQLite3

### Setup

1. **Clone the repository**:
```bash
git clone <repository-url>
cd nq-signal-bot
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Configure environment variables**:
Create a `.env` file in the root directory:
```
WEBHOOK_SECRET=your_webhook_secret_here
```

4. **Initialize the database**:
```bash
python -c "from src.utils.database import DatabaseManager; DatabaseManager('data/nq_signals.db').init_database()"
```

## Usage

### Interactive Mode
Run the bot in interactive mode:
```bash
python main.py
```

This will show a menu with options:
1. **Start Webhook Server** - Listen for TradingView alerts
2. **Paper Trading Mode** - Generate signals without execution
3. **Run Backtest** - Test strategies on historical data
4. **View Performance** - See strategy performance metrics
5. **Cleanup Database** - Remove old data
6. **Exit**

### Command Line Mode

**Start webhook server**:
```bash
python main.py webhook
```

**Paper trading mode**:
```bash
python main.py paper
```

**Run backtest**:
```bash
python main.py backtest 2024-01-01 2024-12-31
```

### TradingView Setup

1. **Create alerts** in TradingView for NQ1! (or your preferred NQ symbol)

2. **Configure webhook URL**:
   - URL: `http://your-server-ip:8000/webhook`
   - Method: POST
   - Body (JSON):
   ```json
   {
     "timestamp": "{{timenow}}",
     "symbol": "{{ticker}}",
     "timeframe": "{{interval}}",
     "open": {{open}},
     "high": {{high}},
     "low": {{low}},
     "close": {{close}},
     "volume": {{volume}}
   }
   ```

3. **Set alert frequency** to "Every time" for continuous signal generation

## Configuration

### Strategy Settings
Edit `config/config.py` to customize strategy parameters:

```python
STRATEGIES = {
    "divergence": {
        "enabled": True,
        "base_confidence": 0.55,
        "rsi_period": 14,
        # ... other parameters
    },
    # ... other strategies
}
```

### Risk Management
- **Daily Loss Limit**: Maximum daily loss before trading stops
- **Position Sizing**: Risk per trade (default 1% of account)
- **Confidence Threshold**: Minimum signal confidence for execution
- **Market Filters**: ATR minimums and chop detection

## API Endpoints

When running the webhook server, these endpoints are available:

- **GET /health** - Health check
- **GET /signals** - Recent trading signals
- **GET /performance** - Strategy performance metrics
- **GET /risk** - Current risk management status
- **POST /webhook** - TradingView webhook endpoint
- **POST /webhook/raw** - Raw webhook endpoint

## Strategy Details

### 1. Divergence Detection
- **Method**: RSI/MACD divergence with swing pivot confirmation
- **Entry**: Price reclaim of EMA(21) after divergence
- **Stop**: 1.0 × ATR(14)
- **Target**: 1.5 × ATR(14)
- **Confidence**: 0.55 base + confluence bonuses

### 2. EMA Pullback
- **Trend**: EMA(21) > EMA(50) > EMA(200) for bullish
- **Entry**: Pullback to EMA(21/50) with rejection candle
- **Filters**: VWAP/session bias, volume confirmation
- **Target**: 2R with trailing stop

### 3. HTF Supertrend
- **HTF Bias**: 15-min EMA(50) determines direction
- **Entry**: 1-min Supertrend pullback in HTF direction
- **Stop**: Supertrend flip or 1.2 × ATR
- **Target**: Trail by Supertrend, hard TP at 2.5R

### 4. Supply/Demand Zones
- **Zones**: ZigZag pivots with 1.5 × ATR impulse minimum
- **Entry**: Rejection candle at zone with confirmation
- **Quality**: Max 2 touches, fresh zones preferred
- **Risk**: Beyond zone + 0.5 × ATR buffer

### 5. VWAP Strategy
- **Ranging**: Fade 2σ band extremes, target VWAP
- **Trending**: Pullback to VWAP with ADX > 18
- **Regime**: Dual approach based on market conditions
- **Stop**: 0.75 × ATR for mean reversion

### 6. Opening Range Breakout
- **ORB**: 09:30-09:45 ET (15 minutes)
- **Entry**: Break ORH/ORL with buffer + volume confirmation
- **Buffer**: max(0.15 × ATR, 3 pts)
- **Alternative**: 5-minute ORB for high volatility

## Risk Management Rules

### Global Controls
- **Max Positions**: 1 concurrent position
- **Daily Loss Limit**: $1,500 or 3R
- **Chop Guard**: Skip if ATR < 6 points
- **News Buffer**: Avoid 2 minutes before/after news

### Position Sizing
- **Risk per Trade**: 1% of account balance
- **Confidence Adjustment**: Multiply by signal confidence
- **Maximum Position**: 10% of account value
- **ATR-Based**: Dynamic stops based on volatility

## Monitoring and Maintenance

### Daily Tasks
- Check risk metrics and performance
- Review signal quality and execution
- Monitor database size and cleanup if needed

### Weekly Tasks
- Analyze strategy performance by market regime
- Adjust strategy parameters if needed
- Review and update risk management settings

### Monthly Tasks
- Comprehensive performance review
- Strategy optimization and parameter tuning
- Risk management calibration

## Troubleshooting

### Common Issues

**No signals generated**:
- Check TradingView alert configuration
- Verify webhook URL is accessible
- Check strategy enabled status in config
- Review market conditions (chop guard may be active)

**Database errors**:
- Ensure write permissions to data directory
- Check SQLite3 installation
- Verify database schema is initialized

**Performance issues**:
- Monitor memory usage with large datasets
- Regular database cleanup recommended
- Consider increasing analysis intervals

### Logs
- Console logs: Real-time output
- File logs: `logs/nq_bot.log` (rotated daily)
- Log level: Configurable in `config.py`

## Development

### Adding New Strategies
1. Create new strategy module in `src/strategies/`
2. Inherit from base strategy interface
3. Add configuration to `config/config.py`
4. Register in `SignalGenerationEngine._init_strategies()`

### Custom Indicators
1. Add to `src/indicators/custom.py`
2. Follow existing indicator patterns
3. Test with sample data before integration

### Database Schema Changes
1. Update `DatabaseManager.init_database()`
2. Add migration scripts for existing databases
3. Update all queries that use modified tables

## Performance Considerations

- **Memory**: Keep reasonable data history limits
- **CPU**: Avoid over-analysis on 1-minute data
- **Database**: Regular cleanup prevents performance degradation
- **Network**: Webhook server optimized for high-frequency alerts

## Disclaimer

This software is for educational and informational purposes only. Trading involves substantial risk of loss. Past performance does not guarantee future results. Always test thoroughly in paper trading mode before using with real money.

## License

MIT License - See LICENSE file for details

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review logs for error messages
3. Create an issue with detailed information
4. Include configuration and log excerpts

## Roadmap

- [ ] Integration with live data feeds
- [ ] Advanced machine learning signals
- [ ] Portfolio-level risk management
- [ ] Real-time performance dashboards
- [ ] Multi-asset support
- [ ] Automated strategy optimization