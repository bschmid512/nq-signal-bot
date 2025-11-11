# NQ Signal Bot - Project Structure

## Overview
This document outlines the complete project structure and file organization for the NQ Intraday Momentum Signal Bot.

## Directory Structure

```
nq-signal-bot/
├── src/                          # Main source code directory
│   ├── core/                     # Core engine and signal processing
│   │   ├── __init__.py
│   │   └── engine.py             # Main signal generation engine
│   ├── strategies/               # Trading strategy implementations
│   │   ├── __init__.py
│   │   ├── divergence.py         # RSI/MACD divergence strategy
│   │   ├── ema_pullback.py       # EMA pullback trend-following
│   │   ├── htf_supertrend.py     # HTF confirmation with Supertrend
│   │   ├── supply_demand.py      # Supply and demand zones
│   │   ├── vwap.py              # VWAP mean reversion and trending
│   │   └── orb.py               # Opening range breakout
│   ├── indicators/              # Custom technical indicators
│   │   ├── __init__.py
│   │   └── custom.py            # VWAP, pivot points, divergence detection
│   ├── utils/                   # Utility functions
│   │   ├── __init__.py
│   │   ├── database.py          # Database management
│   │   └── risk_manager.py      # Risk management and position sizing
│   ├── backtest/                # Backtesting engine
│   │   ├── __init__.py
│   │   └── engine.py            # Backtesting functionality
│   └── webhook_server.py        # FastAPI webhook server
├── config/                      # Configuration files
│   └── config.py               # Main configuration settings
├── data/                       # Database and data files
├── logs/                       # Log files
├── tests/                      # Unit tests
├── docs/                       # Documentation
│   ├── tradingview_setup.md    # TradingView configuration guide
│   └── strategy_examples.md    # Strategy configuration examples
├── requirements.txt            # Python dependencies
├── main.py                     # Main entry point
├── install.sh                  # Installation script
├── README.md                   # Main documentation
├── .env.example                # Environment variables template
└── PROJECT_STRUCTURE.md        # This file
```

## File Descriptions

### Core Components

#### `main.py`
- Main entry point for the application
- Provides interactive menu and command-line modes
- Orchestrates all bot components

#### `src/core/engine.py`
- **SignalGenerationEngine**: Main engine that processes market data and generates signals
- Integrates all strategies and risk management
- Manages data flow and signal distribution

#### `src/webhook_server.py`
- FastAPI server for receiving TradingView webhooks
- REST API endpoints for monitoring and management
- Background task processing for signal generation

### Strategy Modules

#### `src/strategies/divergence.py`
- RSI/MACD divergence detection with swing pivot confirmation
- Base confidence: 0.55, max confidence: 0.75
- Requires price reclaim of EMA(21) for entry

#### `src/strategies/ema_pullback.py`
- Trend-following strategy using EMA alignment
- EMA(21) > EMA(50) > EMA(200) for bullish trend
- Pullback entries to dynamic support/resistance

#### `src/strategies/htf_supertrend.py`
- Higher timeframe confirmation strategy
- 15-min EMA(50) determines directional bias
- 1-min Supertrend for precise entries

#### `src/strategies/supply_demand.py`
- Supply and demand zone trading
- ZigZag pivot identification with impulse requirements
- Fresh zones get confidence bonuses

#### `src/strategies/vwap.py`
- Dual regime approach (ranging vs trending)
- VWAP with standard deviation bands
- ADX-based regime detection

#### `src/strategies/orb.py`
- Opening range breakout strategy
- 15-minute default range (09:30-09:45 ET)
- Volume confirmation and gap handling

### Technical Indicators

#### `src/indicators/custom.py`
- **VWAP Calculation**: Session-based VWAP with bands
- **Pivot Point Detection**: Fractal-based pivot identification
- **Divergence Detection**: Price vs indicator divergence
- **Supertrend**: Custom Supertrend implementation
- **ADX**: Average Directional Index calculation

### Utility Functions

#### `src/utils/database.py`
- SQLite3 database management
- Market data storage and retrieval
- Signal persistence and tracking
- Performance metrics storage

#### `src/utils/risk_manager.py`
- Global risk management controls
- Position sizing calculations
- Market environment assessment
- Daily loss limits and drawdown controls

### Configuration

#### `config/config.py`
- Central configuration management
- Strategy parameters and settings
- Risk management thresholds
- Database and server configuration

### Documentation

#### `README.md`
- Main project documentation
- Installation and usage instructions
- Strategy descriptions and API reference

#### `docs/tradingview_setup.md`
- Detailed TradingView configuration guide
- Webhook setup and troubleshooting
- Alert message formatting

#### `docs/strategy_examples.md`
- Strategy configuration examples
- Parameter descriptions and tuning guides
- Performance optimization tips

## Key Features

### Multi-Strategy Approach
- Six distinct trading strategies
- Each strategy optimized for different market conditions
- Modular design for easy addition of new strategies

### Advanced Risk Management
- Dynamic position sizing based on volatility
- Daily loss limits and drawdown controls
- Market environment assessment (chop detection)
- Confidence-based signal filtering

### Real-Time Processing
- TradingView webhook integration
- SQLite3 database for data persistence
- FastAPI server for high-performance processing
- Background task processing for signal generation

### Comprehensive Monitoring
- Performance tracking by strategy
- Risk metrics and position management
- Database cleanup and maintenance
- Detailed logging and error handling

## Usage Examples

### Starting the Bot
```bash
# Interactive mode
python main.py

# Webhook server mode
python main.py webhook

# Paper trading mode
python main.py paper

# Backtest mode
python main.py backtest 2024-01-01 2024-12-31
```

### API Endpoints
```bash
# Health check
curl http://localhost:8000/health

# Get recent signals
curl http://localhost:8000/signals

# Get performance metrics
curl http://localhost:8000/performance

# Send test webhook
curl -X POST http://localhost:8000/webhook \
  -H "Content-Type: application/json" \
  -d '{"timestamp": "2024-01-01T10:00:00Z", "symbol": "NQ1!", "timeframe": "5", "open": 15000, "high": 15010, "low": 14990, "close": 15005, "volume": 5000}'
```

## Development Guidelines

### Adding New Strategies
1. Create new strategy module in `src/strategies/`
2. Follow existing strategy patterns and interfaces
3. Add configuration to `config/config.py`
4. Register in `SignalGenerationEngine._init_strategies()`
5. Test thoroughly in paper trading mode

### Custom Indicators
1. Add to `src/indicators/custom.py`
2. Follow existing indicator patterns
3. Test with sample data before integration
4. Update strategy modules to use new indicators

### Database Schema Changes
1. Update `DatabaseManager.init_database()`
2. Create migration scripts for existing databases
3. Update all queries that use modified tables
4. Test with both new and existing databases

## Performance Considerations

### Memory Usage
- Keep reasonable data history limits
- Regular database cleanup prevents memory issues
- Monitor memory usage with large datasets

### CPU Usage
- Avoid over-analysis on 1-minute data
- Use appropriate analysis intervals
- Monitor server performance under load

### Database Performance
- Regular cleanup of old data
- Proper indexing for query performance
- Monitor database size and growth

## Support and Maintenance

### Logging
- Console logs: Real-time output
- File logs: `logs/nq_bot.log` (rotated daily)
- Log level: Configurable in `config.py`

### Monitoring
- Health check endpoint for server status
- Performance metrics tracking
- Risk management status monitoring

### Troubleshooting
- Check logs for error messages
- Verify configuration settings
- Test webhook connectivity
- Monitor database performance

## Conclusion

The NQ Signal Bot provides a comprehensive framework for algorithmic trading signal generation. With its modular architecture, advanced risk management, and multi-strategy approach, it offers a robust foundation for NQ futures trading.

The project structure is designed for extensibility and maintainability, making it easy to add new strategies, indicators, and features as needed.