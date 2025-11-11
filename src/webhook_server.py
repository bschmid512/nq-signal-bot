import asyncio
import json
from datetime import datetime
from typing import Dict, Any
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from loguru import logger
import uvicorn

from config.config import config
from src.core.engine import SignalGenerationEngine
from src.utils.database import DatabaseManager

# Pydantic models for request validation
class TradingViewWebhook(BaseModel):
    """TradingView webhook payload structure"""
    timestamp: str
    symbol: str
    timeframe: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    exchange: str = None
    
class SignalResponse(BaseModel):
    """Response structure for signal generation"""
    success: bool
    signals_generated: int
    signals: list = []
    message: str = ""

class WebhookServer:
    """FastAPI server for receiving TradingView webhooks"""
    
    def __init__(self):
        self.app = FastAPI(title="NQ Signal Bot Webhook Server", version="1.0.0")
        self.setup_middleware()
        self.setup_routes()
        
        # Initialize database and engine
        self.db_manager = DatabaseManager(config.DATABASE_PATH)
        self.signal_engine = SignalGenerationEngine(self.db_manager)
        
        logger.info("Webhook server initialized")
    
    def setup_middleware(self):
        """Setup CORS and other middleware"""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure appropriately for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def setup_routes(self):
        """Setup API routes"""
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            return {"status": "healthy", "timestamp": datetime.now().isoformat()}
        
        @self.app.get("/signals")
        async def get_recent_signals(limit: int = 50, strategy: str = None):
            """Get recent trading signals"""
            try:
                signals_df = self.db_manager.get_recent_signals(strategy, limit)
                
                if signals_df.empty:
                    return {"signals": [], "count": 0}
                
                # Convert DataFrame to list of dicts
                signals = signals_df.to_dict('records')
                
                # Format timestamps
                for signal in signals:
                    if 'timestamp' in signal:
                        signal['timestamp'] = signal['timestamp'].isoformat()
                    if 'created_at' in signal:
                        signal['created_at'] = signal['created_at'].isoformat()
                
                return {"signals": signals, "count": len(signals)}
                
            except Exception as e:
                logger.error(f"Error fetching signals: {e}")
                raise HTTPException(status_code=500, detail="Internal server error")
        
        @self.app.get("/performance")
        async def get_performance(days: int = 30):
            """Get strategy performance metrics"""
            try:
                performance_df = self.db_manager.get_strategy_performance(days)
                
                if performance_df.empty:
                    return {"performance": [], "message": "No performance data available"}
                
                performance = performance_df.to_dict('records')
                
                # Format dates
                for record in performance:
                    if 'date' in record:
                        record['date'] = record['date'].isoformat()
                
                return {"performance": performance, "days": days}
                
            except Exception as e:
                logger.error(f"Error fetching performance: {e}")
                raise HTTPException(status_code=500, detail="Internal server error")
        
        @self.app.get("/risk")
        async def get_risk_summary():
            """Get current risk management summary"""
            try:
                risk_summary = self.signal_engine.risk_manager.get_risk_summary()
                return risk_summary
                
            except Exception as e:
                logger.error(f"Error fetching risk summary: {e}")
                raise HTTPException(status_code=500, detail="Internal server error")
        
        @self.app.post("/webhook", response_model=SignalResponse)
        async def receive_webhook(webhook_data: TradingViewWebhook, background_tasks: BackgroundTasks):
            """Receive TradingView webhook data"""
            try:
                logger.info(f"Received webhook: {webhook_data.symbol} {webhook_data.timeframe} {webhook_data.close}")
                
                # Process webhook data
                market_data = {
                    'timestamp': webhook_data.timestamp,
                    'symbol': webhook_data.symbol,
                    'timeframe': webhook_data.timeframe,
                    'open': webhook_data.open,
                    'high': webhook_data.high,
                    'low': webhook_data.low,
                    'close': webhook_data.close,
                    'volume': webhook_data.volume
                }
                
                # Process signal generation in background
                background_tasks.add_task(self._process_signal, market_data)
                
                return SignalResponse(
                    success=True,
                    signals_generated=0,  # Will be updated asynchronously
                    message="Webhook received and processing"
                )
                
            except Exception as e:
                logger.error(f"Error processing webhook: {e}")
                raise HTTPException(status_code=500, detail="Internal server error")
        
        @self.app.post("/webhook/raw")
        async def receive_raw_webhook(request: Request, background_tasks: BackgroundTasks):
            """Receive raw webhook data for flexibility"""
            try:
                # Get raw request data
                body = await request.body()
                data = json.loads(body) if body else {}
                
                logger.info(f"Received raw webhook: {data}")
                
                # Try to parse as TradingView format
                if all(key in data for key in ['timestamp', 'symbol', 'timeframe', 'open', 'high', 'low', 'close', 'volume']):
                    market_data = {
                        'timestamp': data['timestamp'],
                        'symbol': data['symbol'],
                        'timeframe': data['timeframe'],
                        'open': float(data['open']),
                        'high': float(data['high']),
                        'low': float(data['low']),
                        'close': float(data['close']),
                        'volume': int(data['volume'])
                    }
                    
                    background_tasks.add_task(self._process_signal, market_data)
                    
                    return {"success": True, "message": "Webhook processed"}
                else:
                    logger.warning(f"Invalid webhook format: {data}")
                    raise HTTPException(status_code=400, detail="Invalid webhook format")
                    
            except json.JSONDecodeError:
                logger.error("Invalid JSON in webhook")
                raise HTTPException(status_code=400, detail="Invalid JSON")
            except Exception as e:
                logger.error(f"Error processing raw webhook: {e}")
                raise HTTPException(status_code=500, detail="Internal server error")
        
        @self.app.get("/strategies")
        async def get_strategies():
            """Get enabled strategies and their configurations"""
            try:
                strategies = {}
                for name, config in config.STRATEGIES.items():
                    strategies[name] = {
                        'enabled': config.get('enabled', False),
                        'base_confidence': config.get('base_confidence', 0.5)
                    }
                
                return {"strategies": strategies}
                
            except Exception as e:
                logger.error(f"Error fetching strategies: {e}")
                raise HTTPException(status_code=500, detail="Internal server error")
        
        @self.app.post("/cleanup")
        async def cleanup_database():
            """Manual cleanup of old data"""
            try:
                self.db_manager.cleanup_old_data(days_to_keep=30)
                return {"success": True, "message": "Database cleanup completed"}
                
            except Exception as e:
                logger.error(f"Error during cleanup: {e}")
                raise HTTPException(status_code=500, detail="Internal server error")
    
    async def _process_signal(self, market_data: Dict):
        """Process signal generation in background"""
        try:
            # Generate signals
            signals = await self.signal_engine.process_new_data(market_data)
            
            if signals:
                logger.info(f"Generated {len(signals)} signals for {market_data['symbol']}")
                
                # Log signals
                for signal in signals:
                    logger.info(f"Signal: {signal.strategy} {signal.signal_type} "
                               f"{signal.symbol} @ {signal.entry_price} "
                               f"(Confidence: {signal.confidence:.2f})")
            else:
                logger.debug(f"No signals generated for {market_data['symbol']}")
                
        except Exception as e:
            logger.error(f"Error processing signal: {e}")
    
    def run(self):
        """Run the webhook server"""
        try:
            logger.info(f"Starting webhook server on {config.WEBHOOK_HOST}:{config.WEBHOOK_PORT}")
            uvicorn.run(
                self.app,
                host=config.WEBHOOK_HOST,
                port=config.WEBHOOK_PORT,
                log_level="info"
            )
        except Exception as e:
            logger.error(f"Error running webhook server: {e}")
            raise

# Global webhook server instance
webhook_server = WebhookServer()

# Export the FastAPI app for running with uvicorn
app = webhook_server.app

if __name__ == "__main__":
    webhook_server.run()