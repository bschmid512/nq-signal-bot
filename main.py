#!/usr/bin/env python3
"""
NQ Intraday Momentum Signal Bot
Main entry point for the trading signal generation system
"""

import asyncio
import sys
import signal
from datetime import datetime, time
from loguru import logger
import schedule
import time as time_module

from config.config import config
from src.webhook_server import webhook_server
from src.utils.database import DatabaseManager
from src.utils.risk_manager import RiskManager

class NQSignalBot:
    """Main bot class that orchestrates all components"""
    
    def __init__(self):
        self.running = False
        self.db_manager = DatabaseManager(config.DATABASE_PATH)
        self.risk_manager = RiskManager()
        
        # Setup logging
        self.setup_logging()
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        logger.info("NQ Signal Bot initialized")
    
    def setup_logging(self):
        """Setup logging configuration"""
        # Remove default handler
        logger.remove()
        
        # Add console handler
        logger.add(
            sys.stdout,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            level=config.LOG_LEVEL,
            colorize=True
        )
        
        # Add file handler
        logger.add(
            config.LOG_FILE,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level=config.LOG_LEVEL,
            rotation="10 MB",
            retention="30 days",
            compression="zip"
        )
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.running = False
    
    def run_scheduled_tasks(self):
        """Run scheduled maintenance tasks"""
        try:
            # Reset daily metrics at market open
            current_time = datetime.now().time()
            market_open = time(9, 30)
            
            if (current_time.hour == market_open.hour and 
                current_time.minute == market_open.minute):
                self.risk_manager.reset_daily_metrics()
                logger.info("Daily metrics reset for market open")
            
            # Cleanup old data at end of day
            if current_time.hour == 17 and current_time.minute == 0:
                self.db_manager.cleanup_old_data(days_to_keep=30)
                logger.info("Database cleanup completed")
                
        except Exception as e:
            logger.error(f"Error in scheduled tasks: {e}")
    
    def start_webhook_server(self):
        """Start the webhook server"""
        try:
            logger.info("Starting webhook server...")
            webhook_server.run()
        except Exception as e:
            logger.error(f"Error starting webhook server: {e}")
            raise
        
    def start_backtest_mode(self, start_date: str, end_date: str):
        """Start bot in backtest mode"""
        try:
            logger.info(f"Starting backtest mode from {start_date} to {end_date}")
            
            # Import backtest module
            from src.backtest.engine import BacktestEngine
            
            # Initialize and run backtest
            backtest_engine = BacktestEngine(self.db_manager, self.risk_manager)
            results = backtest_engine.run_backtest(start_date, end_date) or {}

            # Safely pull values with defaults
            total_trades = results.get("total_trades", 0)
            win_rate = results.get("win_rate", 0.0)
            total_pnl = results.get("total_pnl", 0.0)
            max_dd = results.get("max_drawdown", 0.0)
            sharpe = results.get("sharpe_ratio", 0.0)

            logger.info("Backtest Results:")
            logger.info(f"Total Trades: {total_trades}")
            logger.info(f"Win Rate: {win_rate:.2%}")
            logger.info(f"Total PnL: ${total_pnl:.2f}")
            logger.info(f"Max Drawdown: {max_dd:.2%}")
            logger.info(f"Sharpe Ratio: {sharpe:.2f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in backtest mode: {e}")
            return {}

    
    def start_paper_trading_mode(self):
        """Start bot in paper trading mode"""
        try:
            logger.info("Starting paper trading mode...")
            
            # In paper trading mode, the bot will generate signals
            # but not execute trades automatically
            
            # Start webhook server for signal generation
            self.start_webhook_server()
            
        except Exception as e:
            logger.error(f"Error in paper trading mode: {e}")
    
    def display_menu(self):
        """Display main menu"""
        print("\n" + "="*50)
        print("NQ Intraday Momentum Signal Bot")
        print("="*50)
        print("1. Start Webhook Server (Live Signals)")
        print("2. Start Paper Trading Mode")
        print("3. Run Backtest")
        print("4. View Performance Summary")
        print("5. Cleanup Database")
        print("6. Exit")
        print("="*50)
    
    def get_user_choice(self) -> str:
        """Get user menu choice"""
        try:
            choice = input("\nEnter your choice (1-6): ").strip()
            return choice
        except KeyboardInterrupt:
            print("\nExiting...")
            return "6"
        except Exception:
            return ""
    
    def handle_menu_choice(self, choice: str):
        """Handle user menu choice"""
        if choice == "1":
            print("\nStarting Webhook Server...")
            print("The bot will listen for TradingView webhooks at:")
            print(f"http://localhost:{config.WEBHOOK_PORT}/webhook")
            print("\nConfigure your TradingView alert to POST to this URL")
            print("Press Ctrl+C to stop the server")
            
            try:
                self.start_webhook_server()
            except KeyboardInterrupt:
                print("\nWebhook server stopped")
            
        elif choice == "2":
            print("\nStarting Paper Trading Mode...")
            self.start_paper_trading_mode()
            
        elif choice == "3":
            print("\nBacktest Mode - using last 20000 bars from database (dates ignored)")
            # Dates are only used if DB is empty (synthetic fallback)
            default_start = "2000-01-01"
            default_end = datetime.now().strftime("%Y-%m-%d")
            self.start_backtest_mode(default_start, default_end)

            
        elif choice == "4":
            print("\nPerformance Summary:")
            risk_summary = self.risk_manager.get_risk_summary()
            for key, value in risk_summary.items():
                if isinstance(value, float):
                    print(f"{key}: {value:.4f}")
                else:
                    print(f"{key}: {value}")
            
        elif choice == "5":
            print("\nCleaning up database...")
            self.db_manager.cleanup_old_data(days_to_keep=30)
            print("Database cleanup completed")
            
        elif choice == "6":
            print("\nExiting NQ Signal Bot...")
            return False
            
        else:
            print("\nInvalid choice. Please try again.")
            
        return True
    
    def run(self):
        """Main run loop"""
        logger.info("Starting NQ Signal Bot")
        
        # Setup scheduled tasks
        schedule.every().day.at("09:30").do(self.risk_manager.reset_daily_metrics)
        schedule.every().day.at("17:00").do(lambda: self.db_manager.cleanup_old_data(30))
        
        if len(sys.argv) > 1:
            # Command line mode
            mode = sys.argv[1].lower()
            
            if mode == "webhook":
                self.start_webhook_server()
            elif mode == "paper":
                self.start_paper_trading_mode()
            elif mode == "backtest" and len(sys.argv) >= 4:
                start_date = sys.argv[2]
                end_date = sys.argv[3]
                self.start_backtest_mode(start_date, end_date)
            else:
                print("Usage:")
                print("  python main.py webhook    # Start webhook server")
                print("  python main.py paper      # Start paper trading")
                print("  python main.py backtest <start_date> <end_date>")
                print("  python main.py            # Interactive mode")
        else:
            # Interactive mode
            self.running = True
            while self.running:
                self.display_menu()
                choice = self.get_user_choice()
                
                if not self.handle_menu_choice(choice):
                    break
                
                # Run scheduled tasks
                schedule.run_pending()
                
                # Small delay to prevent high CPU usage
                time_module.sleep(0.1)

def main():
    """Main entry point"""
    bot = NQSignalBot()
    bot.run()

if __name__ == "__main__":
    main()