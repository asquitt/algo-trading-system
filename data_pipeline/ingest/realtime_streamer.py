"""
Real-Time Data Streamer
Fetches market data and publishes to Redis streams in real-time
"""
import time
import schedule
from typing import List, Dict
from datetime import datetime
from loguru import logger

from data_pipeline.ingest.data_fetcher import MarketDataFetcher
from data_pipeline.ingest.stream_producer import StreamProducer
from data_pipeline.ingest.storage_manager import DataStorageManager


class RealTimeStreamer:
    """
    Real-time data streaming service
    Fetches market data at intervals and publishes to Redis streams
    """
    
    def __init__(
        self,
        symbols: List[str],
        interval: str = '1m',
        update_frequency: int = 60,
        store_to_db: bool = True
    ):
        """
        Initialize real-time streamer
        
        Args:
            symbols: List of symbols to track
            interval: Data interval
            update_frequency: Update frequency in seconds
            store_to_db: Whether to store data in database
        """
        self.symbols = symbols
        self.interval = interval
        self.update_frequency = update_frequency
        self.store_to_db = store_to_db
        
        # Initialize components
        self.fetcher = MarketDataFetcher()
        self.producer = StreamProducer()
        self.storage = DataStorageManager() if store_to_db else None
        
        # Tracking
        self.last_prices = {}
        self.running = False
        
        logger.info(f"Initialized streamer for {len(symbols)} symbols")
        logger.info(f"Update frequency: {update_frequency}s, Interval: {interval}")
    
    def fetch_and_stream(self):
        """Fetch latest data and publish to streams"""
        logger.info(f"Fetching data for {len(self.symbols)} symbols...")
        
        for symbol in self.symbols:
            try:
                # Fetch latest data
                df = self.fetcher.fetch_historical_data(
                    symbol,
                    period='1d',
                    interval=self.interval
                )
                
                if df is None or len(df) == 0:
                    logger.warning(f"No data for {symbol}")
                    continue
                
                # Get latest bar
                latest = df.iloc[-1]
                
                # Check if price changed (avoid duplicate publishes)
                last_price = self.last_prices.get(symbol)
                current_price = float(latest['close'])
                
                if last_price is None or abs(current_price - last_price) > 0.001:
                    # Publish to stream
                    self.producer.publish_market_data(
                        symbol=symbol,
                        timestamp=latest['timestamp'],
                        price_data={
                            'open': float(latest['open']),
                            'high': float(latest['high']),
                            'low': float(latest['low']),
                            'close': current_price,
                            'volume': float(latest['volume'])
                        },
                        metadata={
                            'interval': self.interval,
                            'source': 'yfinance'
                        }
                    )
                    
                    self.last_prices[symbol] = current_price
                    
                    # Store to database if enabled
                    if self.store_to_db and self.storage:
                        self.storage.store_dataframe(df.tail(1))
                    
                    logger.success(f"Streamed {symbol}: ${current_price:.2f}")
                
            except Exception as e:
                logger.error(f"Error streaming {symbol}: {e}")
    
    def start(self):
        """Start real-time streaming"""
        logger.info("="*70)
        logger.info("Starting Real-Time Data Streamer")
        logger.info("="*70)
        logger.info(f"Symbols: {', '.join(self.symbols)}")
        logger.info(f"Update every {self.update_frequency} seconds")
        logger.info("Press Ctrl+C to stop")
        logger.info("="*70 + "\n")
        
        self.running = True
        
        # Initial fetch
        self.fetch_and_stream()
        
        # Schedule updates
        schedule.every(self.update_frequency).seconds.do(self.fetch_and_stream)
        
        try:
            while self.running:
                schedule.run_pending()
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("\nStopping streamer...")
            self.stop()
    
    def stop(self):
        """Stop streaming"""
        self.running = False
        self.producer.close()
        logger.info("Streamer stopped")


class SignalStreamer:
    """
    Real-time signal generation and streaming
    Monitors market data and generates trading signals
    """
    
    def __init__(self, symbols: List[str]):
        """
        Initialize signal streamer
        
        Args:
            symbols: List of symbols to monitor
        """
        self.symbols = symbols
        self.fetcher = MarketDataFetcher()
        self.producer = StreamProducer()
        
        # Import here to avoid circular dependencies
        from data_pipeline.features.technical_indicators import TechnicalIndicators
        self.indicators = TechnicalIndicators()
    
    def check_signals(self):
        """Check for trading signals"""
        logger.info("Checking for signals...")
        
        for symbol in self.symbols:
            try:
                # Fetch recent data
                df = self.fetcher.fetch_historical_data(
                    symbol,
                    period='1mo',
                    interval='1d'
                )
                
                if df is None or len(df) < 50:
                    continue
                
                # Calculate indicators
                df['rsi'] = self.indicators.rsi(df)
                df['macd'], df['macd_signal'], _ = self.indicators.macd(df)
                
                # Get latest values
                latest = df.iloc[-1]
                rsi = latest['rsi']
                macd = latest['macd']
                macd_signal = latest['macd_signal']
                price = float(latest['close'])
                
                # Check for signals
                signal = None
                strength = 0
                reason = ""
                
                # RSI oversold + MACD bullish
                if rsi < 30 and macd > macd_signal:
                    signal = 'BUY'
                    strength = 0.8
                    reason = f"RSI oversold ({rsi:.1f}) + MACD bullish"
                
                # RSI overbought + MACD bearish
                elif rsi > 70 and macd < macd_signal:
                    signal = 'SELL'
                    strength = 0.8
                    reason = f"RSI overbought ({rsi:.1f}) + MACD bearish"
                
                # Publish signal if found
                if signal:
                    self.producer.publish_signal(
                        symbol=symbol,
                        signal_type=signal,
                        strength=strength,
                        price=price,
                        strategy='RSI_MACD_COMBO',
                        metadata={
                            'rsi': rsi,
                            'macd': macd,
                            'reason': reason
                        }
                    )
                    
                    # Also publish alert
                    self.producer.publish_alert(
                        alert_type='TRADING_SIGNAL',
                        message=f"{signal} signal for {symbol} @ ${price:.2f}: {reason}",
                        severity='INFO',
                        data={'symbol': symbol, 'signal': signal, 'price': price}
                    )
                
            except Exception as e:
                logger.error(f"Error checking {symbol}: {e}")
    
    def start(self, check_interval: int = 300):
        """
        Start signal monitoring
        
        Args:
            check_interval: Check interval in seconds (default 5 minutes)
        """
        logger.info("="*70)
        logger.info("Starting Signal Streamer")
        logger.info("="*70)
        logger.info(f"Monitoring {len(self.symbols)} symbols for signals")
        logger.info(f"Check every {check_interval} seconds")
        logger.info("="*70 + "\n")
        
        # Initial check
        self.check_signals()
        
        # Schedule checks
        schedule.every(check_interval).seconds.do(self.check_signals)
        
        try:
            while True:
                schedule.run_pending()
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("\nStopping signal streamer...")
            self.producer.close()


# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python realtime_streamer.py data    # Stream market data")
        print("  python realtime_streamer.py signals # Stream trading signals")
        sys.exit(1)
    
    mode = sys.argv[1]
    
    # Symbols to monitor
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    
    try:
        if mode == 'data':
            # Stream market data
            streamer = RealTimeStreamer(
                symbols=symbols,
                interval='1m',
                update_frequency=60,  # Update every minute
                store_to_db=True
            )
            streamer.start()
        
        elif mode == 'signals':
            # Stream trading signals
            signal_streamer = SignalStreamer(symbols=symbols)
            signal_streamer.start(check_interval=300)  # Check every 5 minutes
        
        else:
            print(f"Unknown mode: {mode}")
            print("Use 'data' or 'signals'")
    
    except KeyboardInterrupt:
        logger.info("\nShutting down...")
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
