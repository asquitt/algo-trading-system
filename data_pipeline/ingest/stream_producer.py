"""
Redis Stream Producer
Publishes market data events to Redis streams for real-time processing
"""
import redis
import json
from typing import Dict, List, Optional
from datetime import datetime
from loguru import logger
import os
from dotenv import load_dotenv

load_dotenv()


class StreamProducer:
    """
    Produces events to Redis streams
    Handles market data updates, signals, and trading events
    """
    
    def __init__(
        self,
        redis_host: str = None,
        redis_port: int = None,
        max_stream_length: int = 10000
    ):
        """
        Initialize stream producer
        
        Args:
            redis_host: Redis host
            redis_port: Redis port
            max_stream_length: Maximum stream length (FIFO eviction)
        """
        self.redis_host = redis_host or os.getenv('REDIS_HOST', 'localhost')
        self.redis_port = redis_port or int(os.getenv('REDIS_PORT', 6380))
        self.max_stream_length = max_stream_length
        
        # Redis connection
        self.redis_client = None
        self._connect()
        
        # Stream names
        self.MARKET_DATA_STREAM = 'market:data'
        self.SIGNALS_STREAM = 'trading:signals'
        self.TRADES_STREAM = 'trading:trades'
        self.ALERTS_STREAM = 'trading:alerts'
    
    def _connect(self):
        """Connect to Redis"""
        try:
            self.redis_client = redis.Redis(
                host=self.redis_host,
                port=self.redis_port,
                db=0,
                decode_responses=True,
                socket_connect_timeout=5
            )
            self.redis_client.ping()
            logger.success(f"✓ Connected to Redis at {self.redis_host}:{self.redis_port}")
        except redis.ConnectionError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    def publish_market_data(
        self,
        symbol: str,
        timestamp: datetime,
        price_data: Dict,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Publish market data update to stream
        
        Args:
            symbol: Stock symbol
            timestamp: Data timestamp
            price_data: Dictionary with OHLCV data
            metadata: Optional metadata
            
        Returns:
            Message ID
        """
        event = {
            'event_type': 'market_data',
            'symbol': symbol,
            'timestamp': timestamp.isoformat(),
            'open': price_data.get('open'),
            'high': price_data.get('high'),
            'low': price_data.get('low'),
            'close': price_data.get('close'),
            'volume': price_data.get('volume'),
            'metadata': json.dumps(metadata or {})
        }
        
        try:
            # Add to stream with max length
            msg_id = self.redis_client.xadd(
                self.MARKET_DATA_STREAM,
                event,
                maxlen=self.max_stream_length,
                approximate=True
            )
            logger.debug(f"Published market data: {symbol} @ {price_data['close']}")
            return msg_id
        except Exception as e:
            logger.error(f"Failed to publish market data: {e}")
            raise
    
    def publish_signal(
        self,
        symbol: str,
        signal_type: str,
        strength: float,
        price: float,
        strategy: str,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Publish trading signal to stream
        
        Args:
            symbol: Stock symbol
            signal_type: BUY, SELL, HOLD
            strength: Signal strength (0-1)
            price: Current price
            strategy: Strategy name
            metadata: Optional metadata
            
        Returns:
            Message ID
        """
        event = {
            'event_type': 'signal',
            'symbol': symbol,
            'signal_type': signal_type,
            'strength': strength,
            'price': price,
            'strategy': strategy,
            'timestamp': datetime.utcnow().isoformat(),
            'metadata': json.dumps(metadata or {})
        }
        
        try:
            msg_id = self.redis_client.xadd(
                self.SIGNALS_STREAM,
                event,
                maxlen=self.max_stream_length,
                approximate=True
            )
            logger.info(f"Published signal: {signal_type} {symbol} @ {price} (strength: {strength})")
            return msg_id
        except Exception as e:
            logger.error(f"Failed to publish signal: {e}")
            raise
    
    def publish_trade(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        order_id: str,
        status: str,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Publish trade execution to stream
        
        Args:
            symbol: Stock symbol
            side: BUY or SELL
            quantity: Number of shares
            price: Execution price
            order_id: Order ID
            status: Order status
            metadata: Optional metadata
            
        Returns:
            Message ID
        """
        event = {
            'event_type': 'trade',
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'price': price,
            'order_id': order_id,
            'status': status,
            'timestamp': datetime.utcnow().isoformat(),
            'metadata': json.dumps(metadata or {})
        }
        
        try:
            msg_id = self.redis_client.xadd(
                self.TRADES_STREAM,
                event,
                maxlen=self.max_stream_length,
                approximate=True
            )
            logger.info(f"Published trade: {side} {quantity} {symbol} @ {price}")
            return msg_id
        except Exception as e:
            logger.error(f"Failed to publish trade: {e}")
            raise
    
    def publish_alert(
        self,
        alert_type: str,
        message: str,
        severity: str = 'INFO',
        data: Optional[Dict] = None
    ) -> str:
        """
        Publish alert/notification to stream
        
        Args:
            alert_type: Type of alert
            message: Alert message
            severity: INFO, WARNING, ERROR, CRITICAL
            data: Optional additional data
            
        Returns:
            Message ID
        """
        event = {
            'event_type': 'alert',
            'alert_type': alert_type,
            'message': message,
            'severity': severity,
            'timestamp': datetime.utcnow().isoformat(),
            'data': json.dumps(data or {})
        }
        
        try:
            msg_id = self.redis_client.xadd(
                self.ALERTS_STREAM,
                event,
                maxlen=self.max_stream_length,
                approximate=True
            )
            logger.warning(f"Alert [{severity}]: {message}")
            return msg_id
        except Exception as e:
            logger.error(f"Failed to publish alert: {e}")
            raise
    
    def publish_batch(
        self,
        stream_name: str,
        events: List[Dict]
    ) -> List[str]:
        """
        Publish multiple events efficiently using pipeline
        
        Args:
            stream_name: Stream to publish to
            events: List of event dictionaries
            
        Returns:
            List of message IDs
        """
        try:
            pipeline = self.redis_client.pipeline()
            
            for event in events:
                pipeline.xadd(
                    stream_name,
                    event,
                    maxlen=self.max_stream_length,
                    approximate=True
                )
            
            msg_ids = pipeline.execute()
            logger.info(f"Published {len(events)} events to {stream_name}")
            return msg_ids
        except Exception as e:
            logger.error(f"Failed to publish batch: {e}")
            raise
    
    def get_stream_info(self, stream_name: str) -> Dict:
        """
        Get information about a stream
        
        Args:
            stream_name: Stream name
            
        Returns:
            Stream info dictionary
        """
        try:
            info = self.redis_client.xinfo_stream(stream_name)
            return {
                'length': info['length'],
                'first_entry': info['first-entry'],
                'last_entry': info['last-entry'],
                'groups': info['groups']
            }
        except redis.ResponseError:
            return {'length': 0, 'exists': False}
    
    def trim_stream(self, stream_name: str, max_length: int):
        """
        Trim stream to maximum length
        
        Args:
            stream_name: Stream to trim
            max_length: Maximum length
        """
        try:
            self.redis_client.xtrim(stream_name, maxlen=max_length, approximate=True)
            logger.info(f"Trimmed {stream_name} to {max_length} entries")
        except Exception as e:
            logger.error(f"Failed to trim stream: {e}")
    
    def close(self):
        """Close Redis connection"""
        if self.redis_client:
            self.redis_client.close()
            logger.info("Redis connection closed")


# Example usage and testing
if __name__ == "__main__":
    from data_pipeline.ingest.data_fetcher import MarketDataFetcher
    import time
    
    logger.info("="*70)
    logger.info("Testing Stream Producer")
    logger.info("="*70)
    
    try:
        # Initialize producer
        producer = StreamProducer()
        
        # Test 1: Publish market data
        logger.info("\nTest 1: Publishing market data...")
        
        fetcher = MarketDataFetcher()
        df = fetcher.fetch_historical_data('AAPL', period='1d', interval='5m')
        
        if df is not None and len(df) > 0:
            # Publish latest bars
            for i, row in df.tail(5).iterrows():
                producer.publish_market_data(
                    symbol='AAPL',
                    timestamp=row['timestamp'],
                    price_data={
                        'open': row['open'],
                        'high': row['high'],
                        'low': row['low'],
                        'close': row['close'],
                        'volume': row['volume']
                    },
                    metadata={'interval': '5m', 'source': 'yfinance'}
                )
                time.sleep(0.1)  # Small delay
        
        # Test 2: Publish trading signal
        logger.info("\nTest 2: Publishing trading signal...")
        producer.publish_signal(
            symbol='AAPL',
            signal_type='BUY',
            strength=0.85,
            price=150.25,
            strategy='RSI_MEAN_REVERSION',
            metadata={'rsi': 28.5, 'reason': 'oversold'}
        )
        
        # Test 3: Publish trade
        logger.info("\nTest 3: Publishing trade...")
        producer.publish_trade(
            symbol='AAPL',
            side='BUY',
            quantity=100,
            price=150.30,
            order_id='TEST_ORDER_001',
            status='FILLED',
            metadata={'commission': 1.00}
        )
        
        # Test 4: Publish alert
        logger.info("\nTest 4: Publishing alert...")
        producer.publish_alert(
            alert_type='PRICE_ALERT',
            message='AAPL reached target price of $150',
            severity='INFO',
            data={'symbol': 'AAPL', 'target': 150, 'current': 150.30}
        )
        
        # Test 5: Get stream info
        logger.info("\nTest 5: Stream information...")
        for stream in [producer.MARKET_DATA_STREAM, producer.SIGNALS_STREAM, 
                      producer.TRADES_STREAM, producer.ALERTS_STREAM]:
            info = producer.get_stream_info(stream)
            if info.get('exists') != False:
                logger.info(f"{stream}: {info['length']} messages")
        
        logger.success("\n✓ All producer tests completed!")
        
    except redis.ConnectionError:
        logger.error("Could not connect to Redis. Make sure Docker is running:")
        logger.error("  cd docker && docker-compose up -d")
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        if 'producer' in locals():
            producer.close()
