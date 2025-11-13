"""
Redis Stream Consumer
Consumes events from Redis streams and processes them
"""
import redis
import json
from typing import Dict, Callable, Optional, List
from datetime import datetime
from loguru import logger
import os
import time
import signal
import sys
from dotenv import load_dotenv

load_dotenv()


class StreamConsumer:
    """
    Consumes events from Redis streams
    Processes market data, signals, and trading events in real-time
    """
    
    def __init__(
        self,
        consumer_name: str,
        group_name: str = 'default',
        redis_host: str = None,
        redis_port: int = None
    ):
        """
        Initialize stream consumer
        
        Args:
            consumer_name: Unique consumer name
            group_name: Consumer group name
            redis_host: Redis host
            redis_port: Redis port
        """
        self.consumer_name = consumer_name
        self.group_name = group_name
        self.redis_host = redis_host or os.getenv('REDIS_HOST', 'localhost')
        self.redis_port = redis_port or int(os.getenv('REDIS_PORT', 6380))
        
        # Redis connection
        self.redis_client = None
        self._connect()
        
        # Stream names
        self.MARKET_DATA_STREAM = 'market:data'
        self.SIGNALS_STREAM = 'trading:signals'
        self.TRADES_STREAM = 'trading:trades'
        self.ALERTS_STREAM = 'trading:alerts'
        
        # Event handlers
        self.handlers = {}
        
        # Running flag
        self.running = False
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
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
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.stop()
    
    def register_handler(
        self,
        event_type: str,
        handler: Callable[[Dict], None]
    ):
        """
        Register event handler
        
        Args:
            event_type: Type of event to handle
            handler: Callable that processes the event
        """
        self.handlers[event_type] = handler
        logger.info(f"Registered handler for: {event_type}")
    
    def create_consumer_group(self, stream_name: str):
        """
        Create consumer group for stream
        
        Args:
            stream_name: Stream name
        """
        try:
            self.redis_client.xgroup_create(
                stream_name,
                self.group_name,
                id='0',
                mkstream=True
            )
            logger.info(f"Created consumer group '{self.group_name}' for {stream_name}")
        except redis.ResponseError as e:
            if 'BUSYGROUP' in str(e):
                logger.debug(f"Consumer group already exists for {stream_name}")
            else:
                raise
    
    def consume_stream(
        self,
        stream_name: str,
        block_ms: int = 5000,
        count: int = 10
    ):
        """
        Consume messages from a single stream
        
        Args:
            stream_name: Stream to consume from
            block_ms: Block timeout in milliseconds
            count: Number of messages to fetch
        """
        try:
            # Ensure consumer group exists
            self.create_consumer_group(stream_name)
            
            while self.running:
                # Read from stream
                messages = self.redis_client.xreadgroup(
                    groupname=self.group_name,
                    consumername=self.consumer_name,
                    streams={stream_name: '>'},
                    count=count,
                    block=block_ms
                )
                
                # Process messages
                if messages:
                    for stream, msg_list in messages:
                        for msg_id, msg_data in msg_list:
                            self._process_message(stream, msg_id, msg_data)
                            
                            # Acknowledge message
                            self.redis_client.xack(stream, self.group_name, msg_id)
                
        except redis.ConnectionError as e:
            logger.error(f"Redis connection error: {e}")
            raise
        except Exception as e:
            logger.error(f"Error consuming stream: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def consume_multiple_streams(
        self,
        stream_names: List[str],
        block_ms: int = 5000,
        count: int = 10
    ):
        """
        Consume messages from multiple streams simultaneously
        
        Args:
            stream_names: List of streams to consume from
            block_ms: Block timeout in milliseconds
            count: Number of messages to fetch per stream
        """
        try:
            # Create consumer groups for all streams
            for stream_name in stream_names:
                self.create_consumer_group(stream_name)
            
            # Build streams dict
            streams = {stream: '>' for stream in stream_names}
            
            logger.info(f"Consuming from streams: {', '.join(stream_names)}")
            
            while self.running:
                # Read from multiple streams
                messages = self.redis_client.xreadgroup(
                    groupname=self.group_name,
                    consumername=self.consumer_name,
                    streams=streams,
                    count=count,
                    block=block_ms
                )
                
                # Process messages
                if messages:
                    for stream, msg_list in messages:
                        for msg_id, msg_data in msg_list:
                            self._process_message(stream, msg_id, msg_data)
                            
                            # Acknowledge message
                            self.redis_client.xack(stream, self.group_name, msg_id)
                
        except redis.ConnectionError as e:
            logger.error(f"Redis connection error: {e}")
            raise
        except Exception as e:
            logger.error(f"Error consuming streams: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def _process_message(self, stream: str, msg_id: str, msg_data: Dict):
        """
        Process a single message
        
        Args:
            stream: Stream name
            msg_id: Message ID
            msg_data: Message data
        """
        try:
            event_type = msg_data.get('event_type', 'unknown')
            
            logger.debug(f"Processing {event_type} from {stream} (ID: {msg_id})")
            
            # Parse JSON fields if present
            if 'metadata' in msg_data:
                try:
                    msg_data['metadata'] = json.loads(msg_data['metadata'])
                except:
                    pass
            
            if 'data' in msg_data:
                try:
                    msg_data['data'] = json.loads(msg_data['data'])
                except:
                    pass
            
            # Call registered handler
            if event_type in self.handlers:
                self.handlers[event_type](msg_data)
            else:
                logger.warning(f"No handler registered for event type: {event_type}")
        
        except Exception as e:
            logger.error(f"Error processing message {msg_id}: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def start(self, streams: Optional[List[str]] = None):
        """
        Start consuming streams
        
        Args:
            streams: List of streams to consume (None = all default streams)
        """
        if streams is None:
            streams = [
                self.MARKET_DATA_STREAM,
                self.SIGNALS_STREAM,
                self.TRADES_STREAM,
                self.ALERTS_STREAM
            ]
        
        self.running = True
        logger.info(f"Starting consumer '{self.consumer_name}' in group '{self.group_name}'")
        
        try:
            self.consume_multiple_streams(streams)
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self.stop()
    
    def stop(self):
        """Stop consuming"""
        self.running = False
        logger.info("Consumer stopped")
    
    def close(self):
        """Close Redis connection"""
        if self.redis_client:
            self.redis_client.close()
            logger.info("Redis connection closed")


# Example handlers
def handle_market_data(event: Dict):
    """Example handler for market data events"""
    logger.info(f"Market data: {event['symbol']} @ ${event['close']} "
               f"(Volume: {event['volume']})")


def handle_signal(event: Dict):
    """Example handler for trading signals"""
    logger.info(f"Signal: {event['signal_type']} {event['symbol']} @ ${event['price']} "
               f"(Strength: {event['strength']}, Strategy: {event['strategy']})")


def handle_trade(event: Dict):
    """Example handler for trade executions"""
    logger.info(f"Trade: {event['side']} {event['quantity']} {event['symbol']} "
               f"@ ${event['price']} (Status: {event['status']})")


def handle_alert(event: Dict):
    """Example handler for alerts"""
    severity_colors = {
        'INFO': 'blue',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red'
    }
    logger.opt(colors=True).info(
        f"<{severity_colors.get(event['severity'], 'white')}>"
        f"Alert [{event['severity']}]: {event['message']}"
        f"</{severity_colors.get(event['severity'], 'white')}>"
    )


# Example usage and testing
if __name__ == "__main__":
    logger.info("="*70)
    logger.info("Testing Stream Consumer")
    logger.info("="*70)
    
    try:
        # Initialize consumer
        consumer = StreamConsumer(
            consumer_name='test_consumer_1',
            group_name='test_group'
        )
        
        # Register handlers
        consumer.register_handler('market_data', handle_market_data)
        consumer.register_handler('signal', handle_signal)
        consumer.register_handler('trade', handle_trade)
        consumer.register_handler('alert', handle_alert)
        
        logger.info("\n" + "="*70)
        logger.info("Consumer is ready. Generate some events with stream_producer.py")
        logger.info("Press Ctrl+C to stop")
        logger.info("="*70 + "\n")
        
        # Start consuming
        consumer.start()
        
    except redis.ConnectionError:
        logger.error("Could not connect to Redis. Make sure Docker is running:")
        logger.error("  cd docker && docker-compose up -d")
    except KeyboardInterrupt:
        logger.info("\nShutting down...")
    except Exception as e:
        logger.error(f"Consumer failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        if 'consumer' in locals():
            consumer.close()
