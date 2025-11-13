"""
Data Storage Manager
Handles storing market data into TimescaleDB efficiently
"""
import pandas as pd
from typing import List, Optional
from datetime import datetime
from loguru import logger
from sqlalchemy import and_

from data_pipeline.storage.database import db_manager
from data_pipeline.storage.models import MarketData


class DataStorageManager:
    """Manage storing market data in TimescaleDB"""
    
    def __init__(self):
        """Initialize the storage manager"""
        db_manager.initialize()
    
    def store_dataframe(
        self, 
        df: pd.DataFrame, 
        check_duplicates: bool = True
    ) -> int:
        """
        Store a DataFrame of market data
        
        Args:
            df: DataFrame with columns: timestamp, symbol, open, high, low, close, volume, interval
            check_duplicates: Whether to check for and skip duplicate records
            
        Returns:
            Number of records inserted
        """
        if df is None or df.empty:
            logger.warning("Empty DataFrame provided, nothing to store")
            return 0
        
        try:
            records_to_insert = []
            
            for _, row in df.iterrows():
                # Check if this record already exists (if requested)
                if check_duplicates:
                    exists = self._record_exists(
                        row['symbol'],
                        row['timestamp'],
                        row.get('interval', '1h')
                    )
                    if exists:
                        continue
                
                # Create MarketData object
                record = MarketData(
                    timestamp=row['timestamp'],
                    symbol=row['symbol'],
                    open=float(row['open']),
                    high=float(row['high']),
                    low=float(row['low']),
                    close=float(row['close']),
                    volume=float(row['volume']),
                    vwap=float(row.get('vwap', 0)) if pd.notna(row.get('vwap')) else None,
                    interval=row.get('interval', '1h'),
                    source=row.get('source', 'yfinance'),
                    exchange=row.get('exchange', 'US')
                )
                records_to_insert.append(record)
            
            # Bulk insert
            if records_to_insert:
                with db_manager.get_session() as session:
                    session.bulk_save_objects(records_to_insert)
                    session.commit()
                
                logger.success(f"✓ Stored {len(records_to_insert)} records")
                return len(records_to_insert)
            else:
                logger.info("No new records to insert (all duplicates)")
                return 0
            
        except Exception as e:
            logger.error(f"Error storing data: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return 0
    
    def store_multiple_dataframes(
        self, 
        data_dict: dict,
        check_duplicates: bool = True
    ) -> dict:
        """
        Store multiple DataFrames (one per symbol)
        
        Args:
            data_dict: Dictionary mapping symbol to DataFrame
            check_duplicates: Whether to check for duplicates
            
        Returns:
            Dictionary with insertion statistics
        """
        results = {
            'total_symbols': len(data_dict),
            'successful': 0,
            'failed': 0,
            'total_records': 0
        }
        
        for symbol, df in data_dict.items():
            try:
                count = self.store_dataframe(df, check_duplicates)
                results['total_records'] += count
                results['successful'] += 1
            except Exception as e:
                logger.error(f"Failed to store {symbol}: {e}")
                results['failed'] += 1
        
        logger.info(f"Storage summary: {results['successful']}/{results['total_symbols']} symbols, "
                   f"{results['total_records']} total records")
        
        return results
    
    def _record_exists(self, symbol: str, timestamp: datetime, interval: str) -> bool:
        """
        Check if a record already exists
        
        Args:
            symbol: Stock symbol
            timestamp: Timestamp
            interval: Data interval
            
        Returns:
            True if record exists
        """
        try:
            with db_manager.get_session() as session:
                exists = session.query(MarketData).filter(
                    and_(
                        MarketData.symbol == symbol,
                        MarketData.timestamp == timestamp,
                        MarketData.interval == interval
                    )
                ).first() is not None
                
                return exists
        except Exception as e:
            logger.error(f"Error checking for duplicates: {e}")
            return False
    
    def get_latest_timestamp(self, symbol: str, interval: str) -> Optional[datetime]:
        """
        Get the most recent timestamp for a symbol
        
        Args:
            symbol: Stock symbol
            interval: Data interval
            
        Returns:
            Latest timestamp or None
        """
        try:
            with db_manager.get_session() as session:
                result = session.query(MarketData).filter(
                    and_(
                        MarketData.symbol == symbol,
                        MarketData.interval == interval
                    )
                ).order_by(MarketData.timestamp.desc()).first()
                
                return result.timestamp if result else None
        except Exception as e:
            logger.error(f"Error getting latest timestamp: {e}")
            return None
    
    def get_data_range(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = '1h'
    ) -> Optional[pd.DataFrame]:
        """
        Retrieve data for a specific date range
        
        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date
            interval: Data interval
            
        Returns:
            DataFrame with the data
        """
        try:
            with db_manager.get_session() as session:
                results = session.query(MarketData).filter(
                    and_(
                        MarketData.symbol == symbol,
                        MarketData.interval == interval,
                        MarketData.timestamp >= start_date,
                        MarketData.timestamp <= end_date
                    )
                ).order_by(MarketData.timestamp).all()
                
                if not results:
                    return None
                
                # Convert to DataFrame
                data = [r.to_dict() for r in results]
                df = pd.DataFrame(data)
                
                return df
        except Exception as e:
            logger.error(f"Error retrieving data: {e}")
            return None
    
    def get_statistics(self) -> dict:
        """
        Get database statistics
        
        Returns:
            Dictionary with statistics
        """
        try:
            with db_manager.get_session() as session:
                # Total records
                total_records = session.query(MarketData).count()
                
                # Unique symbols
                unique_symbols = session.query(MarketData.symbol).distinct().count()
                
                # Date range
                oldest = session.query(MarketData).order_by(
                    MarketData.timestamp.asc()
                ).first()
                
                newest = session.query(MarketData).order_by(
                    MarketData.timestamp.desc()
                ).first()
                
                return {
                    'total_records': total_records,
                    'unique_symbols': unique_symbols,
                    'oldest_data': oldest.timestamp if oldest else None,
                    'newest_data': newest.timestamp if newest else None,
                    'date_range_days': (newest.timestamp - oldest.timestamp).days if (oldest and newest) else 0
                }
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {}
    
    def cleanup_old_data(self, days_to_keep: int = 365) -> int:
        """
        Remove data older than specified days
        
        Args:
            days_to_keep: Number of days to keep
            
        Returns:
            Number of records deleted
        """
        try:
            cutoff_date = datetime.utcnow() - pd.Timedelta(days=days_to_keep)
            
            with db_manager.get_session() as session:
                deleted = session.query(MarketData).filter(
                    MarketData.timestamp < cutoff_date
                ).delete()
                
                session.commit()
                
                logger.info(f"Deleted {deleted} records older than {days_to_keep} days")
                return deleted
        except Exception as e:
            logger.error(f"Error cleaning up data: {e}")
            return 0


# Example usage and testing
if __name__ == "__main__":
    from data_pipeline.ingest.data_fetcher import MarketDataFetcher
    
    logger.info("="*60)
    logger.info("Testing Data Storage Manager")
    logger.info("="*60)
    
    # Initialize
    fetcher = MarketDataFetcher()
    storage = DataStorageManager()
    
    # Test 1: Fetch and store data for one symbol
    logger.info("\nTest 1: Fetch and store AAPL data")
    df = fetcher.fetch_historical_data('AAPL', period='5d', interval='1h')
    
    if df is not None:
        count = storage.store_dataframe(df)
        logger.info(f"Stored {count} records for AAPL")
    
    # Test 2: Fetch and store multiple symbols
    logger.info("\nTest 2: Fetch and store multiple symbols")
    symbols = ['MSFT', 'GOOGL', 'AMZN']
    data = fetcher.fetch_multiple_symbols(symbols, period='5d', interval='1d')
    
    results = storage.store_multiple_dataframes(data)
    logger.info(f"Storage results: {results}")
    
    # Test 3: Get statistics
    logger.info("\nTest 3: Database statistics")
    stats = storage.get_statistics()
    print("\nDatabase Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Test 4: Retrieve data
    logger.info("\nTest 4: Retrieve stored data")
    from datetime import timedelta
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=5)
    
    retrieved_df = storage.get_data_range('AAPL', start_date, end_date, '1h')
    if retrieved_df is not None:
        print(f"\nRetrieved {len(retrieved_df)} records for AAPL")
        print(retrieved_df.tail())
    
    logger.success("\n✓ All storage tests completed!")
