"""
Market Data Fetcher using yfinance (FREE!)
Fetches OHLCV data for stocks and crypto
"""
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from loguru import logger
from typing import List, Dict, Optional
import time


class MarketDataFetcher:
    """
    Fetch market data from free sources
    Primary: yfinance (Yahoo Finance) - unlimited free access
    """
    
    def __init__(self, rate_limit_delay: float = 0.5):
        """
        Initialize the data fetcher
        
        Args:
            rate_limit_delay: Delay between requests in seconds (be nice to APIs)
        """
        self.rate_limit_delay = rate_limit_delay
        self.last_request_time = 0
        
    def _rate_limit(self):
        """Enforce rate limiting between requests"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self.last_request_time = time.time()
    
    def fetch_historical_data(
        self, 
        symbol: str, 
        period: str = "1mo", 
        interval: str = "1h"
    ) -> Optional[pd.DataFrame]:
        """
        Fetch historical OHLCV data using yfinance (FREE!)
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL', 'MSFT', 'BTC-USD')
            period: Data period - Valid: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, max
            interval: Data interval - Valid: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo
            
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume, symbol
        """
        try:
            self._rate_limit()
            
            logger.info(f"Fetching {symbol}: period={period}, interval={interval}")
            
            # Fetch data from Yahoo Finance
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)
            
            if df.empty:
                logger.warning(f"No data retrieved for {symbol}")
                return None
            
            # Clean and standardize the data
            df = self._clean_data(df, symbol, interval)
            
            logger.success(f"✓ Retrieved {len(df)} bars for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None
    
    def fetch_date_range(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str = "1h"
    ) -> Optional[pd.DataFrame]:
        """
        Fetch data for a specific date range
        
        Args:
            symbol: Stock symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            interval: Data interval
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            self._rate_limit()
            
            logger.info(f"Fetching {symbol}: {start_date} to {end_date}, interval={interval}")
            
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, interval=interval)
            
            if df.empty:
                logger.warning(f"No data retrieved for {symbol}")
                return None
            
            df = self._clean_data(df, symbol, interval)
            
            logger.success(f"✓ Retrieved {len(df)} bars for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None
    
    def fetch_latest_price(self, symbol: str) -> Optional[Dict]:
        """
        Fetch the latest price for a symbol
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with latest price info
        """
        try:
            self._rate_limit()
            
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            if not info:
                return None
            
            return {
                'symbol': symbol,
                'price': info.get('currentPrice') or info.get('regularMarketPrice'),
                'volume': info.get('volume'),
                'market_cap': info.get('marketCap'),
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error fetching latest price for {symbol}: {e}")
            return None
    
    def fetch_multiple_symbols(
        self, 
        symbols: List[str], 
        period: str = "1mo",
        interval: str = "1h"
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple symbols
        
        Args:
            symbols: List of stock symbols
            period: Data period
            interval: Data interval
            
        Returns:
            Dictionary mapping symbol to DataFrame
        """
        data = {}
        total = len(symbols)
        
        logger.info(f"Fetching data for {total} symbols...")
        
        for idx, symbol in enumerate(symbols, 1):
            logger.info(f"[{idx}/{total}] Fetching {symbol}...")
            
            df = self.fetch_historical_data(symbol, period, interval)
            
            if df is not None and not df.empty:
                data[symbol] = df
            else:
                logger.warning(f"Skipping {symbol} - no data")
            
            # Progress indicator
            if idx % 5 == 0:
                logger.info(f"Progress: {idx}/{total} symbols completed")
        
        logger.success(f"✓ Successfully fetched data for {len(data)}/{total} symbols")
        return data
    
    def _clean_data(self, df: pd.DataFrame, symbol: str, interval: str) -> pd.DataFrame:
        """
        Clean and standardize the DataFrame
        
        Args:
            df: Raw DataFrame from yfinance
            symbol: Stock symbol
            interval: Data interval
            
        Returns:
            Cleaned DataFrame
        """
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Standardize column names (yfinance uses capitalized names)
        df.columns = [col.lower() for col in df.columns]
        
        # Reset index to make the date/datetime a column
        df.reset_index(inplace=True)
        
        # The index column might be named 'Date' or 'Datetime' depending on the interval
        # Rename it to 'timestamp'
        index_col = df.columns[0]  # First column after reset_index
        if index_col.lower() in ['date', 'datetime', 'index']:
            df.rename(columns={index_col: 'timestamp'}, inplace=True)
        
        # Ensure we have a timestamp column
        if 'timestamp' not in df.columns:
            logger.error(f"No timestamp column found. Columns: {df.columns.tolist()}")
            raise ValueError("No timestamp column in data")
        
        # Add symbol and interval columns
        df['symbol'] = symbol
        df['interval'] = interval
        df['source'] = 'yfinance'
        
        # Ensure timestamp is timezone-aware (UTC)
        if df['timestamp'].dt.tz is None:
            df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize('UTC')
        else:
            df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_convert('UTC')
        
        # Remove rows with missing critical data
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        df = df.dropna(subset=required_cols)
        
        # Remove rows with zero volume (usually bad data)
        df = df[df['volume'] > 0]
        
        # Calculate VWAP if we have the data
        if 'volume' in df.columns and len(df) > 0:
            # Simple VWAP approximation: (high + low + close) / 3
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            df['vwap'] = typical_price
        
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        # Select only the columns we need
        columns_to_keep = [
            'timestamp', 'symbol', 'open', 'high', 'low', 'close', 
            'volume', 'vwap', 'interval', 'source'
        ]
        
        # Only keep columns that exist
        columns_to_keep = [col for col in columns_to_keep if col in df.columns]
        df = df[columns_to_keep]
        
        return df
    
    def get_symbol_info(self, symbol: str) -> Optional[Dict]:
        """
        Get detailed information about a symbol
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with symbol information
        """
        try:
            self._rate_limit()
            
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            if not info:
                return None
            
            return {
                'symbol': symbol,
                'name': info.get('longName') or info.get('shortName'),
                'sector': info.get('sector'),
                'industry': info.get('industry'),
                'market_cap': info.get('marketCap'),
                'currency': info.get('currency'),
                'exchange': info.get('exchange'),
                'description': info.get('longBusinessSummary', '')[:200]  # First 200 chars
            }
            
        except Exception as e:
            logger.error(f"Error fetching info for {symbol}: {e}")
            return None


# Example usage and testing
if __name__ == "__main__":
    fetcher = MarketDataFetcher()
    
    # Test 1: Single symbol
    logger.info("\n" + "="*60)
    logger.info("TEST 1: Fetch single symbol")
    logger.info("="*60)
    
    df = fetcher.fetch_historical_data('AAPL', period='5d', interval='1h')
    if df is not None:
        print(f"\nShape: {df.shape}")
        print(f"\nColumns: {df.columns.tolist()}")
        print(f"\nFirst few rows:")
        print(df.head())
        print(f"\nLast few rows:")
        print(df.tail())
    
    # Test 2: Multiple symbols
    logger.info("\n" + "="*60)
    logger.info("TEST 2: Fetch multiple symbols")
    logger.info("="*60)
    
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    data = fetcher.fetch_multiple_symbols(symbols, period='5d', interval='1d')
    
    for symbol, df in data.items():
        print(f"\n{symbol}: {len(df)} bars")
        print(df.tail(3))
    
    # Test 3: Crypto
    logger.info("\n" + "="*60)
    logger.info("TEST 3: Fetch crypto")
    logger.info("="*60)
    
    btc_df = fetcher.fetch_historical_data('BTC-USD', period='7d', interval='1h')
    if btc_df is not None:
        print(f"\nBTC-USD: {len(btc_df)} bars")
        print(btc_df.tail())
    
    # Test 4: Latest price
    logger.info("\n" + "="*60)
    logger.info("TEST 4: Latest price")
    logger.info("="*60)
    
    price = fetcher.fetch_latest_price('AAPL')
    if price:
        print(f"\nLatest AAPL: ${price['price']}")
    
    logger.success("\n✓ All fetcher tests completed!")
