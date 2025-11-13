"""
Technical Indicators Calculator
Implements popular technical analysis indicators for trading strategies
"""
import pandas as pd
import numpy as np
from typing import Optional, Tuple
from loguru import logger


class TechnicalIndicators:
    """
    Calculate technical indicators for trading strategies
    All methods work on pandas DataFrames with OHLCV data
    """
    
    @staticmethod
    def sma(df: pd.DataFrame, period: int = 20, column: str = 'close') -> pd.Series:
        """
        Simple Moving Average
        
        Args:
            df: DataFrame with price data
            period: Number of periods
            column: Column to calculate on
            
        Returns:
            Series with SMA values
        """
        return df[column].rolling(window=period).mean()
    
    @staticmethod
    def ema(df: pd.DataFrame, period: int = 20, column: str = 'close') -> pd.Series:
        """
        Exponential Moving Average
        
        Args:
            df: DataFrame with price data
            period: Number of periods
            column: Column to calculate on
            
        Returns:
            Series with EMA values
        """
        return df[column].ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def rsi(df: pd.DataFrame, period: int = 14, column: str = 'close') -> pd.Series:
        """
        Relative Strength Index
        
        Args:
            df: DataFrame with price data
            period: Number of periods (typically 14)
            column: Column to calculate on
            
        Returns:
            Series with RSI values (0-100)
        """
        delta = df[column].diff()
        
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @staticmethod
    def macd(
        df: pd.DataFrame,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
        column: str = 'close'
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Moving Average Convergence Divergence
        
        Args:
            df: DataFrame with price data
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line period
            column: Column to calculate on
            
        Returns:
            Tuple of (MACD line, Signal line, Histogram)
        """
        fast_ema = df[column].ewm(span=fast_period, adjust=False).mean()
        slow_ema = df[column].ewm(span=slow_period, adjust=False).mean()
        
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    @staticmethod
    def bollinger_bands(
        df: pd.DataFrame,
        period: int = 20,
        std_dev: float = 2.0,
        column: str = 'close'
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Bollinger Bands
        
        Args:
            df: DataFrame with price data
            period: Number of periods
            std_dev: Number of standard deviations
            column: Column to calculate on
            
        Returns:
            Tuple of (Upper band, Middle band, Lower band)
        """
        middle_band = df[column].rolling(window=period).mean()
        std = df[column].rolling(window=period).std()
        
        upper_band = middle_band + (std * std_dev)
        lower_band = middle_band - (std * std_dev)
        
        return upper_band, middle_band, lower_band
    
    @staticmethod
    def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Average True Range (volatility indicator)
        
        Args:
            df: DataFrame with OHLC data
            period: Number of periods
            
        Returns:
            Series with ATR values
        """
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr
    
    @staticmethod
    def obv(df: pd.DataFrame) -> pd.Series:
        """
        On-Balance Volume
        
        Args:
            df: DataFrame with price and volume data
            
        Returns:
            Series with OBV values
        """
        obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        return obv
    
    @staticmethod
    def stochastic(
        df: pd.DataFrame,
        k_period: int = 14,
        d_period: int = 3
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Stochastic Oscillator
        
        Args:
            df: DataFrame with OHLC data
            k_period: %K period
            d_period: %D period (signal line)
            
        Returns:
            Tuple of (%K, %D)
        """
        low_min = df['low'].rolling(window=k_period).min()
        high_max = df['high'].rolling(window=k_period).max()
        
        k = 100 * ((df['close'] - low_min) / (high_max - low_min))
        d = k.rolling(window=d_period).mean()
        
        return k, d
    
    @staticmethod
    def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Average Directional Index (trend strength)
        
        Args:
            df: DataFrame with OHLC data
            period: Number of periods
            
        Returns:
            Series with ADX values
        """
        high = df['high']
        low = df['low']
        close = df['close']
        
        # Calculate +DM and -DM
        plus_dm = high.diff()
        minus_dm = -low.diff()
        
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate smoothed +DI and -DI
        atr = tr.rolling(window=period).mean()
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        
        # Calculate DX and ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        
        return adx
    
    @staticmethod
    def vwap(df: pd.DataFrame) -> pd.Series:
        """
        Volume-Weighted Average Price
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Series with VWAP values
        """
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        vwap = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
        return vwap
    
    @staticmethod
    def momentum(df: pd.DataFrame, period: int = 10, column: str = 'close') -> pd.Series:
        """
        Price Momentum
        
        Args:
            df: DataFrame with price data
            period: Number of periods
            column: Column to calculate on
            
        Returns:
            Series with momentum values
        """
        return df[column].diff(period)
    
    @staticmethod
    def roc(df: pd.DataFrame, period: int = 10, column: str = 'close') -> pd.Series:
        """
        Rate of Change
        
        Args:
            df: DataFrame with price data
            period: Number of periods
            column: Column to calculate on
            
        Returns:
            Series with ROC values (percentage)
        """
        return ((df[column] - df[column].shift(period)) / df[column].shift(period)) * 100
    
    @staticmethod
    def williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Williams %R
        
        Args:
            df: DataFrame with OHLC data
            period: Number of periods
            
        Returns:
            Series with Williams %R values (-100 to 0)
        """
        highest_high = df['high'].rolling(window=period).max()
        lowest_low = df['low'].rolling(window=period).min()
        
        wr = -100 * ((highest_high - df['close']) / (highest_high - lowest_low))
        
        return wr
    
    @staticmethod
    def cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
        """
        Commodity Channel Index
        
        Args:
            df: DataFrame with OHLC data
            period: Number of periods
            
        Returns:
            Series with CCI values
        """
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        sma = typical_price.rolling(window=period).mean()
        mad = typical_price.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
        
        cci = (typical_price - sma) / (0.015 * mad)
        
        return cci


# Example usage and testing
if __name__ == "__main__":
    from data_pipeline.ingest.data_fetcher import MarketDataFetcher
    
    logger.info("="*60)
    logger.info("Testing Technical Indicators")
    logger.info("="*60)
    
    # Fetch some data
    fetcher = MarketDataFetcher()
    df = fetcher.fetch_historical_data('AAPL', period='3mo', interval='1d')
    
    if df is not None:
        indicators = TechnicalIndicators()
        
        # Test all indicators
        logger.info("\nCalculating indicators...")
        
        # Moving averages
        df['sma_20'] = indicators.sma(df, period=20)
        df['ema_20'] = indicators.ema(df, period=20)
        logger.success("✓ Moving averages calculated")
        
        # RSI
        df['rsi'] = indicators.rsi(df, period=14)
        logger.success("✓ RSI calculated")
        
        # MACD
        df['macd'], df['macd_signal'], df['macd_hist'] = indicators.macd(df)
        logger.success("✓ MACD calculated")
        
        # Bollinger Bands
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = indicators.bollinger_bands(df)
        logger.success("✓ Bollinger Bands calculated")
        
        # ATR
        df['atr'] = indicators.atr(df)
        logger.success("✓ ATR calculated")
        
        # OBV
        df['obv'] = indicators.obv(df)
        logger.success("✓ OBV calculated")
        
        # Stochastic
        df['stoch_k'], df['stoch_d'] = indicators.stochastic(df)
        logger.success("✓ Stochastic calculated")
        
        # ADX
        df['adx'] = indicators.adx(df)
        logger.success("✓ ADX calculated")
        
        # Display results
        print("\nLatest values:")
        print(df[['close', 'sma_20', 'ema_20', 'rsi', 'macd', 'atr']].tail())
        
        # Check for signals
        latest = df.iloc[-1]
        print(f"\nLatest AAPL Analysis:")
        print(f"  Price: ${latest['close']:.2f}")
        print(f"  RSI: {latest['rsi']:.2f} {'(Oversold)' if latest['rsi'] < 30 else '(Overbought)' if latest['rsi'] > 70 else '(Neutral)'}")
        print(f"  MACD: {latest['macd']:.2f} {'(Bullish)' if latest['macd'] > latest['macd_signal'] else '(Bearish)'}")
        print(f"  Price vs SMA20: {'Above' if latest['close'] > latest['sma_20'] else 'Below'} (${latest['sma_20']:.2f})")
        print(f"  ATR: {latest['atr']:.2f}")
        
        logger.success("\n✓ All indicator tests completed!")
    else:
        logger.error("Failed to fetch data for testing")
