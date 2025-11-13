"""
Feature Engineering Pipeline
Automated feature generation for trading strategies
"""
import pandas as pd
import numpy as np
from typing import List, Optional
from loguru import logger

from data_pipeline.features.technical_indicators import TechnicalIndicators


class FeatureEngineer:
    """
    Automated feature engineering for trading data
    Generates technical indicators, statistical features, and derived signals
    """
    
    def __init__(self):
        self.indicators = TechnicalIndicators()
        
    def add_all_features(
        self,
        df: pd.DataFrame,
        include_advanced: bool = True
    ) -> pd.DataFrame:
        """
        Add all features to a DataFrame
        
        Args:
            df: DataFrame with OHLCV data
            include_advanced: Include computationally expensive features
            
        Returns:
            DataFrame with all features added
        """
        logger.info("Generating features...")
        
        df = df.copy()
        
        # Basic features
        df = self.add_price_features(df)
        df = self.add_volume_features(df)
        df = self.add_moving_averages(df)
        df = self.add_momentum_indicators(df)
        df = self.add_volatility_indicators(df)
        df = self.add_trend_indicators(df)
        
        if include_advanced:
            df = self.add_pattern_features(df)
            df = self.add_statistical_features(df)
        
        logger.success(f"✓ Generated {len(df.columns)} total features")
        
        return df
    
    def add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price-based features"""
        df = df.copy()
        
        # Returns
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Price changes
        df['price_change'] = df['close'].diff()
        df['price_change_pct'] = df['close'].pct_change() * 100
        
        # Intraday range
        df['high_low_range'] = df['high'] - df['low']
        df['high_low_pct'] = (df['high'] - df['low']) / df['close'] * 100
        
        # Body and wick (candlestick features)
        df['body'] = abs(df['close'] - df['open'])
        df['body_pct'] = df['body'] / df['close'] * 100
        df['upper_wick'] = df['high'] - df[['close', 'open']].max(axis=1)
        df['lower_wick'] = df[['close', 'open']].min(axis=1) - df['low']
        
        # Gap features
        df['gap'] = df['open'] - df['close'].shift(1)
        df['gap_pct'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1) * 100
        
        logger.info("  ✓ Price features added")
        return df
    
    def add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features"""
        df = df.copy()
        
        # Volume changes
        df['volume_change'] = df['volume'].pct_change()
        df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_20']
        
        # Volume-price correlation
        df['volume_price_corr'] = df['volume'].rolling(window=20).corr(df['close'])
        
        # On-Balance Volume
        df['obv'] = self.indicators.obv(df)
        df['obv_sma'] = df['obv'].rolling(window=20).mean()
        
        # VWAP
        df['vwap'] = self.indicators.vwap(df)
        df['price_vwap_ratio'] = df['close'] / df['vwap']
        
        logger.info("  ✓ Volume features added")
        return df
    
    def add_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add moving average features"""
        df = df.copy()
        
        # Simple moving averages
        for period in [5, 10, 20, 50, 200]:
            df[f'sma_{period}'] = self.indicators.sma(df, period=period)
            df[f'price_sma_{period}_ratio'] = df['close'] / df[f'sma_{period}']
        
        # Exponential moving averages
        for period in [12, 26, 50]:
            df[f'ema_{period}'] = self.indicators.ema(df, period=period)
            df[f'price_ema_{period}_ratio'] = df['close'] / df[f'ema_{period}']
        
        # Moving average crossovers
        df['sma_5_20_cross'] = df['sma_5'] - df['sma_20']
        df['sma_20_50_cross'] = df['sma_20'] - df['sma_50']
        df['ema_12_26_cross'] = df['ema_12'] - df['ema_26']
        
        logger.info("  ✓ Moving averages added")
        return df
    
    def add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum-based indicators"""
        df = df.copy()
        
        # RSI
        df['rsi'] = self.indicators.rsi(df, period=14)
        df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
        df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
        
        # MACD
        df['macd'], df['macd_signal'], df['macd_hist'] = self.indicators.macd(df)
        df['macd_bullish'] = (df['macd'] > df['macd_signal']).astype(int)
        
        # Stochastic
        df['stoch_k'], df['stoch_d'] = self.indicators.stochastic(df)
        df['stoch_oversold'] = (df['stoch_k'] < 20).astype(int)
        df['stoch_overbought'] = (df['stoch_k'] > 80).astype(int)
        
        # Rate of Change
        for period in [5, 10, 20]:
            df[f'roc_{period}'] = self.indicators.roc(df, period=period)
        
        # Momentum
        for period in [5, 10, 20]:
            df[f'momentum_{period}'] = self.indicators.momentum(df, period=period)
        
        # Williams %R
        df['williams_r'] = self.indicators.williams_r(df)
        
        # CCI
        df['cci'] = self.indicators.cci(df)
        
        logger.info("  ✓ Momentum indicators added")
        return df
    
    def add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility indicators"""
        df = df.copy()
        
        # Bollinger Bands
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = self.indicators.bollinger_bands(df)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle'] * 100
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # ATR (Average True Range)
        df['atr'] = self.indicators.atr(df)
        df['atr_pct'] = df['atr'] / df['close'] * 100
        
        # Historical volatility
        for period in [10, 20, 30]:
            df[f'volatility_{period}'] = df['returns'].rolling(window=period).std() * np.sqrt(252) * 100
        
        logger.info("  ✓ Volatility indicators added")
        return df
    
    def add_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trend strength indicators"""
        df = df.copy()
        
        # ADX (Average Directional Index)
        df['adx'] = self.indicators.adx(df)
        df['trending'] = (df['adx'] > 25).astype(int)
        df['strong_trend'] = (df['adx'] > 40).astype(int)
        
        # Linear regression slope
        for period in [10, 20]:
            df[f'lr_slope_{period}'] = df['close'].rolling(window=period).apply(
                lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) == period else np.nan
            )
        
        logger.info("  ✓ Trend indicators added")
        return df
    
    def add_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add candlestick pattern features"""
        df = df.copy()
        
        # Doji patterns
        df['is_doji'] = (abs(df['close'] - df['open']) / df['high_low_range'] < 0.1).astype(int)
        
        # Hammer/Inverted Hammer
        df['is_hammer'] = (
            (df['lower_wick'] > df['body'] * 2) & 
            (df['upper_wick'] < df['body'])
        ).astype(int)
        
        # Shooting Star
        df['is_shooting_star'] = (
            (df['upper_wick'] > df['body'] * 2) & 
            (df['lower_wick'] < df['body'])
        ).astype(int)
        
        # Bullish/Bearish engulfing
        df['is_bullish_engulfing'] = (
            (df['close'] > df['open']) &  # Current is bullish
            (df['close'].shift(1) < df['open'].shift(1)) &  # Previous is bearish
            (df['open'] < df['close'].shift(1)) &  # Opens below previous close
            (df['close'] > df['open'].shift(1))  # Closes above previous open
        ).astype(int)
        
        df['is_bearish_engulfing'] = (
            (df['close'] < df['open']) &  # Current is bearish
            (df['close'].shift(1) > df['open'].shift(1)) &  # Previous is bullish
            (df['open'] > df['close'].shift(1)) &  # Opens above previous close
            (df['close'] < df['open'].shift(1))  # Closes below previous open
        ).astype(int)
        
        logger.info("  ✓ Pattern features added")
        return df
    
    def add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add statistical features"""
        df = df.copy()
        
        # Rolling statistics
        for period in [10, 20]:
            df[f'returns_mean_{period}'] = df['returns'].rolling(window=period).mean()
            df[f'returns_std_{period}'] = df['returns'].rolling(window=period).std()
            df[f'returns_skew_{period}'] = df['returns'].rolling(window=period).skew()
            df[f'returns_kurt_{period}'] = df['returns'].rolling(window=period).kurt()
        
        # Z-score
        for period in [20, 50]:
            mean = df['close'].rolling(window=period).mean()
            std = df['close'].rolling(window=period).std()
            df[f'zscore_{period}'] = (df['close'] - mean) / std
        
        logger.info("  ✓ Statistical features added")
        return df
    
    def get_feature_list(self, df: pd.DataFrame) -> List[str]:
        """Get list of all feature column names"""
        base_cols = ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'vwap', 'interval', 'source']
        feature_cols = [col for col in df.columns if col not in base_cols]
        return feature_cols


# Example usage and testing
if __name__ == "__main__":
    from data_pipeline.ingest.data_fetcher import MarketDataFetcher
    
    logger.info("="*60)
    logger.info("Testing Feature Engineering")
    logger.info("="*60)
    
    # Fetch data
    fetcher = MarketDataFetcher()
    df = fetcher.fetch_historical_data('AAPL', period='6mo', interval='1d')
    
    if df is not None:
        logger.info(f"\nOriginal data: {df.shape}")
        logger.info(f"Columns: {len(df.columns)}")
        
        # Generate all features
        engineer = FeatureEngineer()
        df_features = engineer.add_all_features(df)
        
        logger.info(f"\nWith features: {df_features.shape}")
        logger.info(f"Total columns: {len(df_features.columns)}")
        
        # Get feature list
        feature_cols = engineer.get_feature_list(df_features)
        logger.info(f"Generated features: {len(feature_cols)}")
        
        # Display sample
        print("\nLatest values with features:")
        display_cols = ['close', 'returns', 'rsi', 'macd', 'atr', 'adx', 'bb_position']
        print(df_features[display_cols].tail())
        
        # Check for NaN values
        nan_count = df_features.isnull().sum().sum()
        logger.info(f"\nNaN values: {nan_count}")
        
        # Feature summary
        print("\nFeature Categories:")
        categories = {
            'Price': [c for c in feature_cols if 'price' in c or 'returns' in c or 'gap' in c],
            'Volume': [c for c in feature_cols if 'volume' in c or 'obv' in c or 'vwap' in c],
            'Moving Avg': [c for c in feature_cols if 'sma' in c or 'ema' in c],
            'Momentum': [c for c in feature_cols if 'rsi' in c or 'macd' in c or 'stoch' in c or 'roc' in c or 'momentum' in c],
            'Volatility': [c for c in feature_cols if 'bb' in c or 'atr' in c or 'volatility' in c],
            'Trend': [c for c in feature_cols if 'adx' in c or 'lr_slope' in c or 'trend' in c],
            'Pattern': [c for c in feature_cols if 'is_' in c],
            'Statistical': [c for c in feature_cols if 'zscore' in c or 'skew' in c or 'kurt' in c]
        }
        
        for category, cols in categories.items():
            print(f"  {category}: {len(cols)} features")
        
        logger.success("\n✓ Feature engineering test completed!")
    else:
        logger.error("Failed to fetch data for testing")
