"""
Market Regime Detection
Identify market conditions: trending, ranging, volatile, calm, bullish, bearish
"""
import pandas as pd
import numpy as np
from typing import Dict
from loguru import logger

from data_pipeline.features.technical_indicators import TechnicalIndicators


class MarketRegimeDetector:
    """
    Detect market regimes and conditions
    Helps adapt trading strategies to current market state
    """
    
    def __init__(self):
        self.indicators = TechnicalIndicators()
    
    def detect_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect market regime for each bar
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with regime columns added
        """
        df = df.copy()
        
        # Add necessary indicators if not present
        if 'adx' not in df.columns:
            df['adx'] = self.indicators.adx(df)
        
        if 'atr' not in df.columns:
            df['atr'] = self.indicators.atr(df)
        
        if 'sma_50' not in df.columns:
            df['sma_50'] = self.indicators.sma(df, period=50)
        
        if 'sma_200' not in df.columns:
            df['sma_200'] = self.indicators.sma(df, period=200)
        
        # Trend detection
        df['regime_trend'] = self._detect_trend(df)
        
        # Volatility detection
        df['regime_volatility'] = self._detect_volatility(df)
        
        # Direction detection
        df['regime_direction'] = self._detect_direction(df)
        
        # Combined regime
        df['regime'] = df.apply(self._combine_regimes, axis=1)
        
        logger.info("✓ Market regimes detected")
        
        return df
    
    def _detect_trend(self, df: pd.DataFrame) -> pd.Series:
        """
        Detect if market is trending or ranging
        Uses ADX (Average Directional Index)
        
        Returns:
            Series with values: 'trending', 'ranging'
        """
        conditions = [
            df['adx'] > 25,  # Strong trend
            df['adx'] <= 25  # Weak trend / ranging
        ]
        
        choices = ['trending', 'ranging']
        
        return pd.Series(
            np.select(conditions, choices, default='ranging'),
            index=df.index
        )
    
    def _detect_volatility(self, df: pd.DataFrame) -> pd.Series:
        """
        Detect volatility regime
        Uses ATR (Average True Range) percentile
        
        Returns:
            Series with values: 'high_vol', 'normal_vol', 'low_vol'
        """
        # Calculate ATR percentile over lookback period
        atr_percentile = df['atr'].rolling(window=50).apply(
            lambda x: (x.iloc[-1] / x.max()) * 100 if len(x) > 0 else 50
        )
        
        conditions = [
            atr_percentile > 75,  # High volatility
            atr_percentile < 25   # Low volatility
        ]
        
        choices = ['high_vol', 'low_vol']
        
        return pd.Series(
            np.select(conditions, choices, default='normal_vol'),
            index=df.index
        )
    
    def _detect_direction(self, df: pd.DataFrame) -> pd.Series:
        """
        Detect market direction (bull/bear/neutral)
        Uses price position relative to moving averages
        
        Returns:
            Series with values: 'bullish', 'bearish', 'neutral'
        """
        # Price above both MAs = bullish
        # Price below both MAs = bearish
        # Otherwise = neutral
        
        bullish = (df['close'] > df['sma_50']) & (df['sma_50'] > df['sma_200'])
        bearish = (df['close'] < df['sma_50']) & (df['sma_50'] < df['sma_200'])
        
        conditions = [bullish, bearish]
        choices = ['bullish', 'bearish']
        
        return pd.Series(
            np.select(conditions, choices, default='neutral'),
            index=df.index
        )
    
    def _combine_regimes(self, row: pd.Series) -> str:
        """
        Combine individual regimes into a single label
        
        Returns:
            Combined regime string
        """
        return f"{row['regime_direction']}_{row['regime_trend']}_{row['regime_volatility']}"
    
    def get_regime_stats(self, df: pd.DataFrame) -> Dict:
        """
        Get statistics about regime distribution
        
        Args:
            df: DataFrame with regime columns
            
        Returns:
            Dictionary with regime statistics
        """
        if 'regime' not in df.columns:
            df = self.detect_regime(df)
        
        stats = {
            'total_bars': len(df),
            'regimes': {}
        }
        
        # Count each regime
        regime_counts = df['regime'].value_counts()
        for regime, count in regime_counts.items():
            stats['regimes'][regime] = {
                'count': int(count),
                'percentage': round(count / len(df) * 100, 2)
            }
        
        # Separate stats for each dimension
        stats['trend'] = df['regime_trend'].value_counts().to_dict()
        stats['volatility'] = df['regime_volatility'].value_counts().to_dict()
        stats['direction'] = df['regime_direction'].value_counts().to_dict()
        
        return stats
    
    def get_current_regime(self, df: pd.DataFrame) -> Dict:
        """
        Get current market regime
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Dictionary with current regime information
        """
        if 'regime' not in df.columns:
            df = self.detect_regime(df)
        
        latest = df.iloc[-1]
        
        return {
            'timestamp': latest['timestamp'] if 'timestamp' in latest else None,
            'regime': latest['regime'],
            'trend': latest['regime_trend'],
            'volatility': latest['regime_volatility'],
            'direction': latest['regime_direction'],
            'adx': round(latest['adx'], 2),
            'atr': round(latest['atr'], 2),
            'price': round(latest['close'], 2)
        }
    
    def should_trade(self, df: pd.DataFrame, strategy_type: str = 'trend_following') -> bool:
        """
        Determine if current regime is suitable for trading strategy
        
        Args:
            df: DataFrame with regime data
            strategy_type: Type of strategy ('trend_following', 'mean_reversion', 'any')
            
        Returns:
            Boolean indicating if should trade
        """
        regime = self.get_current_regime(df)
        
        if strategy_type == 'trend_following':
            # Best in trending markets
            return regime['trend'] == 'trending' and regime['volatility'] != 'high_vol'
        
        elif strategy_type == 'mean_reversion':
            # Best in ranging markets
            return regime['trend'] == 'ranging' and regime['volatility'] != 'high_vol'
        
        elif strategy_type == 'volatility':
            # Best in high volatility
            return regime['volatility'] == 'high_vol'
        
        else:  # 'any' or unknown
            # Avoid only extreme high volatility
            return regime['volatility'] != 'high_vol'


# Example usage and testing
if __name__ == "__main__":
    from data_pipeline.ingest.data_fetcher import MarketDataFetcher
    
    logger.info("="*60)
    logger.info("Testing Market Regime Detection")
    logger.info("="*60)
    
    # Fetch data
    fetcher = MarketDataFetcher()
    df = fetcher.fetch_historical_data('AAPL', period='1y', interval='1d')
    
    if df is not None:
        # Detect regimes
        detector = MarketRegimeDetector()
        df_regimes = detector.detect_regime(df)
        
        # Get current regime
        current = detector.get_current_regime(df_regimes)
        
        print("\n" + "="*60)
        print("CURRENT MARKET REGIME")
        print("="*60)
        print(f"Symbol: AAPL")
        print(f"Timestamp: {current['timestamp']}")
        print(f"Price: ${current['price']}")
        print(f"\nRegime: {current['regime']}")
        print(f"  Direction: {current['direction']}")
        print(f"  Trend: {current['trend']}")
        print(f"  Volatility: {current['volatility']}")
        print(f"\nIndicators:")
        print(f"  ADX: {current['adx']}")
        print(f"  ATR: {current['atr']}")
        print("="*60)
        
        # Get regime statistics
        stats = detector.get_regime_stats(df_regimes)
        
        print("\nREGIME DISTRIBUTION")
        print("="*60)
        print(f"Total bars analyzed: {stats['total_bars']}")
        print(f"\nTrend:")
        for regime, count in stats['trend'].items():
            print(f"  {regime}: {count}")
        
        print(f"\nVolatility:")
        for regime, count in stats['volatility'].items():
            print(f"  {regime}: {count}")
        
        print(f"\nDirection:")
        for regime, count in stats['direction'].items():
            print(f"  {regime}: {count}")
        
        print(f"\nTop 5 Combined Regimes:")
        sorted_regimes = sorted(
            stats['regimes'].items(),
            key=lambda x: x[1]['count'],
            reverse=True
        )[:5]
        
        for regime, data in sorted_regimes:
            print(f"  {regime}: {data['count']} bars ({data['percentage']}%)")
        
        # Strategy recommendations
        print("\n" + "="*60)
        print("STRATEGY RECOMMENDATIONS")
        print("="*60)
        
        strategies = ['trend_following', 'mean_reversion', 'volatility']
        for strategy in strategies:
            should_trade = detector.should_trade(df_regimes, strategy)
            status = "✓ TRADE" if should_trade else "✗ WAIT"
            print(f"{strategy.replace('_', ' ').title()}: {status}")
        
        # Display regime changes over time
        print("\n" + "="*60)
        print("RECENT REGIME HISTORY (Last 10 days)")
        print("="*60)
        
        recent = df_regimes[['timestamp', 'close', 'regime', 'regime_trend', 'regime_direction']].tail(10)
        print(recent.to_string(index=False))
        
        logger.success("\n✓ Market regime detection test completed!")
    else:
        logger.error("Failed to fetch data for testing")
