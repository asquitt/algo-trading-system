"""
Advanced Mean Reversion Strategies
Includes statistical arbitrage, pairs trading, and multi-asset mean reversion
"""
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from typing import Tuple, Optional, Dict
from loguru import logger


class MeanReversionAdvanced:
    """
    Advanced mean reversion strategies including statistical tests
    """
    
    def __init__(self):
        pass
    
    def zscore_strategy(
        self,
        df: pd.DataFrame,
        lookback: int = 20,
        entry_threshold: float = 2.0,
        exit_threshold: float = 0.5
    ) -> pd.Series:
        """
        Z-Score Mean Reversion Strategy
        Enter when price deviates significantly from mean
        Exit when price returns closer to mean
        
        Args:
            df: DataFrame with OHLCV data
            lookback: Lookback period for mean/std calculation
            entry_threshold: Z-score threshold for entry (e.g., 2.0 = 2 std devs)
            exit_threshold: Z-score threshold for exit
            
        Returns:
            Series with signals: 1 (buy), -1 (sell), 0 (hold/exit)
        """
        logger.info(f"Generating Z-Score Mean Reversion signals (lookback={lookback})")
        
        # Calculate rolling mean and std
        rolling_mean = df['close'].rolling(lookback).mean()
        rolling_std = df['close'].rolling(lookback).std()
        
        # Calculate z-score
        zscore = (df['close'] - rolling_mean) / rolling_std
        
        # Generate signals
        signals = pd.Series(0, index=df.index)
        
        # Entry signals
        signals[zscore < -entry_threshold] = 1  # Oversold - buy
        signals[zscore > entry_threshold] = -1  # Overbought - sell
        
        # Exit signals (revert to hold when z-score approaches 0)
        signals[(zscore > -exit_threshold) & (zscore < exit_threshold)] = 0
        
        # Forward fill to maintain positions
        signals = signals.replace(0, np.nan).fillna(method='ffill').fillna(0)
        
        return signals
    
    def cointegration_pairs_trading(
        self,
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        lookback: int = 60,
        entry_threshold: float = 2.0,
        exit_threshold: float = 0.5
    ) -> Tuple[pd.Series, Dict]:
        """
        Pairs Trading using Cointegration
        Trade the spread between two cointegrated assets
        
        Args:
            df1: DataFrame for first asset (close prices)
            df2: DataFrame for second asset (close prices)
            lookback: Lookback period for spread calculation
            entry_threshold: Z-score threshold for entry
            exit_threshold: Z-score threshold for exit
            
        Returns:
            Tuple of (signals for asset1, metadata dict)
        """
        logger.info("Generating Cointegration Pairs Trading signals")
        
        # Align the dataframes
        prices1 = df1['close']
        prices2 = df2['close']
        
        # Test for cointegration
        coint_test = stats.coint(prices1, prices2)
        p_value = coint_test[1]
        
        metadata = {
            'cointegration_pvalue': p_value,
            'is_cointegrated': p_value < 0.05
        }
        
        if p_value > 0.05:
            logger.warning(f"Assets may not be cointegrated (p-value: {p_value:.4f})")
        
        # Calculate hedge ratio using linear regression
        model = LinearRegression()
        model.fit(prices2.values.reshape(-1, 1), prices1.values)
        hedge_ratio = model.coef_[0]
        
        metadata['hedge_ratio'] = hedge_ratio
        
        # Calculate spread
        spread = prices1 - hedge_ratio * prices2
        
        # Calculate rolling z-score of spread
        rolling_mean = spread.rolling(lookback).mean()
        rolling_std = spread.rolling(lookback).std()
        zscore = (spread - rolling_mean) / rolling_std
        
        # Generate signals for asset1 (asset2 will be opposite)
        signals = pd.Series(0, index=df1.index)
        
        # Entry signals
        signals[zscore < -entry_threshold] = 1  # Long asset1, short asset2
        signals[zscore > entry_threshold] = -1  # Short asset1, long asset2
        
        # Exit signals
        signals[(zscore > -exit_threshold) & (zscore < exit_threshold)] = 0
        
        # Forward fill positions
        signals = signals.replace(0, np.nan).fillna(method='ffill').fillna(0)
        
        metadata['avg_zscore'] = zscore.abs().mean()
        metadata['max_zscore'] = zscore.abs().max()
        
        logger.info(f"Hedge ratio: {hedge_ratio:.4f}, Cointegration p-value: {p_value:.4f}")
        
        return signals, metadata
    
    def bollinger_mean_reversion(
        self,
        df: pd.DataFrame,
        period: int = 20,
        std_dev: float = 2.0,
        rsi_confirmation: bool = True,
        rsi_oversold: int = 30,
        rsi_overbought: int = 70
    ) -> pd.Series:
        """
        Bollinger Band Mean Reversion with RSI Confirmation
        
        Args:
            df: DataFrame with OHLCV data
            period: Bollinger Band period
            std_dev: Standard deviation multiplier
            rsi_confirmation: Use RSI to confirm signals
            rsi_oversold: RSI oversold threshold
            rsi_overbought: RSI overbought threshold
            
        Returns:
            Series with signals
        """
        logger.info("Generating Bollinger Mean Reversion signals")
        
        # Calculate Bollinger Bands
        rolling_mean = df['close'].rolling(period).mean()
        rolling_std = df['close'].rolling(period).std()
        
        upper = rolling_mean + (std_dev * rolling_std)
        lower = rolling_mean - (std_dev * rolling_std)
        
        signals = pd.Series(0, index=df.index)
        
        if rsi_confirmation and 'rsi' in df.columns:
            # Buy when price touches lower band AND RSI confirms oversold
            signals[(df['close'] < lower) & (df['rsi'] < rsi_oversold)] = 1
            
            # Sell when price touches upper band AND RSI confirms overbought
            signals[(df['close'] > upper) & (df['rsi'] > rsi_overbought)] = -1
            
            # Exit when price returns to middle band
            signals[(df['close'] > rolling_mean * 0.99) & (df['close'] < rolling_mean * 1.01)] = 0
        else:
            # Simple Bollinger Band mean reversion
            signals[df['close'] < lower] = 1
            signals[df['close'] > upper] = -1
            signals[(df['close'] > rolling_mean * 0.99) & (df['close'] < rolling_mean * 1.01)] = 0
        
        # Forward fill positions
        signals = signals.replace(0, np.nan).fillna(method='ffill').fillna(0)
        
        return signals
    
    def channel_breakout_reversion(
        self,
        df: pd.DataFrame,
        channel_period: int = 20,
        breakout_threshold: float = 0.02
    ) -> pd.Series:
        """
        Channel Breakout Mean Reversion
        Fade breakouts that fail to sustain
        
        Args:
            df: DataFrame with OHLCV data
            channel_period: Period for channel calculation
            breakout_threshold: Minimum breakout size (2% = 0.02)
            
        Returns:
            Series with signals
        """
        logger.info("Generating Channel Breakout Reversion signals")
        
        # Calculate channel
        high_channel = df['high'].rolling(channel_period).max()
        low_channel = df['low'].rolling(channel_period).min()
        mid_channel = (high_channel + low_channel) / 2
        
        # Calculate breakout size
        upper_breakout = (df['close'] - high_channel) / high_channel
        lower_breakout = (low_channel - df['close']) / low_channel
        
        signals = pd.Series(0, index=df.index)
        
        # Fade failed breakouts
        # If price breaks above channel but returns, short
        signals[(upper_breakout > breakout_threshold) & (df['close'] < high_channel)] = -1
        
        # If price breaks below channel but returns, long
        signals[(lower_breakout > breakout_threshold) & (df['close'] > low_channel)] = 1
        
        # Exit at mid-channel
        signals[(df['close'] > mid_channel * 0.98) & (df['close'] < mid_channel * 1.02)] = 0
        
        # Forward fill positions
        signals = signals.replace(0, np.nan).fillna(method='ffill').fillna(0)
        
        return signals
    
    def ornstein_uhlenbeck(
        self,
        df: pd.DataFrame,
        lookback: int = 60,
        half_life_threshold: float = 20.0
    ) -> Tuple[pd.Series, Dict]:
        """
        Ornstein-Uhlenbeck Mean Reversion
        Models mean reversion using OU process
        
        Args:
            df: DataFrame with OHLCV data
            lookback: Lookback period
            half_life_threshold: Maximum half-life for trading
            
        Returns:
            Tuple of (signals, metadata)
        """
        logger.info("Generating Ornstein-Uhlenbeck signals")
        
        prices = df['close']
        log_prices = np.log(prices)
        
        # Calculate price differences
        price_diff = log_prices.diff()
        price_lag = log_prices.shift(1)
        
        # Estimate OU parameters using rolling regression
        signals = pd.Series(0, index=df.index)
        half_lives = []
        
        for i in range(lookback, len(df)):
            # Get window
            window_diff = price_diff.iloc[i-lookback:i]
            window_lag = price_lag.iloc[i-lookback:i]
            
            # Remove NaN
            valid = ~(window_diff.isna() | window_lag.isna())
            if valid.sum() < lookback / 2:
                continue
                
            y = window_diff[valid].values
            X = window_lag[valid].values
            
            # Regression
            model = LinearRegression()
            model.fit(X.reshape(-1, 1), y)
            
            theta = -model.coef_[0]  # Mean reversion speed
            
            if theta > 0:
                half_life = np.log(2) / theta
                half_lives.append(half_life)
                
                # Only trade if half-life is reasonable
                if half_life < half_life_threshold:
                    # Calculate expected price
                    expected = np.exp(model.intercept_ / theta)
                    current = prices.iloc[i]
                    
                    # Calculate z-score
                    std = window_diff[valid].std()
                    zscore = (np.log(current / expected)) / std
                    
                    # Generate signal
                    if zscore < -2:
                        signals.iloc[i] = 1
                    elif zscore > 2:
                        signals.iloc[i] = -1
        
        # Forward fill positions
        signals = signals.replace(0, np.nan).fillna(method='ffill').fillna(0)
        
        metadata = {
            'avg_half_life': np.mean(half_lives) if half_lives else None,
            'median_half_life': np.median(half_lives) if half_lives else None
        }
        
        logger.info(f"Average half-life: {metadata['avg_half_life']:.2f} if half_lives else 'N/A'")
        
        return signals, metadata


# Example usage
if __name__ == "__main__":
    from data_pipeline.storage.storage_manager import StorageManager
    from data_pipeline.features.technical_indicators import TechnicalIndicators
    
    # Initialize
    storage = StorageManager()
    indicators = TechnicalIndicators()
    mr = MeanReversionAdvanced()
    
    # Fetch data
    df = storage.fetch_data('AAPL', days=365)
    df = indicators.add_all_indicators(df)
    
    # Test Z-Score strategy
    signals = mr.zscore_strategy(df, lookback=20, entry_threshold=2.0)
    print(f"Z-Score Strategy:")
    print(f"  Buy signals: {(signals == 1).sum()}")
    print(f"  Sell signals: {(signals == -1).sum()}")
    
    # Test Pairs Trading (need two assets)
    df2 = storage.fetch_data('MSFT', days=365)
    signals, metadata = mr.cointegration_pairs_trading(df, df2, lookback=60)
    print(f"\nPairs Trading:")
    print(f"  Cointegrated: {metadata['is_cointegrated']}")
    print(f"  P-value: {metadata['cointegration_pvalue']:.4f}")
    print(f"  Hedge ratio: {metadata['hedge_ratio']:.4f}")
    print(f"  Buy signals: {(signals == 1).sum()}")
    print(f"  Sell signals: {(signals == -1).sum()}")
    
    # Test Bollinger Mean Reversion
    signals = mr.bollinger_mean_reversion(df, period=20, std_dev=2.0, rsi_confirmation=True)
    print(f"\nBollinger Mean Reversion:")
    print(f"  Buy signals: {(signals == 1).sum()}")
    print(f"  Sell signals: {(signals == -1).sum()}")
    
    # Test OU Process
    signals, metadata = mr.ornstein_uhlenbeck(df, lookback=60)
    print(f"\nOrnstein-Uhlenbeck:")
    print(f"  Average half-life: {metadata['avg_half_life']:.2f if metadata['avg_half_life'] else 'N/A'}")
    print(f"  Buy signals: {(signals == 1).sum()}")
    print(f"  Sell signals: {(signals == -1).sum()}")
