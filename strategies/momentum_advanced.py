"""
Advanced Momentum Strategies
Multi-factor momentum, regime-aware momentum, and cross-asset momentum
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from loguru import logger
from sklearn.decomposition import PCA


class MomentumAdvanced:
    """
    Advanced momentum strategies with multiple factors and regime detection
    """
    
    def __init__(self):
        pass
    
    def multi_timeframe_momentum(
        self,
        df: pd.DataFrame,
        short_period: int = 10,
        medium_period: int = 30,
        long_period: int = 60,
        weights: Optional[Tuple[float, float, float]] = None
    ) -> pd.Series:
        """
        Multi-Timeframe Momentum Strategy
        Combines momentum across multiple timeframes with weighting
        
        Args:
            df: DataFrame with OHLCV data
            short_period: Short-term momentum period
            medium_period: Medium-term momentum period
            long_period: Long-term momentum period
            weights: Tuple of (short, medium, long) weights. Default: (0.5, 0.3, 0.2)
            
        Returns:
            Series with signals
        """
        logger.info("Generating Multi-Timeframe Momentum signals")
        
        if weights is None:
            weights = (0.5, 0.3, 0.2)
            
        # Calculate momentum for each timeframe
        mom_short = df['close'].pct_change(short_period)
        mom_medium = df['close'].pct_change(medium_period)
        mom_long = df['close'].pct_change(long_period)
        
        # Weighted average momentum
        composite_momentum = (
            weights[0] * mom_short +
            weights[1] * mom_medium +
            weights[2] * mom_long
        )
        
        # Generate signals
        signals = pd.Series(0, index=df.index)
        
        # Buy when composite momentum is positive
        signals[composite_momentum > 0] = 1
        
        # Sell when composite momentum is negative
        signals[composite_momentum < 0] = -1
        
        return signals
    
    def regime_adaptive_momentum(
        self,
        df: pd.DataFrame,
        volatility_lookback: int = 20,
        low_vol_threshold: float = 0.01,
        high_vol_threshold: float = 0.03,
        momentum_period: int = 20
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Regime-Adaptive Momentum Strategy
        Adjusts momentum strategy based on volatility regime
        
        Args:
            df: DataFrame with OHLCV data
            volatility_lookback: Period for volatility calculation
            low_vol_threshold: Low volatility threshold
            high_vol_threshold: High volatility threshold
            momentum_period: Momentum lookback period
            
        Returns:
            Tuple of (signals, regime)
        """
        logger.info("Generating Regime-Adaptive Momentum signals")
        
        # Calculate returns and volatility
        returns = df['close'].pct_change()
        volatility = returns.rolling(volatility_lookback).std()
        
        # Identify regime
        regime = pd.Series('medium', index=df.index)
        regime[volatility < low_vol_threshold] = 'low_vol'
        regime[volatility > high_vol_threshold] = 'high_vol'
        
        # Calculate momentum
        momentum = df['close'].pct_change(momentum_period)
        
        # Generate signals based on regime
        signals = pd.Series(0, index=df.index)
        
        # Low volatility: Strong momentum signals
        low_vol_mask = regime == 'low_vol'
        signals[low_vol_mask & (momentum > 0.01)] = 1
        signals[low_vol_mask & (momentum < -0.01)] = -1
        
        # High volatility: Weak momentum signals (mean reversion bias)
        high_vol_mask = regime == 'high_vol'
        signals[high_vol_mask & (momentum > 0.05)] = 1
        signals[high_vol_mask & (momentum < -0.05)] = -1
        
        # Medium volatility: Standard momentum
        medium_vol_mask = regime == 'medium'
        signals[medium_vol_mask & (momentum > 0.02)] = 1
        signals[medium_vol_mask & (momentum < -0.02)] = -1
        
        logger.info(f"Regime distribution - Low: {(regime=='low_vol').sum()}, "
                   f"Medium: {(regime=='medium').sum()}, High: {(regime=='high_vol').sum()}")
        
        return signals, regime
    
    def risk_adjusted_momentum(
        self,
        df: pd.DataFrame,
        momentum_period: int = 20,
        volatility_period: int = 20,
        risk_free_rate: float = 0.02
    ) -> pd.Series:
        """
        Risk-Adjusted Momentum (Sharpe Ratio based)
        Trade based on risk-adjusted returns rather than raw returns
        
        Args:
            df: DataFrame with OHLCV data
            momentum_period: Period for momentum calculation
            volatility_period: Period for volatility calculation
            risk_free_rate: Annual risk-free rate (e.g., 0.02 = 2%)
            
        Returns:
            Series with signals
        """
        logger.info("Generating Risk-Adjusted Momentum signals")
        
        # Calculate returns
        returns = df['close'].pct_change()
        
        # Rolling Sharpe Ratio
        rolling_returns = returns.rolling(momentum_period).mean() * 252  # Annualize
        rolling_vol = returns.rolling(volatility_period).std() * np.sqrt(252)  # Annualize
        
        sharpe_ratio = (rolling_returns - risk_free_rate) / rolling_vol
        
        # Generate signals based on Sharpe ratio
        signals = pd.Series(0, index=df.index)
        
        # Buy when Sharpe is positive and significant
        signals[sharpe_ratio > 0.5] = 1
        
        # Sell when Sharpe is negative
        signals[sharpe_ratio < -0.5] = -1
        
        return signals
    
    def volume_confirmed_momentum(
        self,
        df: pd.DataFrame,
        momentum_period: int = 20,
        volume_period: int = 20,
        volume_multiplier: float = 1.5
    ) -> pd.Series:
        """
        Volume-Confirmed Momentum Strategy
        Only take momentum signals when confirmed by above-average volume
        
        Args:
            df: DataFrame with OHLCV data
            momentum_period: Period for momentum
            volume_period: Period for volume average
            volume_multiplier: Minimum volume relative to average
            
        Returns:
            Series with signals
        """
        logger.info("Generating Volume-Confirmed Momentum signals")
        
        # Calculate momentum
        momentum = df['close'].pct_change(momentum_period)
        
        # Calculate volume condition
        avg_volume = df['volume'].rolling(volume_period).mean()
        high_volume = df['volume'] > (avg_volume * volume_multiplier)
        
        # Generate signals (only when volume confirms)
        signals = pd.Series(0, index=df.index)
        
        # Buy signals with volume confirmation
        signals[(momentum > 0.02) & high_volume] = 1
        
        # Sell signals with volume confirmation
        signals[(momentum < -0.02) & high_volume] = -1
        
        return signals
    
    def breakout_momentum(
        self,
        df: pd.DataFrame,
        lookback: int = 20,
        breakout_threshold: float = 1.02,
        volume_confirm: bool = True
    ) -> pd.Series:
        """
        Breakout Momentum Strategy
        Trade breakouts from consolidation with momentum follow-through
        
        Args:
            df: DataFrame with OHLCV data
            lookback: Lookback period for high/low
            breakout_threshold: Threshold above high/below low (1.02 = 2%)
            volume_confirm: Require volume confirmation
            
        Returns:
            Series with signals
        """
        logger.info("Generating Breakout Momentum signals")
        
        # Calculate breakout levels
        high_level = df['high'].rolling(lookback).max()
        low_level = df['low'].rolling(lookback).min()
        
        # Volume confirmation
        if volume_confirm:
            avg_volume = df['volume'].rolling(lookback).mean()
            high_volume = df['volume'] > avg_volume
        else:
            high_volume = pd.Series(True, index=df.index)
        
        # Generate signals
        signals = pd.Series(0, index=df.index)
        
        # Bullish breakout
        bullish_breakout = (df['close'] > high_level * breakout_threshold) & high_volume
        signals[bullish_breakout] = 1
        
        # Bearish breakout
        bearish_breakout = (df['close'] < low_level / breakout_threshold) & high_volume
        signals[bearish_breakout] = -1
        
        # Hold positions until opposite signal
        signals = signals.replace(0, np.nan).fillna(method='ffill').fillna(0)
        
        return signals
    
    def acceleration_momentum(
        self,
        df: pd.DataFrame,
        short_period: int = 5,
        long_period: int = 20
    ) -> pd.Series:
        """
        Acceleration Momentum Strategy
        Trade based on rate of change of momentum (second derivative)
        
        Args:
            df: DataFrame with OHLCV data
            short_period: Short momentum period
            long_period: Long momentum period
            
        Returns:
            Series with signals
        """
        logger.info("Generating Acceleration Momentum signals")
        
        # Calculate momentum
        momentum = df['close'].pct_change(short_period)
        
        # Calculate momentum of momentum (acceleration)
        acceleration = momentum.diff(long_period - short_period)
        
        # Generate signals
        signals = pd.Series(0, index=df.index)
        
        # Buy when momentum is positive AND accelerating
        signals[(momentum > 0) & (acceleration > 0)] = 1
        
        # Sell when momentum is negative AND accelerating down
        signals[(momentum < 0) & (acceleration < 0)] = -1
        
        return signals
    
    def dual_momentum(
        self,
        df: pd.DataFrame,
        absolute_period: int = 12,
        relative_symbols: Optional[list] = None
    ) -> Tuple[pd.Series, Dict]:
        """
        Dual Momentum Strategy (Absolute + Relative)
        Combines absolute momentum (vs cash) with relative momentum (vs peers)
        
        Args:
            df: DataFrame with OHLCV data
            absolute_period: Period for absolute momentum (months)
            relative_symbols: List of symbols for relative comparison
            
        Returns:
            Tuple of (signals, metadata)
        """
        logger.info("Generating Dual Momentum signals")
        
        # Calculate absolute momentum (vs cash/0)
        absolute_momentum = df['close'].pct_change(absolute_period * 21)  # Approx trading days
        
        # Generate basic signal
        signals = pd.Series(0, index=df.index)
        signals[absolute_momentum > 0] = 1  # Positive absolute momentum
        signals[absolute_momentum <= 0] = 0  # Negative - stay in cash
        
        metadata = {
            'avg_absolute_momentum': absolute_momentum.mean(),
            'positive_periods': (absolute_momentum > 0).sum(),
            'negative_periods': (absolute_momentum <= 0).sum()
        }
        
        # TODO: Implement relative momentum when multiple symbols available
        # This would rank symbols and choose top performers
        
        return signals, metadata
    
    def momentum_with_trend_filter(
        self,
        df: pd.DataFrame,
        momentum_period: int = 14,
        trend_period: int = 50,
        momentum_threshold: float = 0.02
    ) -> pd.Series:
        """
        Momentum with Trend Filter
        Only take momentum signals aligned with longer-term trend
        
        Args:
            df: DataFrame with OHLCV data
            momentum_period: Short-term momentum period
            trend_period: Long-term trend period
            momentum_threshold: Minimum momentum for signal
            
        Returns:
            Series with signals
        """
        logger.info("Generating Momentum with Trend Filter signals")
        
        # Calculate short-term momentum
        momentum = df['close'].pct_change(momentum_period)
        
        # Calculate long-term trend (moving average)
        trend_ma = df['close'].rolling(trend_period).mean()
        uptrend = df['close'] > trend_ma
        downtrend = df['close'] < trend_ma
        
        # Generate signals (momentum must align with trend)
        signals = pd.Series(0, index=df.index)
        
        # Buy: Positive momentum in uptrend
        signals[(momentum > momentum_threshold) & uptrend] = 1
        
        # Sell: Negative momentum in downtrend
        signals[(momentum < -momentum_threshold) & downtrend] = -1
        
        return signals


# Example usage
if __name__ == "__main__":
    from data_pipeline.storage.storage_manager import StorageManager
    from data_pipeline.features.technical_indicators import TechnicalIndicators
    
    # Initialize
    storage = StorageManager()
    indicators = TechnicalIndicators()
    momentum = MomentumAdvanced()
    
    # Fetch data
    df = storage.fetch_data('AAPL', days=365)
    df = indicators.add_all_indicators(df)
    
    # Test Multi-Timeframe Momentum
    signals = momentum.multi_timeframe_momentum(df)
    print(f"Multi-Timeframe Momentum:")
    print(f"  Buy signals: {(signals == 1).sum()}")
    print(f"  Sell signals: {(signals == -1).sum()}")
    
    # Test Regime-Adaptive Momentum
    signals, regime = momentum.regime_adaptive_momentum(df)
    print(f"\nRegime-Adaptive Momentum:")
    print(f"  Buy signals: {(signals == 1).sum()}")
    print(f"  Sell signals: {(signals == -1).sum()}")
    print(f"  Low vol periods: {(regime == 'low_vol').sum()}")
    print(f"  High vol periods: {(regime == 'high_vol').sum()}")
    
    # Test Risk-Adjusted Momentum
    signals = momentum.risk_adjusted_momentum(df)
    print(f"\nRisk-Adjusted Momentum:")
    print(f"  Buy signals: {(signals == 1).sum()}")
    print(f"  Sell signals: {(signals == -1).sum()}")
    
    # Test Volume-Confirmed Momentum
    signals = momentum.volume_confirmed_momentum(df)
    print(f"\nVolume-Confirmed Momentum:")
    print(f"  Buy signals: {(signals == 1).sum()}")
    print(f"  Sell signals: {(signals == -1).sum()}")
    
    # Test Breakout Momentum
    signals = momentum.breakout_momentum(df)
    print(f"\nBreakout Momentum:")
    print(f"  Buy signals: {(signals == 1).sum()}")
    print(f"  Sell signals: {(signals == -1).sum()}")
    
    # Test Dual Momentum
    signals, metadata = momentum.dual_momentum(df)
    print(f"\nDual Momentum:")
    print(f"  Buy signals: {(signals == 1).sum()}")
    print(f"  Positive periods: {metadata['positive_periods']}")
    print(f"  Negative periods: {metadata['negative_periods']}")
    
    # Test Momentum with Trend Filter
    signals = momentum.momentum_with_trend_filter(df)
    print(f"\nMomentum with Trend Filter:")
    print(f"  Buy signals: {(signals == 1).sum()}")
    print(f"  Sell signals: {(signals == -1).sum()}")
