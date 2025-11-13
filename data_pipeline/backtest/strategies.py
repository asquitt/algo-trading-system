"""
Strategy Templates
Pre-built trading strategies that you can customize and backtest
"""
import pandas as pd
import numpy as np
from loguru import logger

from data_pipeline.features.technical_indicators import TechnicalIndicators
from data_pipeline.features.feature_engineering import FeatureEngineer


class StrategyTemplates:
    """
    Collection of pre-built trading strategies
    Each strategy returns a signal series: 1 (buy), -1 (sell), 0 (hold)
    """
    
    def __init__(self):
        self.indicators = TechnicalIndicators()
        self.engineer = FeatureEngineer()
    
    def sma_crossover(
        self,
        df: pd.DataFrame,
        fast_period: int = 20,
        slow_period: int = 50
    ) -> pd.Series:
        """
        Simple Moving Average Crossover Strategy
        Buy when fast MA crosses above slow MA
        Sell when fast MA crosses below slow MA
        
        Args:
            df: DataFrame with OHLCV data
            fast_period: Fast MA period
            slow_period: Slow MA period
            
        Returns:
            Series with signals
        """
        logger.info(f"Generating SMA Crossover signals ({fast_period}/{slow_period})")
        
        fast_ma = self.indicators.sma(df, period=fast_period)
        slow_ma = self.indicators.sma(df, period=slow_period)
        
        signals = pd.Series(0, index=df.index)
        signals[fast_ma > slow_ma] = 1
        signals[fast_ma < slow_ma] = -1
        
        return signals
    
    def rsi_mean_reversion(
        self,
        df: pd.DataFrame,
        rsi_period: int = 14,
        oversold: int = 30,
        overbought: int = 70
    ) -> pd.Series:
        """
        RSI Mean Reversion Strategy
        Buy when RSI is oversold
        Sell when RSI is overbought
        
        Args:
            df: DataFrame with OHLCV data
            rsi_period: RSI period
            oversold: Oversold threshold
            overbought: Overbought threshold
            
        Returns:
            Series with signals
        """
        logger.info(f"Generating RSI Mean Reversion signals (RSI={rsi_period})")
        
        rsi = self.indicators.rsi(df, period=rsi_period)
        
        signals = pd.Series(0, index=df.index)
        signals[rsi < oversold] = 1   # Buy oversold
        signals[rsi > overbought] = -1  # Sell overbought
        
        return signals
    
    def macd_trend_following(
        self,
        df: pd.DataFrame,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> pd.Series:
        """
        MACD Trend Following Strategy
        Buy when MACD crosses above signal line
        Sell when MACD crosses below signal line
        
        Args:
            df: DataFrame with OHLCV data
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line period
            
        Returns:
            Series with signals
        """
        logger.info("Generating MACD Trend Following signals")
        
        macd, macd_signal, _ = self.indicators.macd(df, fast, slow, signal)
        
        signals = pd.Series(0, index=df.index)
        signals[macd > macd_signal] = 1
        signals[macd < macd_signal] = -1
        
        return signals
    
    def bollinger_breakout(
        self,
        df: pd.DataFrame,
        period: int = 20,
        std_dev: float = 2.0
    ) -> pd.Series:
        """
        Bollinger Bands Breakout Strategy
        Buy when price breaks above upper band
        Sell when price breaks below lower band
        
        Args:
            df: DataFrame with OHLCV data
            period: Bollinger Bands period
            std_dev: Standard deviations
            
        Returns:
            Series with signals
        """
        logger.info("Generating Bollinger Breakout signals")
        
        upper, middle, lower = self.indicators.bollinger_bands(df, period, std_dev)
        
        signals = pd.Series(0, index=df.index)
        signals[df['close'] > upper] = 1
        signals[df['close'] < lower] = -1
        
        return signals
    
    def triple_ema(
        self,
        df: pd.DataFrame,
        fast: int = 5,
        medium: int = 10,
        slow: int = 20
    ) -> pd.Series:
        """
        Triple EMA Strategy
        Buy when fast > medium > slow (all aligned upward)
        Sell when fast < medium < slow (all aligned downward)
        
        Args:
            df: DataFrame with OHLCV data
            fast: Fast EMA period
            medium: Medium EMA period
            slow: Slow EMA period
            
        Returns:
            Series with signals
        """
        logger.info("Generating Triple EMA signals")
        
        ema_fast = self.indicators.ema(df, period=fast)
        ema_medium = self.indicators.ema(df, period=medium)
        ema_slow = self.indicators.ema(df, period=slow)
        
        signals = pd.Series(0, index=df.index)
        
        # Bullish alignment
        bullish = (ema_fast > ema_medium) & (ema_medium > ema_slow)
        signals[bullish] = 1
        
        # Bearish alignment
        bearish = (ema_fast < ema_medium) & (ema_medium < ema_slow)
        signals[bearish] = -1
        
        return signals
    
    def momentum_strategy(
        self,
        df: pd.DataFrame,
        lookback: int = 20,
        threshold: float = 0.02
    ) -> pd.Series:
        """
        Momentum Strategy
        Buy when price momentum is positive and strong
        Sell when price momentum is negative and strong
        
        Args:
            df: DataFrame with OHLCV data
            lookback: Lookback period for momentum
            threshold: Minimum momentum to trigger signal (2% = 0.02)
            
        Returns:
            Series with signals
        """
        logger.info(f"Generating Momentum signals (lookback={lookback})")
        
        momentum = df['close'].pct_change(lookback)
        
        signals = pd.Series(0, index=df.index)
        signals[momentum > threshold] = 1
        signals[momentum < -threshold] = -1
        
        return signals
    
    def adx_trending_rsi(
        self,
        df: pd.DataFrame,
        adx_threshold: int = 25,
        rsi_oversold: int = 40,
        rsi_overbought: int = 60
    ) -> pd.Series:
        """
        Combined ADX + RSI Strategy
        Only trade when ADX shows trending market
        Use RSI for entry timing
        
        Args:
            df: DataFrame with OHLCV data
            adx_threshold: Minimum ADX for trending
            rsi_oversold: RSI oversold level
            rsi_overbought: RSI overbought level
            
        Returns:
            Series with signals
        """
        logger.info("Generating ADX + RSI combined signals")
        
        adx = self.indicators.adx(df)
        rsi = self.indicators.rsi(df)
        
        signals = pd.Series(0, index=df.index)
        
        # Only trade in trending markets
        trending = adx > adx_threshold
        
        signals[(rsi < rsi_oversold) & trending] = 1
        signals[(rsi > rsi_overbought) & trending] = -1
        
        return signals
    
    def multi_timeframe_trend(
        self,
        df: pd.DataFrame,
        short_period: int = 20,
        medium_period: int = 50,
        long_period: int = 200
    ) -> pd.Series:
        """
        Multi-Timeframe Trend Strategy
        Buy when price > all moving averages (strong uptrend)
        Sell when price < all moving averages (strong downtrend)
        
        Args:
            df: DataFrame with OHLCV data
            short_period: Short-term MA
            medium_period: Medium-term MA
            long_period: Long-term MA
            
        Returns:
            Series with signals
        """
        logger.info("Generating Multi-Timeframe Trend signals")
        
        sma_short = self.indicators.sma(df, period=short_period)
        sma_medium = self.indicators.sma(df, period=medium_period)
        sma_long = self.indicators.sma(df, period=long_period)
        
        signals = pd.Series(0, index=df.index)
        
        # Strong uptrend: price above all MAs
        strong_uptrend = (
            (df['close'] > sma_short) &
            (df['close'] > sma_medium) &
            (df['close'] > sma_long)
        )
        signals[strong_uptrend] = 1
        
        # Strong downtrend: price below all MAs
        strong_downtrend = (
            (df['close'] < sma_short) &
            (df['close'] < sma_medium) &
            (df['close'] < sma_long)
        )
        signals[strong_downtrend] = -1
        
        return signals
    
    def volatility_breakout(
        self,
        df: pd.DataFrame,
        atr_period: int = 14,
        multiplier: float = 2.0
    ) -> pd.Series:
        """
        Volatility Breakout Strategy
        Buy on breakout above volatility-adjusted level
        Sell on breakdown below volatility-adjusted level
        
        Args:
            df: DataFrame with OHLCV data
            atr_period: ATR period
            multiplier: ATR multiplier for breakout threshold
            
        Returns:
            Series with signals
        """
        logger.info("Generating Volatility Breakout signals")
        
        atr = self.indicators.atr(df, period=atr_period)
        sma = self.indicators.sma(df, period=20)
        
        upper_threshold = sma + (atr * multiplier)
        lower_threshold = sma - (atr * multiplier)
        
        signals = pd.Series(0, index=df.index)
        signals[df['close'] > upper_threshold] = 1
        signals[df['close'] < lower_threshold] = -1
        
        return signals


# Example usage and testing
if __name__ == "__main__":
    from data_pipeline.ingest.data_fetcher import MarketDataFetcher
    from data_pipeline.backtest.backtester import Backtester
    
    logger.info("="*70)
    logger.info("Testing Strategy Templates")
    logger.info("="*70)
    
    # Fetch data
    fetcher = MarketDataFetcher()
    df = fetcher.fetch_historical_data('AAPL', period='2y', interval='1d')
    
    if df is not None:
        strategies_obj = StrategyTemplates()
        backtester = Backtester(initial_capital=100000)
        
        # Test multiple strategies
        strategies = [
            ('SMA Crossover', lambda: strategies_obj.sma_crossover(df, 20, 50)),
            ('RSI Mean Reversion', lambda: strategies_obj.rsi_mean_reversion(df)),
            ('MACD Trend', lambda: strategies_obj.macd_trend_following(df)),
            ('Bollinger Breakout', lambda: strategies_obj.bollinger_breakout(df)),
            ('Triple EMA', lambda: strategies_obj.triple_ema(df)),
            ('Momentum', lambda: strategies_obj.momentum_strategy(df)),
            ('ADX + RSI', lambda: strategies_obj.adx_trending_rsi(df)),
            ('Multi-Timeframe', lambda: strategies_obj.multi_timeframe_trend(df)),
        ]
        
        results_summary = []
        
        for strategy_name, strategy_func in strategies:
            print(f"\n{'='*70}")
            print(f"Testing: {strategy_name}")
            print('='*70)
            
            # Generate signals
            signals = strategy_func()
            
            # Run backtest
            results = backtester.run(df, signals)
            
            # Store summary
            results_summary.append({
                'Strategy': strategy_name,
                'Total Return': f"{results['total_return_pct']:.2f}%",
                'Sharpe Ratio': f"{results['sharpe_ratio']:.2f}",
                'Max Drawdown': f"{results['max_drawdown_pct']:.2f}%",
                'Win Rate': f"{results['win_rate_pct']:.2f}%",
                'Total Trades': results['total_trades']
            })
        
        # Print comparison
        print("\n" + "="*70)
        print("STRATEGY COMPARISON")
        print("="*70)
        
        summary_df = pd.DataFrame(results_summary)
        print(summary_df.to_string(index=False))
        
        logger.success("\n✓ Strategy testing completed!")
    else:
        logger.error("Failed to fetch data for testing")
