"""
Strategy Comparison Tool
Test and compare multiple trading strategies side-by-side
Compatible with your actual StorageManager methods
"""
import pandas as pd
import numpy as np
from typing import Dict
from loguru import logger
import sys
import os
from datetime import datetime, timedelta

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from data_pipeline.storage.storage_manager import DataStorageManager as StorageManager
    from data_pipeline.features.technical_indicators import TechnicalIndicators
    from data_pipeline.backtest.backtester import Backtester
except ImportError as e:
    logger.error(f"Import error: {e}")
    sys.exit(1)

from ml_strategies import RandomForestStrategy, GradientBoostingStrategy, EnsembleStrategy
from mean_reversion_advanced import MeanReversionAdvanced
from momentum_advanced import MomentumAdvanced


def add_all_indicators(df: pd.DataFrame, indicators: TechnicalIndicators) -> pd.DataFrame:
    """
    Add all technical indicators to dataframe using individual methods
    
    Args:
        df: DataFrame with OHLCV data
        indicators: TechnicalIndicators instance
        
    Returns:
        DataFrame with all indicators added
    """
    from loguru import logger
    logger.info("Calculating technical indicators...")
    
    df = df.copy()
    
    # Moving averages
    df['sma_20'] = indicators.sma(df, period=20)
    df['sma_50'] = indicators.sma(df, period=50)
    df['sma_200'] = indicators.sma(df, period=200)
    df['ema_12'] = indicators.ema(df, period=12)
    df['ema_26'] = indicators.ema(df, period=26)
    
    # RSI
    df['rsi'] = indicators.rsi(df, period=14)
    
    # MACD
    macd_result = indicators.macd(df)
    if isinstance(macd_result, tuple) and len(macd_result) == 3:
        df['macd'], df['macd_signal'], df['macd_hist'] = macd_result
    else:
        df['macd'] = macd_result
        df['macd_signal'] = 0
        df['macd_hist'] = 0
    
    # Bollinger Bands
    bb_result = indicators.bollinger_bands(df)
    if isinstance(bb_result, tuple) and len(bb_result) == 3:
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = bb_result
    else:
        df['bb_upper'] = df['close'] * 1.02
        df['bb_middle'] = df['close']
        df['bb_lower'] = df['close'] * 0.98
    
    # ATR
    df['atr'] = indicators.atr(df, period=14)
    
    # ADX
    df['adx'] = indicators.adx(df, period=14)
    
    # Stochastic
    stoch_result = indicators.stochastic(df)
    if isinstance(stoch_result, tuple) and len(stoch_result) == 2:
        df['stoch_k'], df['stoch_d'] = stoch_result
    else:
        df['stoch_k'] = stoch_result
        df['stoch_d'] = stoch_result
    
    # CCI
    df['cci'] = indicators.cci(df, period=20)
    
    # Williams %R
    df['williams_r'] = indicators.williams_r(df, period=14)
    
    # Momentum
    df['momentum'] = indicators.momentum(df, period=10)
    
    # ROC
    df['roc'] = indicators.roc(df, period=12)
    
    # OBV
    df['obv'] = indicators.obv(df)
    
    # VWAP
    df['vwap'] = indicators.vwap(df)
    
    return df


class StrategyComparison:
    """
    Compare performance of multiple trading strategies
    """
    
    def __init__(
        self,
        initial_capital: float = 100000,
        commission: float = 0.001,
        slippage: float = 0.0005
    ):
        """
        Initialize comparison tool
        
        Args:
            initial_capital: Starting capital
            commission: Commission per trade (0.001 = 0.1%)
            slippage: Slippage per trade (0.0005 = 0.05%)
        """
        self.storage = StorageManager()
        self.indicators = TechnicalIndicators()
        self.backtester = Backtester(
            initial_capital=initial_capital,
            commission=commission,
            slippage=slippage
        )
        
        # Initialize strategy classes
        self.mr = MeanReversionAdvanced()
        self.momentum = MomentumAdvanced()
        
    def get_data_with_indicators(self, symbol: str, days: int = 365) -> pd.DataFrame:
        """
        Fetch data and add all indicators
        
        Args:
            symbol: Stock symbol
            days: Days of historical data
            
        Returns:
            DataFrame with OHLCV and indicators
        """
        logger.info(f"Fetching data for {symbol}...")
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Use get_data_range method
        df = self.storage.get_data_range(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date
        )
        
        if df is None or df.empty:
            raise ValueError(f"No data found for {symbol}. Try: python3 cli.py fetch {symbol} --period {days}d --interval 1d")
        
        df = add_all_indicators(df, self.indicators)
        
        return df
    
    def test_all_strategies(self, df: pd.DataFrame) -> Dict:
        """
        Test all available strategies
        
        Args:
            df: DataFrame with OHLCV and indicators
            
        Returns:
            Dictionary of strategy results
        """
        logger.info("Testing all strategies...")
        
        results = {}
        
        # Mean Reversion Strategies
        logger.info("\n=== Mean Reversion Strategies ===")
        
        try:
            signals = self.mr.zscore_strategy(df, lookback=20, entry_threshold=2.0)
            result = self.backtester.run(df, signals)
            results['Z-Score MR'] = result
            logger.success(f"✓ Z-Score MR - Return: {result['total_return']:.2%}, Sharpe: {result['sharpe_ratio']:.2f}")
        except Exception as e:
            logger.error(f"✗ Z-Score MR failed: {e}")
        
        try:
            signals = self.mr.bollinger_mean_reversion(df, period=20, std_dev=2.0, rsi_confirmation=True)
            result = self.backtester.run(df, signals)
            results['Bollinger MR'] = result
            logger.success(f"✓ Bollinger MR - Return: {result['total_return']:.2%}, Sharpe: {result['sharpe_ratio']:.2f}")
        except Exception as e:
            logger.error(f"✗ Bollinger MR failed: {e}")
        
        try:
            signals = self.mr.channel_breakout_reversion(df, channel_period=20)
            result = self.backtester.run(df, signals)
            results['Channel Reversion'] = result
            logger.success(f"✓ Channel Reversion - Return: {result['total_return']:.2%}, Sharpe: {result['sharpe_ratio']:.2f}")
        except Exception as e:
            logger.error(f"✗ Channel Reversion failed: {e}")
        
        # Momentum Strategies
        logger.info("\n=== Momentum Strategies ===")
        
        try:
            signals = self.momentum.multi_timeframe_momentum(df, short_period=10, medium_period=30, long_period=60)
            result = self.backtester.run(df, signals)
            results['Multi-TF Momentum'] = result
            logger.success(f"✓ Multi-TF Momentum - Return: {result['total_return']:.2%}, Sharpe: {result['sharpe_ratio']:.2f}")
        except Exception as e:
            logger.error(f"✗ Multi-TF Momentum failed: {e}")
        
        try:
            signals, _ = self.momentum.regime_adaptive_momentum(df)
            result = self.backtester.run(df, signals)
            results['Regime Momentum'] = result
            logger.success(f"✓ Regime Momentum - Return: {result['total_return']:.2%}, Sharpe: {result['sharpe_ratio']:.2f}")
        except Exception as e:
            logger.error(f"✗ Regime Momentum failed: {e}")
        
        try:
            signals = self.momentum.volume_confirmed_momentum(df, momentum_period=20)
            result = self.backtester.run(df, signals)
            results['Volume Momentum'] = result
            logger.success(f"✓ Volume Momentum - Return: {result['total_return']:.2%}, Sharpe: {result['sharpe_ratio']:.2f}")
        except Exception as e:
            logger.error(f"✗ Volume Momentum failed: {e}")
        
        try:
            signals = self.momentum.breakout_momentum(df, lookback=20)
            result = self.backtester.run(df, signals)
            results['Breakout Momentum'] = result
            logger.success(f"✓ Breakout Momentum - Return: {result['total_return']:.2%}, Sharpe: {result['sharpe_ratio']:.2f}")
        except Exception as e:
            logger.error(f"✗ Breakout Momentum failed: {e}")
        
        try:
            signals = self.momentum.momentum_with_trend_filter(df)
            result = self.backtester.run(df, signals)
            results['Trend-Filtered Momentum'] = result
            logger.success(f"✓ Trend-Filtered Momentum - Return: {result['total_return']:.2%}, Sharpe: {result['sharpe_ratio']:.2f}")
        except Exception as e:
            logger.error(f"✗ Trend-Filtered Momentum failed: {e}")
        
        return results
    
    def test_ml_strategies(self, df: pd.DataFrame, train_size: float = 0.8) -> Dict:
        """
        Test ML-based strategies
        
        Args:
            df: DataFrame with OHLCV and indicators
            train_size: Proportion of data for training
            
        Returns:
            Dictionary of ML strategy results
        """
        logger.info("\n=== ML Strategies ===")
        
        results = {}
        
        # Split data for training/testing
        split_idx = int(len(df) * train_size)
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()
        
        # Random Forest
        try:
            logger.info("Training Random Forest...")
            rf = RandomForestStrategy(n_estimators=100, max_depth=8)
            metrics = rf.train(train_df, horizon=1, threshold=0.002, test_size=0.2)
            
            # Test on out-of-sample data
            signals = rf.predict(test_df)
            result = self.backtester.run(test_df, signals)
            results['Random Forest'] = result
            
            logger.success(f"✓ Random Forest - Accuracy: {metrics['test_accuracy']:.3f}, "
                         f"Return: {result['total_return']:.2%}, Sharpe: {result['sharpe_ratio']:.2f}")
        except Exception as e:
            logger.error(f"✗ Random Forest failed: {e}")
        
        # Gradient Boosting
        try:
            logger.info("Training Gradient Boosting...")
            gb = GradientBoostingStrategy(n_estimators=100, learning_rate=0.1, max_depth=5)
            metrics = gb.train(train_df, horizon=1, threshold=0.002, test_size=0.2)
            
            signals = gb.predict(test_df)
            result = self.backtester.run(test_df, signals)
            results['Gradient Boosting'] = result
            
            logger.success(f"✓ Gradient Boosting - Accuracy: {metrics['test_accuracy']:.3f}, "
                         f"Return: {result['total_return']:.2%}, Sharpe: {result['sharpe_ratio']:.2f}")
        except Exception as e:
            logger.error(f"✗ Gradient Boosting failed: {e}")
        
        # Ensemble (if both models trained successfully)
        try:
            if 'Random Forest' in results and 'Gradient Boosting' in results:
                logger.info("Creating Ensemble...")
                ensemble = EnsembleStrategy([rf, gb])
                signals, confidence = ensemble.predict_with_confidence(test_df)
                
                # Filter by confidence
                signals_filtered = signals.copy()
                signals_filtered[confidence < 0.6] = 0  # 60% confidence threshold
                
                result = self.backtester.run(test_df, signals_filtered)
                results['Ensemble (60% conf)'] = result
                
                logger.success(f"✓ Ensemble - Return: {result['total_return']:.2%}, Sharpe: {result['sharpe_ratio']:.2f}")
        except Exception as e:
            logger.error(f"✗ Ensemble failed: {e}")
        
        return results
    
    def create_comparison_table(self, results: Dict) -> pd.DataFrame:
        """
        Create comparison table from results
        
        Args:
            results: Dictionary of strategy results
            
        Returns:
            DataFrame with comparison metrics
        """
        comparison_data = []
        
        for strategy_name, result in results.items():
            comparison_data.append({
                'Strategy': strategy_name,
                'Total Return': f"{result['total_return']:.2%}",
                'Sharpe Ratio': f"{result['sharpe_ratio']:.2f}",
                'Max Drawdown': f"{result['max_drawdown']:.2%}",
                'Win Rate': f"{result['win_rate']:.2%}",
                'Total Trades': result['total_trades'],
                'Avg Win': f"{result.get('avg_win', 0):.2%}",
                'Avg Loss': f"{result.get('avg_loss', 0):.2%}",
            })
        
        df = pd.DataFrame(comparison_data)
        
        return df
    
    def find_best_strategy(self, results: Dict, metric: str = 'sharpe_ratio') -> tuple:
        """
        Find best strategy based on metric
        
        Args:
            results: Dictionary of strategy results
            metric: Metric to optimize ('sharpe_ratio', 'total_return', 'win_rate')
            
        Returns:
            Tuple of (best_strategy_name, best_value)
        """
        best_strategy = None
        best_value = -np.inf
        
        for strategy_name, result in results.items():
            value = result.get(metric, -np.inf)
            if value > best_value:
                best_value = value
                best_strategy = strategy_name
        
        return best_strategy, best_value


def main():
    """
    Main function - run comprehensive strategy comparison
    """
    logger.info("=" * 60)
    logger.info("TRADING STRATEGY COMPARISON")
    logger.info("=" * 60)
    
    # Configuration
    SYMBOL = 'AAPL'
    DAYS = 730  # 2 years of data
    INITIAL_CAPITAL = 100000
    
    logger.info(f"\n💡 Make sure you have data for {SYMBOL}!")
    logger.info(f"If not, run: python3 cli.py fetch {SYMBOL} --period 2y --interval 1d\n")
    
    # Initialize
    comparison = StrategyComparison(
        initial_capital=INITIAL_CAPITAL,
        commission=0.001,  # 0.1%
        slippage=0.0005    # 0.05%
    )
    
    # Get data
    try:
        df = comparison.get_data_with_indicators(SYMBOL, days=DAYS)
        logger.info(f"Loaded {len(df)} bars for {SYMBOL}")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return
    
    # Test traditional strategies
    traditional_results = comparison.test_all_strategies(df)
    
    # Test ML strategies
    ml_results = comparison.test_ml_strategies(df, train_size=0.8)
    
    # Combine results
    all_results = {**traditional_results, **ml_results}
    
    # Create comparison table
    logger.info("\n" + "=" * 60)
    logger.info("STRATEGY COMPARISON RESULTS")
    logger.info("=" * 60 + "\n")
    
    comparison_table = comparison.create_comparison_table(all_results)
    print(comparison_table.to_string(index=False))
    
    # Find best strategies
    logger.info("\n" + "=" * 60)
    logger.info("BEST STRATEGIES")
    logger.info("=" * 60)
    
    best_sharpe, sharpe_value = comparison.find_best_strategy(all_results, 'sharpe_ratio')
    best_return, return_value = comparison.find_best_strategy(all_results, 'total_return')
    best_winrate, winrate_value = comparison.find_best_strategy(all_results, 'win_rate')
    
    logger.info(f"🏆 Best Sharpe Ratio: {best_sharpe} ({sharpe_value:.2f})")
    logger.info(f"💰 Best Total Return: {best_return} ({return_value:.2%})")
    logger.info(f"🎯 Best Win Rate: {best_winrate} ({winrate_value:.2%})")
    
    # Save results
    comparison_table.to_csv('strategy_comparison_results.csv', index=False)
    logger.success("\n✅ Results saved to 'strategy_comparison_results.csv'")
    
    logger.info("\n" + "=" * 60)
    logger.info("COMPARISON COMPLETE!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
