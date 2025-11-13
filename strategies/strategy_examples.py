"""
Quick Start Example - Test Individual Strategies
V4 - Compatible with your actual TechnicalIndicators methods
"""
import pandas as pd
import sys
import os
from datetime import datetime, timedelta

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from loguru import logger

try:
    from data_pipeline.storage.storage_manager import DataStorageManager as StorageManager
    from data_pipeline.features.technical_indicators import TechnicalIndicators
    from data_pipeline.backtest.backtester import Backtester
except ImportError as e:
    logger.error(f"Import error: {e}")
    sys.exit(1)

from ml_strategies import RandomForestStrategy, GradientBoostingStrategy
from mean_reversion_advanced import MeanReversionAdvanced
from momentum_advanced import MomentumAdvanced


def add_all_indicators(df: pd.DataFrame, indicators: TechnicalIndicators) -> pd.DataFrame:
    """
    Add all technical indicators to dataframe using your individual methods
    
    Args:
        df: DataFrame with OHLCV data
        indicators: TechnicalIndicators instance
        
    Returns:
        DataFrame with all indicators added
    """
    logger.info("Calculating technical indicators...")
    
    # Make a copy to avoid modifying original
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
    
    logger.success(f"Added {len([c for c in df.columns if c not in ['open', 'high', 'low', 'close', 'volume']])} indicators")
    
    return df


def fetch_data(symbol: str, days: int = 365) -> pd.DataFrame:
    """
    Fetch data using your StorageManager's get_data_range method
    
    Args:
        symbol: Stock symbol
        days: Days of historical data
        
    Returns:
        DataFrame with OHLCV data
    """
    storage = StorageManager()
    
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    # Use get_data_range method
    df = storage.get_data_range(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date
    )
    
    if df is None or df.empty:
        logger.error(f"No data found for {symbol}")
        logger.info(f"Try fetching data first: python3 cli.py fetch {symbol} --period {days}d --interval 1d")
        return pd.DataFrame()
    
    logger.info(f"Loaded {len(df)} bars for {symbol}")
    return df


def print_results(strategy_name: str, results: dict):
    """Pretty print backtest results"""
    logger.info(f"\n{'='*60}")
    logger.info(f"📊 {strategy_name}")
    logger.info(f"{'='*60}")
    logger.info(f"Total Return:      {results['total_return']:>10.2%}")
    logger.info(f"Sharpe Ratio:      {results['sharpe_ratio']:>10.2f}")
    logger.info(f"Max Drawdown:      {results['max_drawdown']:>10.2%}")
    logger.info(f"Win Rate:          {results['win_rate']:>10.2%}")
    logger.info(f"Total Trades:      {results['total_trades']:>10d}")
    logger.info(f"Profit Factor:     {results.get('profit_factor', 0):>10.2f}")
    logger.info(f"Avg Win:           {results.get('avg_win', 0):>10.2%}")
    logger.info(f"Avg Loss:          {results.get('avg_loss', 0):>10.2%}")
    logger.info(f"{'='*60}\n")


def example_1_zscore_mean_reversion():
    """Example 1: Z-Score Mean Reversion Strategy"""
    logger.info("\n🔵 Example 1: Z-Score Mean Reversion")
    
    # Setup
    indicators = TechnicalIndicators()
    mr = MeanReversionAdvanced()
    
    # Get data
    df = fetch_data('AAPL', days=365)
    if df.empty:
        return None
    
    df = add_all_indicators(df, indicators)
    
    # Generate signals
    signals = mr.zscore_strategy(
        df,
        lookback=20,
        entry_threshold=2.0,
        exit_threshold=0.5
    )
    
    logger.info(f"Buy signals: {(signals == 1).sum()}")
    logger.info(f"Sell signals: {(signals == -1).sum()}")
    
    # Backtest
    bt = Backtester(initial_capital=100000, commission=0.001, slippage=0.0005)
    results = bt.run(df, signals)
    
    print_results("Z-Score Mean Reversion", results)
    return results


def example_2_pairs_trading():
    """Example 2: Pairs Trading (Cointegration)"""
    logger.info("\n🔵 Example 2: Pairs Trading")
    
    # Setup
    indicators = TechnicalIndicators()
    mr = MeanReversionAdvanced()
    
    # Get data for two correlated stocks
    df1 = fetch_data('AAPL', days=365)
    df2 = fetch_data('MSFT', days=365)
    
    if df1.empty or df2.empty:
        logger.error("Could not fetch data for both symbols")
        return None
    
    df1 = add_all_indicators(df1, indicators)
    df2 = add_all_indicators(df2, indicators)
    
    # Generate signals
    signals, metadata = mr.cointegration_pairs_trading(
        df1, df2,
        lookback=60,
        entry_threshold=2.0,
        exit_threshold=0.5
    )
    
    logger.info(f"Cointegrated: {metadata['is_cointegrated']}")
    logger.info(f"P-value: {metadata['cointegration_pvalue']:.4f}")
    logger.info(f"Hedge Ratio: {metadata['hedge_ratio']:.4f}")
    logger.info(f"Buy signals: {(signals == 1).sum()}")
    logger.info(f"Sell signals: {(signals == -1).sum()}")
    
    # Backtest (on first asset)
    bt = Backtester(initial_capital=100000, commission=0.001, slippage=0.0005)
    results = bt.run(df1, signals)
    
    print_results("Pairs Trading (AAPL vs MSFT)", results)
    return results


def example_3_multi_timeframe_momentum():
    """Example 3: Multi-Timeframe Momentum"""
    logger.info("\n🔵 Example 3: Multi-Timeframe Momentum")
    
    # Setup
    indicators = TechnicalIndicators()
    momentum = MomentumAdvanced()
    
    # Get data
    df = fetch_data('AAPL', days=365)
    if df.empty:
        return None
    
    df = add_all_indicators(df, indicators)
    
    # Generate signals
    signals = momentum.multi_timeframe_momentum(
        df,
        short_period=10,
        medium_period=30,
        long_period=60,
        weights=(0.5, 0.3, 0.2)
    )
    
    logger.info(f"Buy signals: {(signals == 1).sum()}")
    logger.info(f"Sell signals: {(signals == -1).sum()}")
    
    # Backtest
    bt = Backtester(initial_capital=100000, commission=0.001, slippage=0.0005)
    results = bt.run(df, signals)
    
    print_results("Multi-Timeframe Momentum", results)
    return results


def example_4_regime_adaptive_momentum():
    """Example 4: Regime-Adaptive Momentum"""
    logger.info("\n🔵 Example 4: Regime-Adaptive Momentum")
    
    # Setup
    indicators = TechnicalIndicators()
    momentum = MomentumAdvanced()
    
    # Get data
    df = fetch_data('AAPL', days=365)
    if df.empty:
        return None
    
    df = add_all_indicators(df, indicators)
    
    # Generate signals
    signals, regime = momentum.regime_adaptive_momentum(
        df,
        volatility_lookback=20,
        low_vol_threshold=0.01,
        high_vol_threshold=0.03,
        momentum_period=20
    )
    
    logger.info(f"Low vol periods: {(regime == 'low_vol').sum()}")
    logger.info(f"Medium vol periods: {(regime == 'medium').sum()}")
    logger.info(f"High vol periods: {(regime == 'high_vol').sum()}")
    logger.info(f"Buy signals: {(signals == 1).sum()}")
    logger.info(f"Sell signals: {(signals == -1).sum()}")
    
    # Backtest
    bt = Backtester(initial_capital=100000, commission=0.001, slippage=0.0005)
    results = bt.run(df, signals)
    
    print_results("Regime-Adaptive Momentum", results)
    return results


def example_5_random_forest_ml():
    """Example 5: Random Forest ML Strategy"""
    logger.info("\n🔵 Example 5: Random Forest ML Strategy")
    
    # Setup
    indicators = TechnicalIndicators()
    
    # Get data (need more for ML)
    df = fetch_data('AAPL', days=730)  # 2 years
    if df.empty:
        return None
    
    df = add_all_indicators(df, indicators)
    
    # Split for train/test
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    # Train model
    logger.info("Training Random Forest...")
    rf = RandomForestStrategy(n_estimators=100, max_depth=8)
    metrics = rf.train(
        train_df,
        horizon=1,
        threshold=0.002,
        test_size=0.2
    )
    
    logger.info(f"Training Accuracy: {metrics['train_accuracy']:.3f}")
    logger.info(f"Test Accuracy: {metrics['test_accuracy']:.3f}")
    logger.info(f"Number of Features: {metrics['n_features']}")
    
    # Get feature importance
    importance = rf.get_feature_importance()
    logger.info("\nTop 10 Features:")
    print(importance.head(10).to_string(index=False))
    
    # Generate signals on out-of-sample data
    signals = rf.predict(test_df)
    
    logger.info(f"\nOut-of-sample signals:")
    logger.info(f"Buy signals: {(signals == 1).sum()}")
    logger.info(f"Sell signals: {(signals == -1).sum()}")
    logger.info(f"Hold signals: {(signals == 0).sum()}")
    
    # Backtest
    bt = Backtester(initial_capital=100000, commission=0.001, slippage=0.0005)
    results = bt.run(test_df, signals)
    
    print_results("Random Forest ML", results)
    
    # Save model
    os.makedirs('models', exist_ok=True)
    rf.save_model('models/rf_example.pkl')
    logger.info("✓ Model saved to 'models/rf_example.pkl'")
    
    return results


def example_6_volume_confirmed_momentum():
    """Example 6: Volume-Confirmed Momentum"""
    logger.info("\n🔵 Example 6: Volume-Confirmed Momentum")
    
    # Setup
    indicators = TechnicalIndicators()
    momentum = MomentumAdvanced()
    
    # Get data
    df = fetch_data('AAPL', days=365)
    if df.empty:
        return None
    
    df = add_all_indicators(df, indicators)
    
    # Generate signals
    signals = momentum.volume_confirmed_momentum(
        df,
        momentum_period=20,
        volume_period=20,
        volume_multiplier=1.5
    )
    
    logger.info(f"Buy signals: {(signals == 1).sum()}")
    logger.info(f"Sell signals: {(signals == -1).sum()}")
    
    # Backtest
    bt = Backtester(initial_capital=100000, commission=0.001, slippage=0.0005)
    results = bt.run(df, signals)
    
    print_results("Volume-Confirmed Momentum", results)
    return results


def menu():
    """Display menu and run selected example"""
    logger.info("\n" + "="*60)
    logger.info("🚀 STRATEGY EXAMPLES - QUICK START")
    logger.info("="*60)
    logger.info("\nChoose an example to run:\n")
    logger.info("1. Z-Score Mean Reversion")
    logger.info("2. Pairs Trading (Cointegration)")
    logger.info("3. Multi-Timeframe Momentum")
    logger.info("4. Regime-Adaptive Momentum")
    logger.info("5. Random Forest ML Strategy")
    logger.info("6. Volume-Confirmed Momentum")
    logger.info("7. Run ALL Examples")
    logger.info("0. Exit\n")
    
    choice = input("Enter your choice (0-7): ").strip()
    
    examples = {
        '1': example_1_zscore_mean_reversion,
        '2': example_2_pairs_trading,
        '3': example_3_multi_timeframe_momentum,
        '4': example_4_regime_adaptive_momentum,
        '5': example_5_random_forest_ml,
        '6': example_6_volume_confirmed_momentum,
    }
    
    if choice == '0':
        logger.info("Goodbye! 👋")
        return
    elif choice == '7':
        logger.info("\n🔥 Running ALL examples...\n")
        for example_func in examples.values():
            try:
                example_func()
            except Exception as e:
                logger.error(f"Example failed: {e}")
                import traceback
                traceback.print_exc()
        logger.info("\n✅ All examples complete!")
    elif choice in examples:
        try:
            examples[choice]()
        except Exception as e:
            logger.error(f"Example failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        logger.error("Invalid choice! Please enter 0-7")
        menu()


if __name__ == "__main__":
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Check if we have data
    logger.info("\n💡 TIP: If you get 'No data found' errors, fetch data first:")
    logger.info("   python3 cli.py fetch AAPL --period 2y --interval 1d")
    logger.info("   python3 cli.py fetch MSFT --period 2y --interval 1d\n")
    
    # Run menu
    menu()
