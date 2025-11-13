"""
Backtesting Engine
Fast vectorized backtesting for trading strategies
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta
from loguru import logger


class Backtester:
    """
    Vectorized backtesting engine for trading strategies
    Uses pandas/numpy for fast calculations on historical data
    """
    
    def __init__(
        self,
        initial_capital: float = 100000,
        commission: float = 0.001,  # 0.1% per trade
        slippage: float = 0.0005,   # 0.05% slippage
        position_size: float = 1.0   # Fraction of capital per trade
    ):
        """
        Initialize the backtester
        
        Args:
            initial_capital: Starting capital in dollars
            commission: Commission rate (0.001 = 0.1%)
            slippage: Slippage rate (0.0005 = 0.05%)
            position_size: Fraction of capital to use per trade (0-1)
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.position_size = position_size
        
        self.results = None
        self.trades = None
        self.equity_curve = None
    
    def run(
        self,
        df: pd.DataFrame,
        signals: pd.Series,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> Dict:
        """
        Run backtest on data with signals
        
        Args:
            df: DataFrame with OHLCV data
            signals: Series with 1 (buy), -1 (sell), 0 (hold)
            stop_loss: Stop loss percentage (e.g., 0.02 for 2%)
            take_profit: Take profit percentage (e.g., 0.05 for 5%)
            
        Returns:
            Dictionary with backtest results
        """
        logger.info("Running backtest...")
        
        df = df.copy()
        df['signal'] = signals
        
        # Initialize columns with proper dtypes
        df['position'] = 0  # Current position: 1 (long), -1 (short), 0 (flat)
        df['shares'] = 0    # Number of shares held
        df['cash'] = float(self.initial_capital)
        df['holdings'] = 0.0  # Value of holdings
        df['equity'] = float(self.initial_capital)
        df['returns'] = 0.0
        df['trade_pnl'] = 0.0
        
        # Track trades
        trades = []
        current_position = 0
        entry_price = 0
        entry_idx = 0
        shares_held = 0
        cash = float(self.initial_capital)
        
        # Iterate through data
        for i in range(len(df)):
            row = df.iloc[i]
            price = float(row['close'])
            signal = row['signal']
            
            # Update holdings value
            holdings_value = shares_held * price
            equity = cash + holdings_value
            
            # Check stop loss and take profit
            if current_position != 0 and entry_price > 0:
                pnl_pct = (price - entry_price) / entry_price * current_position
                
                if stop_loss and pnl_pct <= -stop_loss:
                    # Stop loss hit
                    signal = -current_position
                    logger.debug(f"Stop loss triggered at {price}")
                
                elif take_profit and pnl_pct >= take_profit:
                    # Take profit hit
                    signal = -current_position
                    logger.debug(f"Take profit triggered at {price}")
            
            # Process signals
            if signal == 1 and current_position <= 0:  # Buy signal
                # Close short if any
                if current_position == -1:
                    pnl = (entry_price - price) * shares_held
                    cost = price * shares_held * (self.commission + self.slippage)
                    cash += entry_price * shares_held - cost
                    
                    trades.append({
                        'entry_date': df.index[entry_idx],
                        'exit_date': df.index[i],
                        'type': 'SHORT',
                        'entry_price': entry_price,
                        'exit_price': price,
                        'shares': shares_held,
                        'pnl': pnl - cost,
                        'return': (entry_price - price) / entry_price
                    })
                
                # Open long
                shares_to_buy = int((cash * self.position_size) / price)
                if shares_to_buy > 0:
                    cost = price * shares_to_buy * (1 + self.commission + self.slippage)
                    if cost <= cash:
                        cash -= cost
                        shares_held = shares_to_buy
                        entry_price = price
                        entry_idx = i
                        current_position = 1
            
            elif signal == -1 and current_position >= 0:  # Sell signal
                # Close long if any
                if current_position == 1:
                    pnl = (price - entry_price) * shares_held
                    proceeds = price * shares_held
                    cost = proceeds * (self.commission + self.slippage)
                    cash += proceeds - cost
                    
                    trades.append({
                        'entry_date': df.index[entry_idx],
                        'exit_date': df.index[i],
                        'type': 'LONG',
                        'entry_price': entry_price,
                        'exit_price': price,
                        'shares': shares_held,
                        'pnl': pnl - cost,
                        'return': (price - entry_price) / entry_price
                    })
                    
                    shares_held = 0
                    current_position = 0
                
                # Open short (optional - can be disabled)
                # For simplicity, we'll skip short selling for now
            
            # Update DataFrame with proper float types
            df.at[df.index[i], 'position'] = current_position
            df.at[df.index[i], 'shares'] = shares_held
            df.at[df.index[i], 'cash'] = float(cash)
            df.at[df.index[i], 'holdings'] = float(shares_held * price)
            df.at[df.index[i], 'equity'] = float(cash + shares_held * price)
        
        # Calculate returns
        df['returns'] = df['equity'].pct_change()
        
        # Store results
        self.equity_curve = df[['equity', 'position', 'cash', 'holdings']].copy()
        self.trades = pd.DataFrame(trades) if trades else pd.DataFrame()
        
        # Calculate metrics
        metrics = self._calculate_metrics(df, self.trades)
        
        self.results = metrics
        
        logger.success(f"✓ Backtest complete: {len(trades)} trades, "
                      f"Final equity: ${metrics['final_equity']:,.2f}")
        
        return metrics
    
    def _calculate_metrics(self, df: pd.DataFrame, trades: pd.DataFrame) -> Dict:
        """Calculate performance metrics"""
        
        final_equity = df['equity'].iloc[-1]
        total_return = (final_equity - self.initial_capital) / self.initial_capital
        
        # Returns statistics
        returns = df['returns'].dropna()
        
        # Sharpe Ratio (annualized)
        if len(returns) > 0 and returns.std() > 0:
            sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std()
        else:
            sharpe_ratio = 0
        
        # Maximum Drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Trade statistics
        if len(trades) > 0:
            winning_trades = trades[trades['pnl'] > 0]
            losing_trades = trades[trades['pnl'] < 0]
            
            win_rate = len(winning_trades) / len(trades) if len(trades) > 0 else 0
            
            avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
            avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0
            
            profit_factor = (
                abs(winning_trades['pnl'].sum() / losing_trades['pnl'].sum())
                if len(losing_trades) > 0 and losing_trades['pnl'].sum() != 0
                else float('inf') if len(winning_trades) > 0 else 0
            )
            
            # Average holding period
            trades['holding_period'] = (
                pd.to_datetime(trades['exit_date']) - pd.to_datetime(trades['entry_date'])
            ).dt.days
            avg_holding_period = trades['holding_period'].mean()
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
            avg_holding_period = 0
        
        # Calculate total days - handle both datetime and integer indices
        try:
            if isinstance(df.index[0], (pd.Timestamp, datetime)):
                total_days = (df.index[-1] - df.index[0]).days
            else:
                # If index is integer, calculate from timestamp column
                if 'timestamp' in df.columns:
                    total_days = (pd.to_datetime(df['timestamp'].iloc[-1]) - 
                                 pd.to_datetime(df['timestamp'].iloc[0])).days
                else:
                    total_days = len(df)  # Fallback to number of bars
        except:
            total_days = len(df)
        
        metrics = {
            'initial_capital': self.initial_capital,
            'final_equity': final_equity,
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown * 100,
            'total_trades': len(trades),
            'winning_trades': len(winning_trades) if len(trades) > 0 else 0,
            'losing_trades': len(losing_trades) if len(trades) > 0 else 0,
            'win_rate': win_rate,
            'win_rate_pct': win_rate * 100,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'avg_holding_period': avg_holding_period,
            'start_date': df.index[0],
            'end_date': df.index[-1],
            'total_days': total_days
        }
        
        return metrics
    
    def print_results(self):
        """Print backtest results in a nice format"""
        if self.results is None:
            logger.error("No results to display. Run backtest first.")
            return
        
        r = self.results
        
        print("\n" + "="*70)
        print("BACKTEST RESULTS")
        print("="*70)
        
        print(f"\nPeriod: {r['start_date']} to {r['end_date']} ({r['total_days']} days)")
        
        print(f"\n{'PORTFOLIO PERFORMANCE':<50}")
        print("-"*70)
        print(f"{'Initial Capital:':<40} ${r['initial_capital']:>20,.2f}")
        print(f"{'Final Equity:':<40} ${r['final_equity']:>20,.2f}")
        print(f"{'Total Return:':<40} {r['total_return_pct']:>19,.2f}%")
        print(f"{'Sharpe Ratio:':<40} {r['sharpe_ratio']:>20,.2f}")
        print(f"{'Maximum Drawdown:':<40} {r['max_drawdown_pct']:>19,.2f}%")
        
        print(f"\n{'TRADE STATISTICS':<50}")
        print("-"*70)
        print(f"{'Total Trades:':<40} {r['total_trades']:>20,}")
        print(f"{'Winning Trades:':<40} {r['winning_trades']:>20,}")
        print(f"{'Losing Trades:':<40} {r['losing_trades']:>20,}")
        print(f"{'Win Rate:':<40} {r['win_rate_pct']:>19,.2f}%")
        print(f"{'Average Win:':<40} ${r['avg_win']:>20,.2f}")
        print(f"{'Average Loss:':<40} ${r['avg_loss']:>20,.2f}")
        print(f"{'Profit Factor:':<40} {r['profit_factor']:>20,.2f}")
        print(f"{'Avg Holding Period:':<40} {r['avg_holding_period']:>17,.1f} days")
        
        print("="*70 + "\n")
    
    def get_trades(self) -> pd.DataFrame:
        """Get DataFrame of all trades"""
        return self.trades.copy() if self.trades is not None else pd.DataFrame()
    
    def get_equity_curve(self) -> pd.DataFrame:
        """Get equity curve DataFrame"""
        return self.equity_curve.copy() if self.equity_curve is not None else pd.DataFrame()


# Example usage
if __name__ == "__main__":
    from data_pipeline.ingest.data_fetcher import MarketDataFetcher
    from data_pipeline.features.technical_indicators import TechnicalIndicators
    
    logger.info("="*70)
    logger.info("Testing Backtesting Engine")
    logger.info("="*70)
    
    # Fetch data
    fetcher = MarketDataFetcher()
    df = fetcher.fetch_historical_data('AAPL', period='1y', interval='1d')
    
    if df is not None:
        # Simple moving average crossover strategy
        indicators = TechnicalIndicators()
        df['sma_20'] = indicators.sma(df, period=20)
        df['sma_50'] = indicators.sma(df, period=50)
        
        # Generate signals
        df['signal'] = 0
        df.loc[df['sma_20'] > df['sma_50'], 'signal'] = 1   # Buy when 20 > 50
        df.loc[df['sma_20'] < df['sma_50'], 'signal'] = -1  # Sell when 20 < 50
        
        # Run backtest
        backtester = Backtester(
            initial_capital=100000,
            commission=0.001,
            slippage=0.0005
        )
        
        results = backtester.run(df, df['signal'])
        
        # Print results
        backtester.print_results()
        
        # Show some trades
        trades = backtester.get_trades()
        if len(trades) > 0:
            print("\nSample Trades:")
            print(trades.head(10).to_string(index=False))
        
        logger.success("\n✓ Backtesting test completed!")
    else:
        logger.error("Failed to fetch data for testing")
