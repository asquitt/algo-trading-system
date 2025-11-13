"""
Visualization Tools for Backtesting
Plot equity curves, drawdowns, and trade analysis
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Optional
from loguru import logger


class BacktestVisualizer:
    """
    Visualize backtesting results
    Create charts for equity curves, drawdowns, trade analysis, etc.
    """
    
    def __init__(self, figsize: tuple = (15, 10)):
        """
        Initialize visualizer
        
        Args:
            figsize: Default figure size (width, height)
        """
        self.figsize = figsize
        plt.style.use('seaborn-v0_8-darkgrid')
    
    def plot_equity_curve(
        self,
        equity_curve: pd.DataFrame,
        benchmark: Optional[pd.Series] = None,
        title: str = "Equity Curve"
    ):
        """
        Plot equity curve over time
        
        Args:
            equity_curve: DataFrame with 'equity' column
            benchmark: Optional benchmark series to compare
            title: Chart title
        """
        fig, ax = plt.subplots(figsize=(self.figsize[0], 6))
        
        # Plot equity
        ax.plot(equity_curve.index, equity_curve['equity'], 
                label='Strategy', linewidth=2, color='#2E86AB')
        
        # Plot benchmark if provided
        if benchmark is not None:
            ax.plot(benchmark.index, benchmark, 
                   label='Benchmark', linewidth=2, color='#A23B72', alpha=0.7)
        
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Equity ($)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Format y-axis as currency
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        plt.tight_layout()
        plt.show()
    
    def plot_drawdown(self, equity_curve: pd.DataFrame, title: str = "Drawdown"):
        """
        Plot drawdown over time
        
        Args:
            equity_curve: DataFrame with 'equity' column
            title: Chart title
        """
        # Calculate drawdown
        equity = equity_curve['equity']
        running_max = equity.expanding().max()
        drawdown = (equity - running_max) / running_max * 100
        
        fig, ax = plt.subplots(figsize=(self.figsize[0], 5))
        
        # Plot drawdown
        ax.fill_between(drawdown.index, drawdown, 0, 
                        color='#E63946', alpha=0.3, label='Drawdown')
        ax.plot(drawdown.index, drawdown, 
               color='#E63946', linewidth=1.5)
        
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Drawdown (%)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add max drawdown line
        max_dd = drawdown.min()
        ax.axhline(y=max_dd, color='red', linestyle='--', linewidth=2, 
                  label=f'Max Drawdown: {max_dd:.2f}%')
        ax.legend(loc='best', fontsize=10)
        
        plt.tight_layout()
        plt.show()
    
    def plot_returns_distribution(
        self,
        equity_curve: pd.DataFrame,
        bins: int = 50,
        title: str = "Returns Distribution"
    ):
        """
        Plot distribution of returns
        
        Args:
            equity_curve: DataFrame with equity data
            bins: Number of histogram bins
            title: Chart title
        """
        returns = equity_curve['equity'].pct_change().dropna() * 100
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Histogram
        ax.hist(returns, bins=bins, alpha=0.7, color='#06A77D', edgecolor='black')
        
        # Add mean line
        mean_return = returns.mean()
        ax.axvline(x=mean_return, color='red', linestyle='--', linewidth=2,
                  label=f'Mean: {mean_return:.3f}%')
        
        ax.set_xlabel('Daily Return (%)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_trade_analysis(self, trades: pd.DataFrame):
        """
        Plot trade analysis (win/loss, P&L distribution)
        
        Args:
            trades: DataFrame with trade data
        """
        if len(trades) == 0:
            logger.warning("No trades to visualize")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        
        # 1. P&L by trade
        axes[0, 0].bar(range(len(trades)), trades['pnl'], 
                       color=['green' if x > 0 else 'red' for x in trades['pnl']],
                       alpha=0.7)
        axes[0, 0].set_xlabel('Trade Number')
        axes[0, 0].set_ylabel('P&L ($)')
        axes[0, 0].set_title('P&L by Trade')
        axes[0, 0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Cumulative P&L
        cumulative_pnl = trades['pnl'].cumsum()
        axes[0, 1].plot(range(len(trades)), cumulative_pnl, 
                       linewidth=2, color='#2E86AB')
        axes[0, 1].fill_between(range(len(trades)), cumulative_pnl, 0, 
                               alpha=0.3, color='#2E86AB')
        axes[0, 1].set_xlabel('Trade Number')
        axes[0, 1].set_ylabel('Cumulative P&L ($)')
        axes[0, 1].set_title('Cumulative P&L')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Win/Loss distribution
        winning = trades[trades['pnl'] > 0]['pnl']
        losing = trades[trades['pnl'] <= 0]['pnl']
        
        axes[1, 0].hist(winning, bins=20, alpha=0.7, color='green', label='Wins')
        axes[1, 0].hist(losing, bins=20, alpha=0.7, color='red', label='Losses')
        axes[1, 0].set_xlabel('P&L ($)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Win/Loss Distribution')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Return distribution
        axes[1, 1].hist(trades['return'] * 100, bins=30, alpha=0.7, 
                       color='#06A77D', edgecolor='black')
        axes[1, 1].axvline(x=0, color='red', linestyle='--', linewidth=2)
        axes[1, 1].set_xlabel('Return (%)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Trade Returns Distribution')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_monthly_returns(self, equity_curve: pd.DataFrame):
        """
        Plot monthly returns heatmap
        
        Args:
            equity_curve: DataFrame with equity data
        """
        # Check if index is datetime, if not, skip this plot
        if not isinstance(equity_curve.index, pd.DatetimeIndex):
            logger.warning("Monthly returns plot requires DatetimeIndex, skipping...")
            return
        
        # Calculate monthly returns (use 'ME' for month end instead of deprecated 'M')
        equity_monthly = equity_curve['equity'].resample('ME').last()
        monthly_returns = equity_monthly.pct_change().dropna() * 100
        
        if len(monthly_returns) == 0:
            logger.warning("Not enough data for monthly returns")
            return
        
        # Reshape into year x month grid
        returns_pivot = pd.DataFrame({
            'Year': monthly_returns.index.year,
            'Month': monthly_returns.index.month,
            'Return': monthly_returns.values
        }).pivot(index='Year', columns='Month', values='Return')
        
        # Month names
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        returns_pivot.columns = [month_names[i-1] for i in returns_pivot.columns]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Create heatmap
        im = ax.imshow(returns_pivot.values, cmap='RdYlGn', aspect='auto', 
                      vmin=-10, vmax=10)
        
        # Set ticks
        ax.set_xticks(np.arange(len(returns_pivot.columns)))
        ax.set_yticks(np.arange(len(returns_pivot.index)))
        ax.set_xticklabels(returns_pivot.columns)
        ax.set_yticklabels(returns_pivot.index.astype(int))
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Return (%)', rotation=270, labelpad=20)
        
        # Add text annotations
        for i in range(len(returns_pivot.index)):
            for j in range(len(returns_pivot.columns)):
                value = returns_pivot.iloc[i, j]
                if pd.notna(value):
                    text = ax.text(j, i, f'{value:.1f}%',
                                 ha="center", va="center", color="black", fontsize=9)
        
        ax.set_title('Monthly Returns Heatmap', fontsize=14, fontweight='bold')
        ax.set_xlabel('Month', fontsize=12)
        ax.set_ylabel('Year', fontsize=12)
        
        plt.tight_layout()
        plt.show()
    
    def plot_full_report(
        self,
        equity_curve: pd.DataFrame,
        trades: pd.DataFrame,
        results: dict
    ):
        """
        Plot comprehensive backtest report
        
        Args:
            equity_curve: DataFrame with equity data
            trades: DataFrame with trade data
            results: Dictionary with backtest results
        """
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # 1. Equity Curve
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(equity_curve.index, equity_curve['equity'], 
                linewidth=2, color='#2E86AB')
        ax1.set_xlabel('Bar Number')
        ax1.set_ylabel('Equity ($)')
        ax1.set_title('Equity Curve', fontweight='bold')
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        ax1.grid(True, alpha=0.3)
        
        # 2. Drawdown
        ax2 = fig.add_subplot(gs[1, 0])
        equity = equity_curve['equity']
        running_max = equity.expanding().max()
        drawdown = (equity - running_max) / running_max * 100
        ax2.fill_between(range(len(drawdown)), drawdown, 0, color='#E63946', alpha=0.3)
        ax2.plot(range(len(drawdown)), drawdown, color='#E63946', linewidth=1.5)
        ax2.set_xlabel('Bar Number')
        ax2.set_ylabel('Drawdown (%)')
        ax2.set_title('Drawdown', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # 3. Returns Distribution
        ax3 = fig.add_subplot(gs[1, 1])
        returns = equity_curve['equity'].pct_change().dropna() * 100
        if len(returns) > 0:
            ax3.hist(returns, bins=30, alpha=0.7, color='#06A77D', edgecolor='black')
            ax3.axvline(x=returns.mean(), color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {returns.mean():.2f}%')
            ax3.set_xlabel('Return (%)')
            ax3.set_ylabel('Frequency')
            ax3.set_title('Returns Distribution', fontweight='bold')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 4. Trade P&L
        if len(trades) > 0:
            ax4 = fig.add_subplot(gs[2, 0])
            cumulative_pnl = trades['pnl'].cumsum()
            ax4.plot(range(len(trades)), cumulative_pnl, linewidth=2, color='#06A77D')
            ax4.fill_between(range(len(trades)), cumulative_pnl, 0, alpha=0.3, color='#06A77D')
            ax4.set_xlabel('Trade Number')
            ax4.set_ylabel('Cumulative P&L ($)')
            ax4.set_title('Cumulative Trade P&L', fontweight='bold')
            ax4.grid(True, alpha=0.3)
            
            # 5. Win/Loss Distribution
            ax5 = fig.add_subplot(gs[2, 1])
            winning = trades[trades['pnl'] > 0]['pnl']
            losing = trades[trades['pnl'] <= 0]['pnl']
            if len(winning) > 0:
                ax5.hist(winning, bins=20, alpha=0.7, color='green', label='Wins')
            if len(losing) > 0:
                ax5.hist(losing, bins=20, alpha=0.7, color='red', label='Losses')
            ax5.set_xlabel('P&L ($)')
            ax5.set_ylabel('Frequency')
            ax5.set_title('Win/Loss Distribution', fontweight='bold')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        
        plt.suptitle(f"Backtest Report - Total Return: {results['total_return_pct']:.2f}% | "
                    f"Sharpe: {results['sharpe_ratio']:.2f} | "
                    f"Max DD: {results['max_drawdown_pct']:.2f}%",
                    fontsize=16, fontweight='bold', y=0.995)
        
        plt.show()


# Example usage
if __name__ == "__main__":
    from data_pipeline.ingest.data_fetcher import MarketDataFetcher
    from data_pipeline.backtest.backtester import Backtester
    from data_pipeline.backtest.strategies import StrategyTemplates
    
    logger.info("="*70)
    logger.info("Testing Backtest Visualization")
    logger.info("="*70)
    
    # Fetch data
    fetcher = MarketDataFetcher()
    df = fetcher.fetch_historical_data('AAPL', period='2y', interval='1d')
    
    if df is not None:
        # Generate signals
        strategies = StrategyTemplates()
        signals = strategies.sma_crossover(df, 20, 50)
        
        # Run backtest
        backtester = Backtester(initial_capital=100000)
        results = backtester.run(df, signals)
        
        # Get data
        equity_curve = backtester.get_equity_curve()
        trades = backtester.get_trades()
        
        # Visualize
        viz = BacktestVisualizer()
        
        print("\nGenerating visualizations...")
        print("Close the plot windows to continue...")
        
        # Full report
        viz.plot_full_report(equity_curve, trades, results)
        
        # Individual plots
        viz.plot_equity_curve(equity_curve)
        viz.plot_drawdown(equity_curve)
        
        if len(trades) > 0:
            viz.plot_trade_analysis(trades)
        
        # Monthly returns (only if datetime index)
        if isinstance(equity_curve.index, pd.DatetimeIndex):
            viz.plot_monthly_returns(equity_curve)
        
        logger.success("\n✓ Visualization test completed!")
    else:
        logger.error("Failed to fetch data for testing")
