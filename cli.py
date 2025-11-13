"""
CLI Tool for Market Data Collection
Easy command-line interface to fetch and store market data
"""
import argparse
import yaml
from pathlib import Path
from loguru import logger
from datetime import datetime

from data_pipeline.ingest.data_fetcher import MarketDataFetcher
from data_pipeline.ingest.storage_manager import DataStorageManager


def load_config():
    """Load configuration from data_sources.yaml"""
    config_path = Path('config/data_sources.yaml')
    
    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    else:
        logger.warning("Config file not found, using defaults")
        return {
            'watchlist': {
                'tech': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA'],
                'crypto': ['BTC-USD', 'ETH-USD']
            },
            'collection': {
                'historical_period': '1mo',
                'interval': '1h'
            }
        }


def fetch_symbol(args):
    """Fetch data for a single symbol"""
    logger.info(f"Fetching {args.symbol}...")
    
    fetcher = MarketDataFetcher()
    storage = DataStorageManager()
    
    # Fetch data
    if args.start and args.end:
        df = fetcher.fetch_date_range(
            args.symbol,
            args.start,
            args.end,
            args.interval
        )
    else:
        df = fetcher.fetch_historical_data(
            args.symbol,
            period=args.period,
            interval=args.interval
        )
    
    if df is None or df.empty:
        logger.error(f"No data retrieved for {args.symbol}")
        return
    
    # Store data
    count = storage.store_dataframe(df, check_duplicates=not args.force)
    
    logger.success(f"✓ Successfully stored {count} records for {args.symbol}")


def fetch_watchlist(args):
    """Fetch data for all symbols in watchlist"""
    config = load_config()
    
    # Get symbols from specified category or all
    if args.category:
        if args.category in config['watchlist']:
            symbols = config['watchlist'][args.category]
        else:
            logger.error(f"Category '{args.category}' not found in config")
            logger.info(f"Available categories: {list(config['watchlist'].keys())}")
            return
    else:
        # All symbols from all categories
        symbols = []
        for category in config['watchlist'].values():
            symbols.extend(category)
    
    logger.info(f"Fetching data for {len(symbols)} symbols...")
    
    fetcher = MarketDataFetcher()
    storage = DataStorageManager()
    
    # Fetch data
    data = fetcher.fetch_multiple_symbols(
        symbols,
        period=args.period,
        interval=args.interval
    )
    
    # Store data
    results = storage.store_multiple_dataframes(data, check_duplicates=not args.force)
    
    logger.success(f"✓ Completed: {results['successful']}/{results['total_symbols']} symbols, "
                  f"{results['total_records']} total records stored")


def show_stats(args):
    """Show database statistics"""
    storage = DataStorageManager()
    stats = storage.get_statistics()
    
    print("\n" + "="*60)
    print("DATABASE STATISTICS")
    print("="*60)
    print(f"Total Records:    {stats.get('total_records', 0):,}")
    print(f"Unique Symbols:   {stats.get('unique_symbols', 0)}")
    print(f"Oldest Data:      {stats.get('oldest_data', 'N/A')}")
    print(f"Newest Data:      {stats.get('newest_data', 'N/A')}")
    print(f"Date Range:       {stats.get('date_range_days', 0)} days")
    print("="*60)


def show_symbols(args):
    """Show configured watchlist"""
    config = load_config()
    
    print("\n" + "="*60)
    print("CONFIGURED WATCHLIST")
    print("="*60)
    
    for category, symbols in config['watchlist'].items():
        print(f"\n{category.upper()}:")
        for symbol in symbols:
            print(f"  • {symbol}")
    
    total = sum(len(symbols) for symbols in config['watchlist'].values())
    print(f"\nTotal: {total} symbols")
    print("="*60)


def update_data(args):
    """Update data for all symbols (fetch only new data)"""
    config = load_config()
    
    # Get all symbols
    symbols = []
    for category in config['watchlist'].values():
        symbols.extend(category)
    
    logger.info(f"Updating data for {len(symbols)} symbols...")
    
    fetcher = MarketDataFetcher()
    storage = DataStorageManager()
    
    updated_count = 0
    
    for symbol in symbols:
        try:
            # Get latest timestamp for this symbol
            latest = storage.get_latest_timestamp(symbol, args.interval)
            
            if latest:
                # Fetch only new data since last update
                logger.info(f"{symbol}: Last data from {latest}, fetching updates...")
                # Note: yfinance doesn't support "since" param well, so we fetch recent period
                df = fetcher.fetch_historical_data(symbol, period='5d', interval=args.interval)
            else:
                # No existing data, fetch full history
                logger.info(f"{symbol}: No existing data, fetching full history...")
                df = fetcher.fetch_historical_data(
                    symbol,
                    period=config['collection']['historical_period'],
                    interval=args.interval
                )
            
            if df is not None and not df.empty:
                count = storage.store_dataframe(df, check_duplicates=True)
                updated_count += count
            
        except Exception as e:
            logger.error(f"Error updating {symbol}: {e}")
    
    logger.success(f"✓ Update complete: {updated_count} new records stored")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description='Market Data Collection Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fetch single symbol
  python cli.py fetch AAPL --period 1mo --interval 1h
  
  # Fetch all tech stocks
  python cli.py watchlist --category tech
  
  # Fetch all symbols
  python cli.py watchlist
  
  # Update all symbols with latest data
  python cli.py update
  
  # Show statistics
  python cli.py stats
  
  # Show configured symbols
  python cli.py symbols
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Fetch single symbol
    fetch_parser = subparsers.add_parser('fetch', help='Fetch data for a single symbol')
    fetch_parser.add_argument('symbol', help='Stock symbol (e.g., AAPL, BTC-USD)')
    fetch_parser.add_argument('--period', default='1mo', 
                             help='Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, max)')
    fetch_parser.add_argument('--interval', default='1h',
                             help='Data interval (1m, 5m, 15m, 1h, 1d)')
    fetch_parser.add_argument('--start', help='Start date (YYYY-MM-DD)')
    fetch_parser.add_argument('--end', help='End date (YYYY-MM-DD)')
    fetch_parser.add_argument('--force', action='store_true',
                             help='Force insert even if duplicates exist')
    
    # Fetch watchlist
    watchlist_parser = subparsers.add_parser('watchlist', help='Fetch data for watchlist')
    watchlist_parser.add_argument('--category', help='Watchlist category (tech, finance, crypto, etc.)')
    watchlist_parser.add_argument('--period', default='1mo',
                                  help='Data period')
    watchlist_parser.add_argument('--interval', default='1h',
                                  help='Data interval')
    watchlist_parser.add_argument('--force', action='store_true',
                                  help='Force insert even if duplicates exist')
    
    # Update data
    update_parser = subparsers.add_parser('update', help='Update all symbols with latest data')
    update_parser.add_argument('--interval', default='1h', help='Data interval')
    
    # Show statistics
    subparsers.add_parser('stats', help='Show database statistics')
    
    # Show symbols
    subparsers.add_parser('symbols', help='Show configured watchlist')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Execute command
    if args.command == 'fetch':
        fetch_symbol(args)
    elif args.command == 'watchlist':
        fetch_watchlist(args)
    elif args.command == 'update':
        update_data(args)
    elif args.command == 'stats':
        show_stats(args)
    elif args.command == 'symbols':
        show_symbols(args)


if __name__ == '__main__':
    main()
