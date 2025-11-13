# Algorithmic Trading System

A production-grade algorithmic trading system built from scratch with real-time data ingestion, feature engineering, backtesting, and strategy development capabilities.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-14+-316192.svg)
![Redis](https://img.shields.io/badge/Redis-7+-DC382D.svg)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)
![Cost](https://img.shields.io/badge/Cost-$0-success.svg)

## 🎯 Overview

This system provides a complete end-to-end solution for algorithmic trading, from data collection to strategy backtesting. Built with performance, scalability, and cost-efficiency in mind.

### Key Features

- ✅ **Real-time Data Collection** - Free market data from Yahoo Finance
- ✅ **Time-Series Database** - PostgreSQL with TimescaleDB for optimized storage
- ✅ **100+ Technical Indicators** - RSI, MACD, Bollinger Bands, ADX, and more
- ✅ **Automated Feature Engineering** - Generate trading signals from raw data
- ✅ **Market Regime Detection** - Identify trending/ranging/volatile markets
- ✅ **Professional Backtesting** - Test strategies with realistic transaction costs
- ✅ **8 Pre-Built Strategies** - Ready-to-use trading strategies
- ✅ **Visualization Tools** - Equity curves, drawdowns, trade analysis
- ✅ **CLI Tools** - Easy command-line interface for all operations
- ✅ **$0 Cost** - Everything runs locally with free data sources

## 📊 System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Data Sources (Free)                      │
│              Yahoo Finance | Alpha Vantage                  │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                  Data Pipeline Layer                        │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐    │
│  │Data Fetcher  │→ │ Validator    │→ │Storage Manager  │    │
│  └──────────────┘  └──────────────┘  └─────────────────┘    │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│               Storage Layer (TimescaleDB)                   │
│  ┌────────────┐  ┌────────────┐  ┌───────────────────┐      │
│  │Market Data │  │  Signals   │  │      Trades       │      │
│  └────────────┘  └────────────┘  └───────────────────┘      │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│            Feature Engineering Layer                        │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐    │
│  │ Indicators   │  │  Patterns    │  │ Market Regime   │    │
│  └──────────────┘  └──────────────┘  └─────────────────┘    │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│              Strategy & Backtesting Layer                   │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐    │
│  │  Strategies  │→ │  Backtester  │→ │  Visualization  │    │
│  └──────────────┘  └──────────────┘  └─────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Docker & Docker Compose
- 8GB RAM minimum
- Git

### Installation

1. **Clone the repository:**
```bash
git clone <your-repo-url>
cd algo-trading-system
```

2. **Install Python dependencies:**
```bash
pip install -r requirements.txt
```

3. **Start infrastructure:**
```bash
cd docker
docker-compose up -d
```

4. **Setup database:**
```bash
cd ..
python data_pipeline/storage/timescale_setup.py
```

5. **Test the installation:**
```bash
python tests/test_database.py
```

### First Data Collection

```bash
# Fetch Apple stock data
python cli.py fetch AAPL --period 1mo --interval 1h

# Fetch all tech stocks
python cli.py watchlist --category tech

# Check what was collected
python cli.py stats
```

## 📁 Project Structure

```
algo-trading-system/
├── data_pipeline/
│   ├── ingest/
│   │   ├── data_fetcher.py          # Fetch market data
│   │   ├── storage_manager.py       # Store data in DB
│   │   └── stream_processor.py      # Redis streaming
│   ├── storage/
│   │   ├── database.py              # Database connection
│   │   ├── models.py                # SQLAlchemy models
│   │   └── timescale_setup.py       # DB initialization
│   ├── features/
│   │   ├── technical_indicators.py  # Technical analysis
│   │   ├── feature_engineering.py   # Feature generation
│   │   └── market_regime.py         # Market conditions
│   └── backtest/
│       ├── backtester.py            # Backtesting engine
│       ├── strategies.py            # Pre-built strategies
│       └── visualization.py         # Charts and plots
├── config/
│   └── data_sources.yaml            # Configuration
├── docker/
│   └── docker-compose.yml           # Infrastructure
├── tests/
│   └── test_database.py             # Test suite
├── logs/                            # Log files
├── cli.py                           # Command-line interface
├── requirements.txt                 # Python dependencies
├── .env                             # Environment variables
└── README.md                        # This file
```

## 💻 Usage Examples

### Data Collection

```python
from data_pipeline.ingest.data_fetcher import MarketDataFetcher
from data_pipeline.ingest.storage_manager import DataStorageManager

# Fetch data
fetcher = MarketDataFetcher()
df = fetcher.fetch_historical_data('AAPL', period='1y', interval='1d')

# Store in database
storage = DataStorageManager()
storage.store_dataframe(df)
```

### Feature Engineering

```python
from data_pipeline.features.feature_engineering import FeatureEngineer

engineer = FeatureEngineer()
df_with_features = engineer.add_all_features(df)

# Now you have 100+ features including:
# - Price features (returns, gaps, ranges)
# - Volume features (OBV, VWAP, ratios)
# - Technical indicators (RSI, MACD, Bollinger Bands)
# - Momentum indicators (Stochastic, Williams %R)
# - Trend indicators (ADX, moving averages)
# - Pattern detection (candlestick patterns)
```

### Market Regime Detection

```python
from data_pipeline.features.market_regime import MarketRegimeDetector

detector = MarketRegimeDetector()
df_with_regime = detector.detect_regime(df)

# Get current market conditions
current = detector.get_current_regime(df_with_regime)
print(f"Direction: {current['direction']}")  # bullish/bearish/neutral
print(f"Trend: {current['trend']}")          # trending/ranging
print(f"Volatility: {current['volatility']}")  # high/normal/low

# Check if should trade
should_trade = detector.should_trade(df_with_regime, 'trend_following')
```

### Backtesting

```python
from data_pipeline.backtest.backtester import Backtester
from data_pipeline.backtest.strategies import StrategyTemplates

# Generate signals
strategies = StrategyTemplates()
signals = strategies.macd_trend_following(df)

# Run backtest
backtester = Backtester(
    initial_capital=100000,
    commission=0.001,     # 0.1% per trade
    slippage=0.0005       # 0.05% slippage
)

results = backtester.run(df, signals, stop_loss=0.02, take_profit=0.05)

# Print results
backtester.print_results()

# Visualize
from data_pipeline.backtest.visualization import BacktestVisualizer
viz = BacktestVisualizer()
viz.plot_full_report(
    backtester.get_equity_curve(),
    backtester.get_trades(),
    results
)
```

## 🎮 CLI Commands

### Data Management
```bash
# Fetch single symbol
python cli.py fetch AAPL --period 1mo --interval 1h

# Fetch by date range
python cli.py fetch AAPL --start 2024-01-01 --end 2024-12-31 --interval 1d

# Fetch watchlist category
python cli.py watchlist --category tech
python cli.py watchlist --category crypto

# Fetch all symbols
python cli.py watchlist

# Update all symbols with latest data
python cli.py update

# Show database statistics
python cli.py stats

# Show configured symbols
python cli.py symbols
```

### Strategy Testing
```bash
# Test a strategy (in Python)
python data_pipeline/backtest/backtester.py

# Compare all strategies
python data_pipeline/backtest/strategies.py

# Generate visualizations
python data_pipeline/backtest/visualization.py
```

## 📊 Available Strategies

| Strategy | Best For | Key Indicators |
|----------|----------|----------------|
| **SMA Crossover** | Trending markets | Moving averages |
| **RSI Mean Reversion** | Ranging markets | RSI |
| **MACD Trend** | Strong trends | MACD |
| **Bollinger Breakout** | Volatility | Bollinger Bands |
| **Triple EMA** | Trend confirmation | Multiple EMAs |
| **Momentum** | Strong moves | Price momentum |
| **ADX + RSI** | Trend + timing | ADX, RSI |
| **Multi-Timeframe** | Trend alignment | Multiple MAs |

## 📈 Performance Metrics

The backtester calculates comprehensive metrics:

- **Total Return**: Overall profit/loss percentage
- **Sharpe Ratio**: Risk-adjusted returns (> 1.5 is good)
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Gross profit / Gross loss ratio
- **Average Win/Loss**: Average profit per winning/losing trade
- **Average Holding Period**: Days per trade

## 🔧 Configuration

### Environment Variables (.env)
```env
# Database
DB_HOST=localhost
DB_PORT=5433
DB_NAME=trading_data
DB_USER=trader
DB_PASSWORD=trading123

# Redis
REDIS_HOST=localhost
REDIS_PORT=6380

# API Keys (optional)
ALPHA_VANTAGE_KEY=your_key_here
```

### Data Sources (config/data_sources.yaml)
```yaml
watchlist:
  tech:
    - AAPL
    - MSFT
    - GOOGL
  crypto:
    - BTC-USD
    - ETH-USD

collection:
  historical_period: "1mo"
  interval: "1h"
  update_frequency: 300
```

## 🛠️ Infrastructure

### Services

The system uses Docker Compose to manage:

- **TimescaleDB** (localhost:5433) - Time-series optimized PostgreSQL
- **Redis** (localhost:6380) - Streaming and caching
- **Grafana** (localhost:3001) - Monitoring dashboards

### Database Schema

**market_data** - OHLCV data with hypertable partitioning
- Indexed by timestamp and symbol
- Automatic compression after 7 days
- Data retention for 365 days
- Continuous aggregates for 1h and 1d data

**trading_signals** - Generated trading signals
**trades** - Executed trades with P&L tracking
**positions** - Current open positions

## 📚 Documentation

Detailed guides available:

- **Phase 1**: `TRADING_SYSTEM_SETUP.md` - Initial setup
- **Phase 2**: `PHASE2_GUIDE.md` - Data collection
- **Phase 3**: `PHASE3_GUIDE.md` - Feature engineering
- **Phase 4**: `PHASE4_GUIDE.md` - Backtesting

## 🧪 Testing

Run the test suite:

```bash
# Database tests
python tests/test_database.py

# Data fetcher tests
python data_pipeline/ingest/data_fetcher.py

# Feature engineering tests
python data_pipeline/features/feature_engineering.py

# Backtesting tests
python data_pipeline/backtest/backtester.py
```

## 💰 Cost Breakdown

### Development (Current)
- **Data**: $0/month (Yahoo Finance free tier)
- **Storage**: $0/month (local Docker)
- **Compute**: $0/month (local machine)
- **Total**: **$0/month**

### Production Options

**Tier 1 - Basic ($50/month)**
- DigitalOcean Droplet (4GB): $24/month
- Polygon.io Basic: $29/month

**Tier 2 - Professional ($250-500/month)**
- AWS/GCP Compute: $100-150/month
- Real-time market data: $99-199/month
- Monitoring: $30-100/month

**Tier 3 - High-Frequency ($1000+/month)**
- Dedicated servers: $300-500/month
- Premium data feeds: $500-2000/month
- Co-location: $500-1000/month

## 🚦 Getting Started Workflow

1. **Setup** → Install dependencies, start Docker
2. **Collect Data** → Fetch historical data for symbols
3. **Explore** → Generate features, detect market regimes
4. **Develop Strategy** → Create or customize trading logic
5. **Backtest** → Test on historical data
6. **Optimize** → Tune parameters, improve performance
7. **Paper Trade** → Test with real-time data (no money)
8. **Go Live** → Deploy with real capital (when profitable!)

## ⚠️ Important Notes

### Backtesting Best Practices

- **Avoid Overfitting**: Test on out-of-sample data
- **Include Costs**: Always model commissions and slippage
- **Minimum Trades**: Need 30+ trades for statistical significance
- **Walk-Forward Testing**: Validate across multiple time periods
- **Risk Management**: Always use stop losses

### Data Considerations

- Free data has rate limits (500-2000 calls/day)
- 1-minute data only available for last 7 days
- Crypto trading is 24/7, stocks only during market hours
- Always validate data quality before trading

## 🤝 Contributing

This is a personal project for learning and portfolio purposes. Feel free to fork and customize for your own use!

## 📝 License

This project is for educational and portfolio purposes. Not financial advice.

## ⚠️ Disclaimer

**IMPORTANT**: This software is for educational purposes only. Trading financial instruments carries risk. Past performance does not guarantee future results. Always do your own research and never risk more than you can afford to lose.

The authors are not responsible for any financial losses incurred through the use of this software.

## 🎯 Roadmap

Future enhancements to consider:

- [ ] Machine learning models (LSTM, Random Forest)
- [ ] Live trading integration (Alpaca, Interactive Brokers)
- [ ] Portfolio optimization
- [ ] Multi-asset strategies
- [ ] Real-time alerting (Telegram, Email)
- [ ] Web dashboard
- [ ] Advanced risk management
- [ ] Sentiment analysis from news/social media

## 📞 Support

For questions or issues:
- Review the phase guides in the project
- Check the test files for examples
- Refer to the code documentation

---

**Built with** ❤️ **for learning algorithmic trading**

**Status**: Development Complete ✅  
**Cost**: $0 💰  
**Technologies**: Python, PostgreSQL, TimescaleDB, Redis, Docker, Pandas, NumPy  
**Last Updated**: November 2025
