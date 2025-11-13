"""
Test database setup, connectivity, and basic operations
Run this after setting up TimescaleDB
"""
from data_pipeline.storage.database import db_manager
from data_pipeline.storage.models import MarketData, TradingSignal, Trade, Position
from datetime import datetime, timedelta
from loguru import logger
import sys

def test_database_connection():
    """Test basic database connectivity"""
    logger.info("\n" + "="*60)
    logger.info("TEST 1: Database Connection")
    logger.info("="*60)
    
    try:
        success = db_manager.initialize()
        assert success, "Failed to connect to database"
        logger.success("✓ Database connection successful")
        
        # Test TimescaleDB extension
        result = db_manager.execute_sql(
            "SELECT extversion FROM pg_extension WHERE extname='timescaledb';"
        )
        version = result.fetchone()
        if version:
            logger.success(f"✓ TimescaleDB version: {version[0]}")
        else:
            logger.error("✗ TimescaleDB extension not found")
            return False
            
        return True
    except Exception as e:
        logger.error(f"✗ Connection test failed: {e}")
        return False


def test_insert_market_data():
    """Test inserting and retrieving market data"""
    logger.info("\n" + "="*60)
    logger.info("TEST 2: Market Data Insertion")
    logger.info("="*60)
    
    try:
        with db_manager.get_session() as session:
            # Create sample data
            now = datetime.utcnow()
            test_data = []
            
            # Insert 10 bars of test data
            for i in range(10):
                data = MarketData(
                    timestamp=now - timedelta(hours=i),
                    symbol='AAPL',
                    exchange='NASDAQ',
                    open=100.0 + i,
                    high=105.0 + i,
                    low=99.0 + i,
                    close=103.0 + i,
                    volume=1000000 * (i + 1),
                    interval='1h',
                    source='test',
                    extra_data={'test': True, 'batch': 1}
                )
                test_data.append(data)
                session.add(data)
            
            session.commit()
            logger.success(f"✓ Inserted {len(test_data)} market data records")
            
            # Query it back
            result = session.query(MarketData).filter(
                MarketData.symbol == 'AAPL',
                MarketData.source == 'test'
            ).order_by(MarketData.timestamp.desc()).all()
            
            assert len(result) >= 10, "Failed to retrieve all inserted data"
            assert result[0].close == 103.0, "Data mismatch in most recent record"
            
            logger.success(f"✓ Retrieved {len(result)} records successfully")
            logger.info(f"  Latest: {result[0].symbol} @ ${result[0].close} on {result[0].timestamp}")
            
            # Test to_dict method
            data_dict = result[0].to_dict()
            assert 'symbol' in data_dict, "to_dict() missing fields"
            logger.success("✓ Data serialization works")
            
            return True
            
    except Exception as e:
        logger.error(f"✗ Market data test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def test_insert_trading_signal():
    """Test inserting trading signals"""
    logger.info("\n" + "="*60)
    logger.info("TEST 3: Trading Signal Insertion")
    logger.info("="*60)
    
    try:
        with db_manager.get_session() as session:
            # Create sample signal
            signal = TradingSignal(
                timestamp=datetime.utcnow(),
                symbol='AAPL',
                strategy='mean_reversion',
                signal_type='BUY',
                strength=0.85,
                price=150.25,
                target_price=155.00,
                stop_loss=148.00,
                extra_data={'confidence': 'high', 'test': True}
            )
            
            session.add(signal)
            session.commit()
            
            logger.success("✓ Trading signal inserted")
            
            # Query it back
            result = session.query(TradingSignal).filter(
                TradingSignal.symbol == 'AAPL',
                TradingSignal.strategy == 'mean_reversion'
            ).first()
            
            assert result is not None, "Failed to retrieve signal"
            assert result.signal_type == 'BUY', "Signal type mismatch"
            assert result.strength == 0.85, "Signal strength mismatch"
            
            logger.success(f"✓ Signal retrieved: {result.signal_type} {result.symbol} @ strength {result.strength}")
            
            return True
            
    except Exception as e:
        logger.error(f"✗ Trading signal test failed: {e}")
        return False


def test_insert_trade():
    """Test inserting trades"""
    logger.info("\n" + "="*60)
    logger.info("TEST 4: Trade Insertion")
    logger.info("="*60)
    
    try:
        with db_manager.get_session() as session:
            # Create sample trade
            trade = Trade(
                timestamp=datetime.utcnow(),
                symbol='AAPL',
                side='BUY',
                quantity=100,
                price=150.25,
                order_type='MARKET',
                status='FILLED',
                strategy='mean_reversion',
                order_id='TEST_ORDER_001',
                commission=1.00,
                slippage=0.02,
                total_cost=15026.00,
                extra_data={'test': True}
            )
            
            session.add(trade)
            session.commit()
            
            logger.success("✓ Trade inserted")
            
            # Query it back
            result = session.query(Trade).filter(
                Trade.order_id == 'TEST_ORDER_001'
            ).first()
            
            assert result is not None, "Failed to retrieve trade"
            assert result.status == 'FILLED', "Trade status mismatch"
            assert result.quantity == 100, "Trade quantity mismatch"
            
            logger.success(f"✓ Trade retrieved: {result.side} {result.quantity} {result.symbol} @ ${result.price}")
            
            return True
            
    except Exception as e:
        logger.error(f"✗ Trade test failed: {e}")
        return False


def test_hypertable_setup():
    """Test that hypertables are set up correctly"""
    logger.info("\n" + "="*60)
    logger.info("TEST 5: Hypertable Configuration")
    logger.info("="*60)
    
    try:
        # Check hypertables
        result = db_manager.execute_sql("""
            SELECT 
                hypertable_name,
                num_chunks,
                num_dimensions
            FROM timescaledb_information.hypertables
            ORDER BY hypertable_name;
        """)
        
        hypertables = {row[0]: row[1] for row in result}
        
        expected_tables = ['market_data', 'trading_signals', 'trades']
        
        for table in expected_tables:
            if table in hypertables:
                logger.success(f"✓ {table} is a hypertable (chunks: {hypertables[table]})")
            else:
                logger.error(f"✗ {table} is NOT a hypertable")
                return False
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Hypertable test failed: {e}")
        return False


def test_time_queries():
    """Test time-based queries (TimescaleDB specialty)"""
    logger.info("\n" + "="*60)
    logger.info("TEST 6: Time-Series Queries")
    logger.info("="*60)
    
    try:
        with db_manager.get_session() as session:
            # Query recent data
            one_hour_ago = datetime.utcnow() - timedelta(hours=1)
            
            result = session.query(MarketData).filter(
                MarketData.timestamp >= one_hour_ago,
                MarketData.symbol == 'AAPL'
            ).order_by(MarketData.timestamp.desc()).all()
            
            logger.success(f"✓ Time-range query returned {len(result)} records")
            
            # Test aggregation query
            query = """
                SELECT 
                    time_bucket('1 hour', timestamp) as hour,
                    symbol,
                    count(*) as num_bars,
                    avg(close) as avg_price
                FROM market_data
                WHERE symbol = 'AAPL' AND source = 'test'
                GROUP BY hour, symbol
                ORDER BY hour DESC
                LIMIT 5;
            """
            
            result = db_manager.execute_sql(query)
            rows = result.fetchall()
            
            logger.success(f"✓ Time-bucket aggregation returned {len(rows)} buckets")
            
            if rows:
                logger.info(f"  Sample bucket: {rows[0][0]} - Avg price: ${rows[0][3]:.2f}")
            
            return True
            
    except Exception as e:
        logger.error(f"✗ Time-series query test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def test_continuous_aggregates():
    """Test continuous aggregates"""
    logger.info("\n" + "="*60)
    logger.info("TEST 7: Continuous Aggregates")
    logger.info("="*60)
    
    try:
        # Check if continuous aggregates exist
        result = db_manager.execute_sql("""
            SELECT view_name 
            FROM timescaledb_information.continuous_aggregates;
        """)
        
        aggregates = [row[0] for row in result]
        
        if aggregates:
            logger.success(f"✓ Found {len(aggregates)} continuous aggregates:")
            for agg in aggregates:
                logger.info(f"  • {agg}")
            return True
        else:
            logger.warning("⚠ No continuous aggregates found (this is optional)")
            return True
            
    except Exception as e:
        logger.error(f"✗ Continuous aggregate test failed: {e}")
        return False


def test_database_performance():
    """Test database performance with bulk inserts"""
    logger.info("\n" + "="*60)
    logger.info("TEST 8: Database Performance")
    logger.info("="*60)
    
    try:
        import time
        
        with db_manager.get_session() as session:
            start_time = time.time()
            
            # Bulk insert 1000 records
            records = []
            now = datetime.utcnow()
            
            for i in range(1000):
                record = MarketData(
                    timestamp=now - timedelta(minutes=i),
                    symbol='PERF_TEST',
                    open=100.0,
                    high=101.0,
                    low=99.0,
                    close=100.5,
                    volume=1000000,
                    interval='1m',
                    source='performance_test'
                )
                records.append(record)
            
            session.bulk_save_objects(records)
            session.commit()
            
            elapsed = time.time() - start_time
            rate = 1000 / elapsed
            
            logger.success(f"✓ Inserted 1000 records in {elapsed:.2f}s ({rate:.0f} records/sec)")
            
            # Clean up test data
            session.query(MarketData).filter(
                MarketData.source == 'performance_test'
            ).delete()
            session.commit()
            
            return True
            
    except Exception as e:
        logger.error(f"✗ Performance test failed: {e}")
        return False


def cleanup_test_data():
    """Clean up all test data"""
    logger.info("\n" + "="*60)
    logger.info("CLEANUP: Removing Test Data")
    logger.info("="*60)
    
    try:
        with db_manager.get_session() as session:
            # Delete test market data
            deleted_market = session.query(MarketData).filter(
                MarketData.source == 'test'
            ).delete()
            
            # Delete test signals
            deleted_signals = session.query(TradingSignal).filter(
                TradingSignal.extra_data['test'].astext == 'true'
            ).delete()
            
            # Delete test trades
            deleted_trades = session.query(Trade).filter(
                Trade.extra_data['test'].astext == 'true'
            ).delete()
            
            session.commit()
            
            logger.success(f"✓ Cleaned up {deleted_market} market data records")
            logger.success(f"✓ Cleaned up {deleted_signals} signals")
            logger.success(f"✓ Cleaned up {deleted_trades} trades")
            
            return True
            
    except Exception as e:
        logger.warning(f"⚠ Cleanup had issues: {e}")
        return True  # Don't fail on cleanup issues


def run_all_tests():
    """Run all database tests"""
    logger.info("\n" + "🧪" * 30)
    logger.info("RUNNING DATABASE TEST SUITE")
    logger.info("🧪" * 30)
    
    tests = [
        ("Database Connection", test_database_connection),
        ("Market Data Insertion", test_insert_market_data),
        ("Trading Signal Insertion", test_insert_trading_signal),
        ("Trade Insertion", test_insert_trade),
        ("Hypertable Configuration", test_hypertable_setup),
        ("Time-Series Queries", test_time_queries),
        ("Continuous Aggregates", test_continuous_aggregates),
        ("Database Performance", test_database_performance),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            logger.error(f"✗ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Cleanup
    cleanup_test_data()
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("TEST SUMMARY")
    logger.info("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"  {status}: {test_name}")
    
    logger.info("="*60)
    
    if passed == total:
        logger.success(f"\n🎉 All {total} tests passed!")
        logger.info("\n✅ Your database is ready for production!")
        logger.info("\nNext steps:")
        logger.info("  1. Start collecting market data")
        logger.info("  2. Build trading strategies")
        logger.info("  3. Run backtests")
        return True
    else:
        logger.error(f"\n❌ {total - passed} of {total} tests failed!")
        logger.info("\nPlease fix the issues above before continuing.")
        return False


if __name__ == "__main__":
    try:
        success = run_all_tests()
        sys.exit(0 if success else 1)
    finally:
        db_manager.close()
