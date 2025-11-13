"""
Setup TimescaleDB hypertables, continuous aggregates, and optimizations
Run this once after creating the database
"""
from sqlalchemy import text
from loguru import logger
from data_pipeline.storage.database import db_manager
from data_pipeline.storage.models import Base


def setup_timescaledb():
    """Setup TimescaleDB hypertables and optimizations"""
    
    logger.info("=" * 60)
    logger.info("Setting up TimescaleDB for Algorithmic Trading System")
    logger.info("=" * 60)
    
    try:
        # Initialize database
        logger.info("Step 1: Initializing database connection...")
        if not db_manager.initialize():
            logger.error("Failed to initialize database")
            return False
        logger.success("✓ Database connected")
        
        # Create tables
        logger.info("\nStep 2: Creating database tables...")
        Base.metadata.create_all(bind=db_manager.engine)
        logger.success("✓ Tables created")
        
        # Convert market_data to hypertable
        logger.info("\nStep 3: Converting market_data to hypertable...")
        try:
            db_manager.execute_sql("""
                SELECT create_hypertable(
                    'market_data', 
                    'timestamp',
                    chunk_time_interval => INTERVAL '1 day',
                    if_not_exists => TRUE
                );
            """)
            logger.success("✓ market_data hypertable created (1-day chunks)")
        except Exception as e:
            if "already a hypertable" in str(e):
                logger.info("  market_data is already a hypertable")
            else:
                raise
        
        # Convert trading_signals to hypertable
        logger.info("\nStep 4: Converting trading_signals to hypertable...")
        try:
            db_manager.execute_sql("""
                SELECT create_hypertable(
                    'trading_signals',
                    'timestamp',
                    chunk_time_interval => INTERVAL '1 day',
                    if_not_exists => TRUE
                );
            """)
            logger.success("✓ trading_signals hypertable created")
        except Exception as e:
            if "already a hypertable" in str(e):
                logger.info("  trading_signals is already a hypertable")
            else:
                raise
        
        # Convert trades to hypertable
        logger.info("\nStep 5: Converting trades to hypertable...")
        try:
            db_manager.execute_sql("""
                SELECT create_hypertable(
                    'trades',
                    'timestamp',
                    chunk_time_interval => INTERVAL '1 week',
                    if_not_exists => TRUE
                );
            """)
            logger.success("✓ trades hypertable created (1-week chunks)")
        except Exception as e:
            if "already a hypertable" in str(e):
                logger.info("  trades is already a hypertable")
            else:
                raise
        
        # Create continuous aggregate for 1-hour OHLCV data
        logger.info("\nStep 6: Creating continuous aggregates...")
        try:
            db_manager.execute_sql("""
                CREATE MATERIALIZED VIEW IF NOT EXISTS market_data_1h
                WITH (timescaledb.continuous) AS
                SELECT
                    time_bucket('1 hour', timestamp) AS bucket,
                    symbol,
                    first(open, timestamp) AS open,
                    max(high) AS high,
                    min(low) AS low,
                    last(close, timestamp) AS close,
                    sum(volume) AS volume,
                    count(*) AS num_bars
                FROM market_data
                WHERE interval = '1m'
                GROUP BY bucket, symbol
                WITH NO DATA;
            """)
            logger.success("✓ 1-hour continuous aggregate created")
        except Exception as e:
            if "already exists" in str(e):
                logger.info("  Continuous aggregate already exists")
            else:
                logger.warning(f"  Could not create 1h aggregate: {e}")
        
        # Add refresh policy for continuous aggregate
        logger.info("\nStep 7: Setting up automatic refresh policy...")
        try:
            db_manager.execute_sql("""
                SELECT add_continuous_aggregate_policy('market_data_1h',
                    start_offset => INTERVAL '3 hours',
                    end_offset => INTERVAL '1 hour',
                    schedule_interval => INTERVAL '1 hour',
                    if_not_exists => TRUE
                );
            """)
            logger.success("✓ Auto-refresh policy added (updates every hour)")
        except Exception as e:
            if "already exists" in str(e):
                logger.info("  Refresh policy already exists")
            else:
                logger.warning(f"  Could not add refresh policy: {e}")
        
        # Create daily aggregate
        logger.info("\nStep 8: Creating daily continuous aggregate...")
        try:
            db_manager.execute_sql("""
                CREATE MATERIALIZED VIEW IF NOT EXISTS market_data_1d
                WITH (timescaledb.continuous) AS
                SELECT
                    time_bucket('1 day', timestamp) AS bucket,
                    symbol,
                    first(open, timestamp) AS open,
                    max(high) AS high,
                    min(low) AS low,
                    last(close, timestamp) AS close,
                    sum(volume) AS volume,
                    avg(close) AS avg_price,
                    count(*) AS num_bars
                FROM market_data
                WHERE interval IN ('1m', '5m', '15m', '1h')
                GROUP BY bucket, symbol
                WITH NO DATA;
            """)
            logger.success("✓ Daily continuous aggregate created")
            
            # Add refresh policy
            db_manager.execute_sql("""
                SELECT add_continuous_aggregate_policy('market_data_1d',
                    start_offset => INTERVAL '3 days',
                    end_offset => INTERVAL '1 day',
                    schedule_interval => INTERVAL '1 day',
                    if_not_exists => TRUE
                );
            """)
            logger.success("✓ Daily refresh policy added")
        except Exception as e:
            if "already exists" in str(e):
                logger.info("  Daily aggregate already exists")
            else:
                logger.warning(f"  Could not create daily aggregate: {e}")
        
        # Create retention policy (keep data for 1 year)
        logger.info("\nStep 9: Setting up data retention policy...")
        try:
            db_manager.execute_sql("""
                SELECT add_retention_policy('market_data', 
                    INTERVAL '365 days',
                    if_not_exists => TRUE
                );
            """)
            logger.success("✓ Retention policy set (365 days)")
        except Exception as e:
            if "already exists" in str(e):
                logger.info("  Retention policy already exists")
            else:
                logger.warning(f"  Could not add retention policy: {e}")
        
        # Create compression policy for older data
        logger.info("\nStep 10: Setting up compression policy...")
        try:
            db_manager.execute_sql("""
                ALTER TABLE market_data SET (
                    timescaledb.compress,
                    timescaledb.compress_segmentby = 'symbol',
                    timescaledb.compress_orderby = 'timestamp DESC'
                );
            """)
            
            db_manager.execute_sql("""
                SELECT add_compression_policy('market_data', 
                    INTERVAL '7 days',
                    if_not_exists => TRUE
                );
            """)
            logger.success("✓ Compression policy set (compress after 7 days)")
        except Exception as e:
            if "already exists" in str(e) or "already compressed" in str(e):
                logger.info("  Compression already configured")
            else:
                logger.warning(f"  Could not set compression: {e}")
        
        # Create useful views
        logger.info("\nStep 11: Creating helper views...")
        try:
            db_manager.execute_sql("""
                CREATE OR REPLACE VIEW latest_prices AS
                SELECT DISTINCT ON (symbol)
                    symbol,
                    timestamp,
                    close as price,
                    volume,
                    source
                FROM market_data
                ORDER BY symbol, timestamp DESC;
            """)
            logger.success("✓ latest_prices view created")
        except Exception as e:
            logger.warning(f"  Could not create view: {e}")
        
        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("SETUP SUMMARY")
        logger.info("=" * 60)
        
        # Get hypertable info
        result = db_manager.execute_sql("""
            SELECT hypertable_name, num_chunks 
            FROM timescaledb_information.hypertables;
        """)
        
        logger.info("\nHypertables created:")
        for row in result:
            logger.info(f"  • {row[0]}: {row[1]} chunks")
        
        # Get continuous aggregates - using simpler query that works across versions
        try:
            result = db_manager.execute_sql("""
                SELECT view_name
                FROM timescaledb_information.continuous_aggregates;
            """)
            
            aggregates = [row[0] for row in result]
            
            if aggregates:
                logger.info("\nContinuous aggregates:")
                for agg in aggregates:
                    logger.info(f"  • {agg}")
        except Exception as e:
            logger.warning(f"Could not list continuous aggregates: {e}")
        
        logger.info("\n" + "=" * 60)
        logger.success("✓ TimescaleDB setup completed successfully!")
        logger.info("=" * 60)
        logger.info("\nYour database is now ready for:")
        logger.info("  • High-performance time-series queries")
        logger.info("  • Automatic data aggregation")
        logger.info("  • Efficient data compression")
        logger.info("  • Automatic data retention")
        logger.info("\n" + "=" * 60)
        
        return True
        
    except Exception as e:
        logger.error(f"\n✗ Error during setup: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False
    finally:
        db_manager.close()


def verify_setup():
    """Verify TimescaleDB setup"""
    logger.info("\nVerifying TimescaleDB setup...")
    
    try:
        db_manager.initialize()
        
        # Check hypertables
        result = db_manager.execute_sql("""
            SELECT hypertable_name FROM timescaledb_information.hypertables;
        """)
        hypertables = [row[0] for row in result]
        
        expected = ['market_data', 'trading_signals', 'trades']
        for table in expected:
            if table in hypertables:
                logger.success(f"✓ {table} is a hypertable")
            else:
                logger.error(f"✗ {table} is NOT a hypertable")
        
        return True
    except Exception as e:
        logger.error(f"Verification failed: {e}")
        return False
    finally:
        db_manager.close()


if __name__ == "__main__":
    # Run setup
    success = setup_timescaledb()
    
    if success:
        # Verify
        verify_setup()
        
        logger.info("\n🎉 Database is ready for trading data!")
        logger.info("\nNext steps:")
        logger.info("  1. Run: PYTHONPATH=. python3 tests/test_database.py")
        logger.info("  2. Start collecting market data")
        logger.info("  3. Build your trading strategies!")
    else:
        logger.error("\n❌ Database setup failed!")
        logger.error("Check the error messages above and try again.")
