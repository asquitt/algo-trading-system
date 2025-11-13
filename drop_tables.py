"""
Drop all existing tables to start fresh
Run this if you need to reset the database
"""
from data_pipeline.storage.database import db_manager
from loguru import logger

def drop_all_tables():
    """Drop all tables to start fresh"""
    logger.warning("⚠️  This will DROP ALL TABLES and DATA!")
    logger.info("Dropping tables in 3 seconds... (Ctrl+C to cancel)")
    
    import time
    time.sleep(3)
    
    try:
        db_manager.initialize()
        
        tables_to_drop = [
            'market_data_1h',  # Continuous aggregate
            'market_data_1d',  # Continuous aggregate
            'market_data',
            'trading_signals',
            'trades',
            'positions'
        ]
        
        for table in tables_to_drop:
            try:
                logger.info(f"Dropping {table}...")
                db_manager.execute_sql(f"DROP TABLE IF EXISTS {table} CASCADE;")
                logger.success(f"✓ Dropped {table}")
            except Exception as e:
                logger.warning(f"Could not drop {table}: {e}")
        
        # Drop views
        try:
            db_manager.execute_sql("DROP VIEW IF EXISTS latest_prices CASCADE;")
            logger.success("✓ Dropped views")
        except Exception as e:
            logger.warning(f"Could not drop views: {e}")
        
        logger.success("\n✓ All tables dropped successfully!")
        logger.info("Now run: python data_pipeline/storage/timescale_setup.py")
        
    except Exception as e:
        logger.error(f"Error dropping tables: {e}")
    finally:
        db_manager.close()

if __name__ == "__main__":
    drop_all_tables()
