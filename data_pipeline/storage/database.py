"""
Database connection and management for TimescaleDB
Handles connections, sessions, and basic database operations
"""
from sqlalchemy import create_engine, text, MetaData
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.pool import QueuePool
from contextlib import contextmanager
from loguru import logger
import os
from dotenv import load_dotenv

load_dotenv()

# Database configuration from environment variables
DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = os.getenv('DB_PORT', '5432')
DB_NAME = os.getenv('DB_NAME', 'trading_data')
DB_USER = os.getenv('DB_USER', 'trader')
DB_PASSWORD = os.getenv('DB_PASSWORD', 'trading123')

# Create database URL
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# SQLAlchemy setup
Base = declarative_base()
metadata = MetaData()


class DatabaseManager:
    """Manages database connections and operations"""
    
    def __init__(self):
        self.engine = None
        self.SessionLocal = None
        
    def initialize(self):
        """Initialize database connection with connection pooling"""
        try:
            self.engine = create_engine(
                DATABASE_URL,
                poolclass=QueuePool,
                pool_size=5,
                max_overflow=10,
                pool_pre_ping=True,  # Verify connections before using
                echo=False  # Set to True for SQL debugging
            )
            
            self.SessionLocal = sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self.engine
            )
            
            # Test connection
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT version();"))
                version = result.fetchone()[0]
                logger.info(f"Connected to database: {version}")
                
                # Enable TimescaleDB extension
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;"))
                conn.commit()
                logger.success("TimescaleDB extension enabled")
                
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            return False
    
    @contextmanager
    def get_session(self):
        """
        Get database session with automatic cleanup
        Usage:
            with db_manager.get_session() as session:
                session.query(...)
        """
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Session error: {e}")
            raise
        finally:
            session.close()
    
    def execute_sql(self, sql: str, params: dict = None):
        """
        Execute raw SQL query
        
        Args:
            sql: SQL query string
            params: Dictionary of parameters for the query
            
        Returns:
            Result object from SQLAlchemy
        """
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(sql), params or {})
                conn.commit()
                return result
        except Exception as e:
            logger.error(f"SQL execution error: {e}")
            raise
    
    def create_tables(self):
        """Create all tables defined in models"""
        Base.metadata.create_all(bind=self.engine)
        logger.info("Tables created successfully")
    
    def drop_tables(self):
        """Drop all tables (use with caution!)"""
        Base.metadata.drop_all(bind=self.engine)
        logger.warning("All tables dropped")
    
    def close(self):
        """Close database connections"""
        if self.engine:
            self.engine.dispose()
            logger.info("Database connections closed")
    
    def get_table_info(self, table_name: str):
        """Get information about a table"""
        query = """
        SELECT 
            column_name, 
            data_type, 
            is_nullable
        FROM information_schema.columns
        WHERE table_name = :table_name
        ORDER BY ordinal_position;
        """
        result = self.execute_sql(query, {'table_name': table_name})
        return result.fetchall()
    
    def get_table_size(self, table_name: str):
        """Get the size of a table"""
        query = f"SELECT pg_size_pretty(pg_total_relation_size('{table_name}'));"
        result = self.execute_sql(query)
        return result.fetchone()[0]


# Global database instance
db_manager = DatabaseManager()


# Test the connection when run directly
if __name__ == "__main__":
    logger.info("Testing database connection...")
    
    if db_manager.initialize():
        logger.success("✓ Database connection successful!")
        
        # Test query
        result = db_manager.execute_sql("SELECT NOW();")
        current_time = result.fetchone()[0]
        logger.info(f"Current database time: {current_time}")
        
        # Test TimescaleDB
        result = db_manager.execute_sql("SELECT extversion FROM pg_extension WHERE extname='timescaledb';")
        version = result.fetchone()
        if version:
            logger.success(f"✓ TimescaleDB version: {version[0]}")
        
        db_manager.close()
    else:
        logger.error("✗ Database connection failed!")
