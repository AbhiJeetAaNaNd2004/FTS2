from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool
from typing import Generator
import logging
from .settings import get_settings

settings = get_settings()

# Create database engine with connection pooling
engine = create_engine(
    settings.database_url,
    poolclass=QueuePool,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
    pool_recycle=3600,
    echo=settings.debug,
    connect_args={"options": "-c timezone=utc"}
)

# Create session maker
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create declarative base
Base = declarative_base()


def get_db() -> Generator:
    """
    Dependency to get database session.
    Yields a database session and ensures proper cleanup.
    """
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        db.rollback()
        logging.error(f"Database session error: {e}")
        raise e
    finally:
        db.close()


def create_tables():
    """Create all database tables."""
    try:
        Base.metadata.create_all(bind=engine)
        logging.info("Database tables created successfully")
    except Exception as e:
        logging.error(f"Error creating database tables: {e}")
        raise e


def drop_tables():
    """Drop all database tables (use with caution)."""
    try:
        Base.metadata.drop_all(bind=engine)
        logging.info("Database tables dropped successfully")
    except Exception as e:
        logging.error(f"Error dropping database tables: {e}")
        raise e


class DatabaseManager:
    """Enhanced database manager with better error handling and logging."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def get_session(self):
        """Get a new database session."""
        return SessionLocal()
    
    def close_session(self, session):
        """Close a database session safely."""
        try:
            session.close()
        except Exception as e:
            self.logger.error(f"Error closing database session: {e}")
    
    def health_check(self) -> bool:
        """Check if database connection is healthy."""
        try:
            with engine.connect() as conn:
                conn.execute("SELECT 1")
            return True
        except Exception as e:
            self.logger.error(f"Database health check failed: {e}")
            return False