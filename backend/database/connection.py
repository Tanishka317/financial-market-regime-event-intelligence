"""
Database Connection & Engine Utility
Financial Market Regime & Event Intelligence Engine
"""

import os
from typing import Optional
from dotenv import load_dotenv
from sqlalchemy import create_engine, Engine
from sqlalchemy.orm import sessionmaker, Session

# Load environment variables from .env if present
load_dotenv()

# Singleton engine and sessionmaker instances for the application process
_engine: Optional[Engine] = None
_SessionLocal: Optional[sessionmaker] = None


def get_database_url() -> str:
    """
    Retrieves the PostgreSQL database connection URL from the DATABASE_URL environment variable.
    Raises ValueError if DATABASE_URL is missing.
    """
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise ValueError(
            "DATABASE_URL environment variable is not set. "
            "Please configure DATABASE_URL in environment or .env file. "
            "Example: postgresql+psycopg://postgres:<password>@localhost:5432/financial_intelligence"
        )
    return db_url


def get_engine() -> Engine:
    """
    Returns the singleton SQLAlchemy Engine instance configured with pre-ping connection pool.
    Lazy-initializes the engine on first call and reuses it for all subsequent calls.
    """
    global _engine
    if _engine is None:
        db_url = get_database_url()
        _engine = create_engine(
            db_url,
            pool_pre_ping=True,
            pool_size=10,
            max_overflow=20
        )
    return _engine


def get_session() -> Session:
    """
    Returns a new SQLAlchemy Session instance bound to the shared singleton engine.
    """
    global _SessionLocal
    if _SessionLocal is None:
        engine = get_engine()
        _SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    return _SessionLocal()
