"""
Database Initialization & Table Creation Script
Financial Market Regime & Event Intelligence Engine
"""

import sys
import os
import logging
from urllib.parse import urlparse

# Ensure workspace root directory is in python path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from dotenv import load_dotenv
from sqlalchemy import inspect, text
from backend.database.connection import get_engine, get_database_url
from backend.database.models import Base

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("init_db")


def mask_db_url(url_str: str) -> str:
    """Masks database password in connection string for safe log output."""
    try:
        parsed = urlparse(url_str)
        if parsed.password:
            netloc = parsed.netloc.replace(f":{parsed.password}@", ":****@")
            return parsed._replace(netloc=netloc).geturl()
        return url_str
    except Exception:
        return "...@localhost"


def init_db() -> bool:
    """
    Initializes PostgreSQL database schema by creating all 5 tables and indexes.
    Verifies connectivity, database reachability, and schema creation.
    """
    load_dotenv()
    
    db_url = get_database_url()
    logger.info(f"Target Database URL: {mask_db_url(db_url)}")
    
    engine = get_engine()
    
    # 1. Test basic connectivity & reachability
    logger.info("Testing PostgreSQL connectivity...")
    with engine.connect() as conn:
        ver_result = conn.execute(text("SELECT version();"))
        db_version = ver_result.scalar()
        logger.info(f"PostgreSQL connection successful! Server version: {db_version}")
        
        db_name_result = conn.execute(text("SELECT current_database();"))
        current_db = db_name_result.scalar()
        logger.info(f"Reachable database: '{current_db}'")

    # 2. Create tables using SQLAlchemy Metadata
    logger.info("Creating database schema (5 tables + indexes + constraints)...")
    Base.metadata.create_all(bind=engine)
    logger.info("SQLAlchemy Metadata create_all executed successfully.")

    # 3. Verify created tables using Inspector
    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())
    
    expected_tables = [
        "market_data",
        "regime_predictions",
        "news",
        "news_analysis",
        "event_market_analysis"
    ]
    
    logger.info(f"Discovered database tables: {sorted(list(existing_tables))}")
    
    missing = [t for t in expected_tables if t not in existing_tables]
    if missing:
        raise RuntimeError(f"Database table verification failed! Missing tables: {missing}")
        
    logger.info("SUCCESS: All 5 tables successfully created and verified in 'financial_intelligence'!")
    return True


if __name__ == "__main__":
    init_db()
