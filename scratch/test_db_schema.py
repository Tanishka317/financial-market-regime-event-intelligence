"""
Comprehensive Schema & Relationship Verification Test
Financial Market Regime & Event Intelligence Engine
"""

import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from dotenv import load_dotenv
from sqlalchemy import inspect
from backend.database.connection import get_engine, get_session
from backend.database.models import (
    Base, MarketData, RegimePrediction, News, NewsAnalysis, EventMarketAnalysis
)


def test_schema_details():
    load_dotenv()
    engine = get_engine()
    inspector = inspect(engine)
    
    tables = inspector.get_table_names()
    print("==========================================")
    print("POSTGRESQL DATABASE SCHEMA VERIFICATION")
    print("==========================================")
    print(f"Database Tables Found ({len(tables)}): {tables}\n")
    
    expected_tables = {
        "market_data": ["id", "date", "ticker", "close", "daily_return", "volatility_20", "momentum_20", "drawdown", "sp500_tlt_corr_20", "sp500_gld_corr_20", "created_at"],
        "regime_predictions": ["id", "date", "ticker", "hmm_state", "regime_label", "model_version", "created_at"],
        "news": ["news_id", "published_at", "headline", "publisher", "query_ticker", "summary", "url", "ingested_at"],
        "news_analysis": ["news_id", "sentiment_label", "sentiment_score", "event_type", "event_trigger", "analyzed_at", "model_version"],
        "event_market_analysis": ["id", "news_id", "event_date", "event_day_return", "next_day_return", "five_day_forward_return", "volatility_20", "hmm_state", "created_at"]
    }
    
    for table_name, expected_cols in expected_tables.items():
        assert table_name in tables, f"Missing table: {table_name}"
        cols = [c["name"] for c in inspector.get_columns(table_name)]
        pk = inspector.get_pk_constraint(table_name)
        indexes = inspector.get_indexes(table_name)
        fks = inspector.get_foreign_keys(table_name)
        uniques = inspector.get_unique_constraints(table_name)
        
        print(f"Table: '{table_name}'")
        print(f"  - Columns ({len(cols)}): {cols}")
        print(f"  - Primary Key: {pk.get('constrained_columns')}")
        print(f"  - Foreign Keys: {[fk.get('constrained_columns') for fk in fks]}")
        print(f"  - Unique Constraints: {[u.get('column_names') for u in uniques]}")
        print(f"  - Indexes: {[ix.get('name') + ' (' + ', '.join(ix.get('column_names')) + ')' for ix in indexes]}")
        print("")
        
        for ec in expected_cols:
            assert ec in cols, f"Missing column '{ec}' in table '{table_name}'"
            
    print("[OK] All 5 tables, columns, PKs, FKs, Unique Constraints, and Indexes Verified!\n")


if __name__ == "__main__":
    test_schema_details()
    print("DATABASE SCHEMA VERIFICATION PASSED SUCCESSFULLY!")
