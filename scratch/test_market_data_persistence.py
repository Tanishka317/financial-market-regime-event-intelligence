"""
Market Data Persistence & Idempotency Verification Test (Non-Destructive)
Financial Market Regime & Event Intelligence Engine
"""

import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from dotenv import load_dotenv
from sqlalchemy import func
from app.utils.market_data import get_market_dataset
from backend.database.connection import get_session
from backend.database.market_data_repo import persist_market_data
from backend.database.models import MarketData


def test_market_data_persistence():
    load_dotenv()
    print("==================================================")
    print("NON-DESTRUCTIVE MARKET DATA PERSISTENCE TEST")
    print("==================================================\n")

    # 1. Fetch real market dataset sample from pipeline
    print("1. Fetching real market dataset from pipeline...")
    clean_df = get_market_dataset(period="5y")
    total_len = len(clean_df)
    # Select a real 30-session sample subset for non-destructive persistence testing
    sample_df = clean_df.tail(30)
    sample_len = len(sample_df)
    
    print(f"   [OK] Retrieved real market dataset with {total_len:,} total rows.")
    print(f"   Testing persistence on real 30-session sample from {sample_df.index.min().strftime('%Y-%m-%d')} to {sample_df.index.max().strftime('%Y-%m-%d')}\n")

    # 2. Record initial DB row count before test
    session = get_session()
    initial_db_count = session.query(func.count(MarketData.id)).scalar()
    session.close()
    print(f"2. Initial PostgreSQL 'market_data' table row count: {initial_db_count:,}")

    # 3. Run FIRST persistence operation
    print("3. Executing FIRST persistence operation on sample...")
    count_1 = persist_market_data(sample_df, ticker="^GSPC")
    print(f"   Processed {count_1:,} sample records in Run 1.")

    session = get_session()
    db_count_run_1 = session.query(func.count(MarketData.id)).scalar()
    session.close()
    print(f"   PostgreSQL 'market_data' total row count after Run 1: {db_count_run_1:,}\n")

    # 4. Run SECOND persistence operation (Idempotency Check)
    print("4. Executing SECOND persistence operation on sample (Idempotency Check)...")
    count_2 = persist_market_data(sample_df, ticker="^GSPC")
    print(f"   Processed {count_2:,} sample records in Run 2.")

    session = get_session()
    db_count_run_2 = session.query(func.count(MarketData.id)).scalar()
    sample_row = session.query(MarketData).order_by(MarketData.date.desc()).first()
    session.close()

    print(f"   PostgreSQL 'market_data' total row count after Run 2: {db_count_run_2:,}")

    # 5. Verify row count did not increase during Run 2
    assert db_count_run_2 == db_count_run_1, f"Idempotency failed! DB row count changed from {db_count_run_1} to {db_count_run_2}"
    print("   [OK] Idempotency verified! Total row count did NOT increase during repeated execution.\n")

    print("5. Sample Persisted Row Verification:")
    print(f"   - Ticker: {sample_row.ticker}")
    print(f"   - Date: {sample_row.date}")
    print(f"   - Close: {sample_row.close}")
    print(f"   - Daily Return: {sample_row.daily_return}")
    print(f"   - 20D Volatility: {sample_row.volatility_20}")
    print(f"   - 20D Momentum: {sample_row.momentum_20}")
    print(f"   - Drawdown: {sample_row.drawdown}")
    print(f"   - S&P/TLT Corr: {sample_row.sp500_tlt_corr_20}")
    print(f"   - S&P/GLD Corr: {sample_row.sp500_gld_corr_20}")
    print(f"   - Created At: {sample_row.created_at}\n")

    print("NON-DESTRUCTIVE MARKET DATA PERSISTENCE TEST PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    test_market_data_persistence()
