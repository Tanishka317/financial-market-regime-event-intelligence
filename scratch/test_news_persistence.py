"""
Non-Destructive Financial News Persistence Test
Financial Market Regime & Event Intelligence Engine
"""

import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from dotenv import load_dotenv
from sqlalchemy import select, func
from backend.database.connection import get_engine, get_session
from backend.database.models import News
from backend.database.news_repo import persist_news
from app.utils.news_data import fetch_financial_news


def test_raw_news_persistence():
    load_dotenv()
    print("==========================================")
    print("PHASE 10E-3: RAW NEWS PERSISTENCE TEST")
    print("==========================================")

    # 1. Measure initial database row count
    engine = get_engine()
    with get_session() as session:
        initial_count = session.scalar(select(func.count()).select_from(News))
    print(f"Initial PostgreSQL 'news' row count: {initial_count}")

    # 2. Fetch real news using existing news retrieval pipeline
    print("Fetching live financial news via app/utils/news_data.py...")
    news_df = fetch_financial_news()
    print(f"Retrieved {len(news_df)} raw news articles from yfinance.")

    if news_df.empty:
        print("[WARNING] Live news fetch returned 0 articles (network/yfinance unavailable).")
        print("Tolerating empty result as permitted by Requirement 15.")
        print("NEWS PERSISTENCE TEST COMPLETED (PASSED WITH WARNING).")
        return

    # Take the full fetched set
    sample_df = news_df.copy()
    sample_size = len(sample_df)
    print(f"Selected full batch of {sample_size} real news articles for persistence testing.")

    # 3. First persistence run
    print("\nExecuting Pass 1: Persisting news sample into PostgreSQL...")
    processed_run1 = persist_news(sample_df)
    
    with get_session() as session:
        count_after_run1 = session.scalar(select(func.count()).select_from(News))
    print(f"Pass 1 processed: {processed_run1} records.")
    print(f"PostgreSQL 'news' row count after Pass 1: {count_after_run1}")
    assert count_after_run1 >= initial_count, "Database count decreased unexpectedly!"

    # 4. Second persistence run (Idempotency test)
    print("\nExecuting Pass 2: Re-persisting identical news sample into PostgreSQL...")
    processed_run2 = persist_news(sample_df)

    with get_session() as session:
        count_after_run2 = session.scalar(select(func.count()).select_from(News))
    print(f"Pass 2 processed: {processed_run2} records.")
    print(f"PostgreSQL 'news' row count after Pass 2: {count_after_run2}")

    # Assert idempotency
    assert count_after_run1 == count_after_run2, (
        f"Idempotency failure! DB row count changed from {count_after_run1} to {count_after_run2}."
    )
    print("[SUCCESS] Idempotency verified: 0 duplicate rows created during repeated execution!")

    # 5. Verify sample row metadata in PostgreSQL
    print("\nVerifying sample persisted article metadata in PostgreSQL...")
    with get_session() as session:
        sample_row = session.execute(
            select(News).order_by(News.ingested_at.desc()).limit(1)
        ).scalar_one_or_none()

        assert sample_row is not None, "No persisted news row found in database!"
        assert sample_row.news_id, "Missing news_id!"
        assert sample_row.headline, "Missing headline!"
        assert sample_row.publisher, "Missing publisher!"
        assert sample_row.query_ticker, "Missing query_ticker!"
        assert sample_row.published_at, "Missing published_at!"

        print(f"  - news_id:      {sample_row.news_id}")
        print(f"  - query_ticker: {sample_row.query_ticker}")
        print(f"  - headline:     {sample_row.headline[:60]}...")
        print(f"  - publisher:    {sample_row.publisher}")
        print(f"  - published_at: {sample_row.published_at}")
        print(f"  - summary:      {sample_row.summary[:60] if sample_row.summary else None}...")
        print(f"  - url:          {sample_row.url}")
        print(f"  - ingested_at:  {sample_row.ingested_at}")

    print("\n[OK] RAW NEWS PERSISTENCE TEST PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    test_raw_news_persistence()
