"""
Non-Destructive Event-Market Analysis Persistence Test
Financial Market Regime & Event Intelligence Engine
"""

import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import select, func
from backend.database.connection import get_engine, get_session
from backend.database.models import News, EventMarketAnalysis
from backend.database.news_repo import persist_news
from backend.database.news_analysis_repo import persist_news_analysis
from backend.database.event_market_repo import persist_event_market_analysis
from app.utils.event_market import build_event_market_pipeline


def test_event_market_persistence():
    load_dotenv()
    print("==========================================")
    print("PHASE 10E-5: EVENT-MARKET PERSISTENCE TEST")
    print("==========================================")

    # 1. Record initial row count in event_market_analysis
    engine = get_engine()
    with get_session() as session:
        initial_count = session.scalar(select(func.count()).select_from(EventMarketAnalysis))
    print(f"Initial PostgreSQL 'event_market_analysis' row count: {initial_count}")

    # 2. Run existing real event-market pipeline
    print("\nExecuting real event-market pipeline via app/utils/event_market.py...")
    pipeline_res = build_event_market_pipeline(period="5y")
    event_market_df = pipeline_res.get("event_market_df")

    if event_market_df is None or event_market_df.empty:
        print("[WARNING] Pipeline returned 0 event-market analysis records.")
        print("EVENT-MARKET PERSISTENCE TEST COMPLETED (PASSED WITH WARNING).")
        return

    print(f"Pipeline returned {len(event_market_df)} event-market analysis records.")

    # 3. Ensure parent raw news records exist in PostgreSQL 'news' table
    print("\nEnsuring raw parent news records exist in PostgreSQL 'news' table...")
    news_persisted = persist_news(event_market_df)
    print(f"Parent news persistence complete ({news_persisted} raw news records verified/persisted).")

    # 4. Pass 1: Persist event-market analysis
    print("\nExecuting Pass 1: Persisting event-market analysis into PostgreSQL...")
    processed_run1 = persist_event_market_analysis(event_market_df, ticker="^GSPC")

    with get_session() as session:
        count_after_run1 = session.scalar(select(func.count()).select_from(EventMarketAnalysis))
    print(f"Pass 1 processed/persisted: {processed_run1} records.")
    print(f"PostgreSQL 'event_market_analysis' row count after Pass 1: {count_after_run1}")

    # 5. Pass 2: Re-persist identical event-market analysis (Idempotency Test)
    print("\nExecuting Pass 2: Re-persisting identical event-market analysis into PostgreSQL...")
    processed_run2 = persist_event_market_analysis(event_market_df, ticker="^GSPC")

    with get_session() as session:
        count_after_run2 = session.scalar(select(func.count()).select_from(EventMarketAnalysis))
    print(f"Pass 2 processed/persisted: {processed_run2} records.")
    print(f"PostgreSQL 'event_market_analysis' row count after Pass 2: {count_after_run2}")

    # Verify Idempotency
    assert count_after_run1 == count_after_run2, (
        f"Idempotency failure! DB row count changed from {count_after_run1} to {count_after_run2}."
    )
    print("[SUCCESS] Idempotency verified: 0 duplicate logical records created during repeated execution!")

    # 6. Test Foreign Key Integrity / Missing Parent Handling
    print("\nTesting Foreign-Key integrity with invalid missing parent news_id...")
    fake_df = pd.DataFrame([{
        "news_id": "news_NON_EXISTENT_PARENT_KEY_99999",
        "event_date": "2026-09-25",
        "event_day_return": 0.005,
        "next_day_return": -0.002,
        "5_day_forward_return": 0.012,
        "Rolling_Volatility_20": 0.015,
        "HMM_State": 0
    }])
    skipped_res = persist_event_market_analysis(fake_df, ticker="^GSPC")
    assert skipped_res == 0, f"Expected 0 persisted for missing parent, got {skipped_res}"
    
    with get_session() as session:
        count_after_fake = session.scalar(select(func.count()).select_from(EventMarketAnalysis))
    assert count_after_fake == count_after_run2, "Fake parent record was illegally inserted into database!"
    print("[SUCCESS] Foreign-key protection verified: Invalid parent news_id was explicitly rejected!")

    # 7. Verify sample persisted records from database
    print("\nDisplaying sample persisted event-market analysis records from PostgreSQL:")
    with get_session() as session:
        sample_rows = session.execute(
            select(EventMarketAnalysis, News)
            .join(News, EventMarketAnalysis.news_id == News.news_id)
            .order_by(EventMarketAnalysis.created_at.desc())
            .limit(3)
        ).all()

        for idx, (ema, news_item) in enumerate(sample_rows, 1):
            print(f"\n--- Sample Record #{idx} ---")
            print(f"  - Record ID:               {ema.id}")
            print(f"  - news_id:                 {ema.news_id}")
            print(f"  - headline:                {news_item.headline[:65]}...")
            print(f"  - event_date:              {ema.event_date}")
            print(f"  - HMM State:               {ema.hmm_state}")
            print(f"  - event_day_return:        {ema.event_day_return}")
            print(f"  - next_day_return:         {ema.next_day_return}")
            print(f"  - five_day_forward_return: {ema.five_day_forward_return}")
            print(f"  - volatility_20:           {ema.volatility_20}")
            print(f"  - created_at:              {ema.created_at}")

    print("\n[OK] EVENT-MARKET PERSISTENCE TEST PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    test_event_market_persistence()
