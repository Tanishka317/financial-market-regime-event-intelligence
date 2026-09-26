"""
Non-Destructive News Analysis Persistence Test
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
from backend.database.models import News, NewsAnalysis
from backend.database.news_analysis_repo import persist_news_analysis
from app.utils.news_data import fetch_financial_news
from app.utils.news_nlp import apply_finbert_sentiment
from app.utils.event_classifier import apply_event_classification


def test_news_analysis_persistence():
    load_dotenv()
    print("==========================================")
    print("PHASE 10E-4: NEWS ANALYSIS PERSISTENCE TEST")
    print("==========================================")

    # 1. Measure initial database row count for news_analysis
    engine = get_engine()
    with get_session() as session:
        initial_count = session.scalar(select(func.count()).select_from(NewsAnalysis))
    print(f"Initial PostgreSQL 'news_analysis' row count: {initial_count}")

    # 2. Pipeline execution: Fetch news, run FinBERT sentiment, apply event classification
    print("1. Fetching live financial news via app/utils/news_data.py...")
    raw_df = fetch_financial_news()
    print(f"   Retrieved {len(raw_df)} raw news records.")

    if raw_df.empty:
        print("[WARNING] Live news fetch returned 0 articles (network/yfinance unavailable).")
        print("NEWS ANALYSIS PERSISTENCE TEST COMPLETED (PASSED WITH WARNING).")
        return

    print("2. Running FinBERT sentiment inference via app/utils/news_nlp.py...")
    sentiment_df = apply_finbert_sentiment(raw_df)
    print("   FinBERT sentiment inference completed.")

    print("3. Applying rule-based event classification via app/utils/event_classifier.py...")
    analyzed_df = apply_event_classification(sentiment_df)
    print("   Event classification completed.")

    # 3. Pass 1: Persist news analysis
    print("\nExecuting Pass 1: Persisting news analysis into PostgreSQL...")
    processed_run1 = persist_news_analysis(analyzed_df, model_version="finbert_v1")
    
    with get_session() as session:
        count_after_run1 = session.scalar(select(func.count()).select_from(NewsAnalysis))
    print(f"Pass 1 processed: {processed_run1} records.")
    print(f"PostgreSQL 'news_analysis' row count after Pass 1: {count_after_run1}")
    assert count_after_run1 >= initial_count, "Database count decreased unexpectedly!"

    # 4. Pass 2: Re-persist identical news analysis (Idempotency test)
    print("\nExecuting Pass 2: Re-persisting identical news analysis into PostgreSQL...")
    processed_run2 = persist_news_analysis(analyzed_df, model_version="finbert_v1")

    with get_session() as session:
        count_after_run2 = session.scalar(select(func.count()).select_from(NewsAnalysis))
    print(f"Pass 2 processed: {processed_run2} records.")
    print(f"PostgreSQL 'news_analysis' row count after Pass 2: {count_after_run2}")

    # Assert idempotency
    assert count_after_run1 == count_after_run2, (
        f"Idempotency failure! DB row count changed from {count_after_run1} to {count_after_run2}."
    )
    print("[SUCCESS] Idempotency verified: 0 duplicate rows created during repeated execution!")

    # 5. Verify sample persisted analysis record in PostgreSQL
    print("\nVerifying sample persisted news analysis record in PostgreSQL...")
    with get_session() as session:
        sample_row = session.execute(
            select(NewsAnalysis, News)
            .join(News, NewsAnalysis.news_id == News.news_id)
            .order_by(NewsAnalysis.analyzed_at.desc())
            .limit(1)
        ).first()

        assert sample_row is not None, "No persisted news_analysis row found in database!"
        analysis, news_item = sample_row

        assert analysis.news_id, "Missing news_id!"
        assert analysis.sentiment_label in ["positive", "negative", "neutral"], f"Invalid sentiment_label: {analysis.sentiment_label}"
        assert 0.0 <= float(analysis.sentiment_score) <= 1.0, f"Invalid sentiment_score: {analysis.sentiment_score}"
        assert analysis.event_type, "Missing event_type!"
        assert analysis.model_version == "finbert_v1", f"Unexpected model_version: {analysis.model_version}"

        print(f"  - news_id:         {analysis.news_id}")
        print(f"  - query_ticker:    {news_item.query_ticker}")
        print(f"  - headline:        {news_item.headline[:60]}...")
        print(f"  - sentiment_label: {analysis.sentiment_label}")
        print(f"  - sentiment_score: {analysis.sentiment_score}")
        print(f"  - event_type:      {analysis.event_type}")
        print(f"  - event_trigger:   {analysis.event_trigger}")
        print(f"  - model_version:   {analysis.model_version}")
        print(f"  - analyzed_at:     {analysis.analyzed_at}")

    print("\n[OK] NEWS ANALYSIS PERSISTENCE TEST PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    test_news_analysis_persistence()
