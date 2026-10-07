"""
News, News Analysis, and Event-Market Analysis API Test Suite
Financial Market Regime & Event Intelligence Engine
"""

import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from dotenv import load_dotenv
from fastapi.testclient import TestClient
from sqlalchemy import select, func
from backend.api.main import app
from backend.database.connection import get_engine, get_session
from backend.database.models import MarketData, RegimePrediction, News, NewsAnalysis, EventMarketAnalysis


def test_fastapi_news_and_events():
    load_dotenv()
    print("==========================================")
    print("PHASE 10F-2B: NEWS & EVENTS API TEST SUITE")
    print("==========================================")

    client = TestClient(app)
    print("FastAPI application and TestClient initialized successfully.\n")

    # 1. Record initial database row counts across all tables
    with get_session() as session:
        initial_counts = {
            "market_data": session.scalar(select(func.count()).select_from(MarketData)),
            "regime_predictions": session.scalar(select(func.count()).select_from(RegimePrediction)),
            "news": session.scalar(select(func.count()).select_from(News)),
            "news_analysis": session.scalar(select(func.count()).select_from(NewsAnalysis)),
            "event_market_analysis": session.scalar(select(func.count()).select_from(EventMarketAnalysis)),
        }
        
        # Grab a sample valid news_id from database for endpoint testing
        sample_news = session.execute(select(News.news_id, News.query_ticker).limit(1)).first()
        sample_news_id = sample_news[0] if sample_news else None
        sample_ticker = sample_news[1] if sample_news else None
        
        # Grab a sample analyzed news_id
        sample_analysis = session.execute(select(NewsAnalysis.news_id, NewsAnalysis.event_type).limit(1)).first()
        sample_analyzed_id = sample_analysis[0] if sample_analysis else None
        sample_event_type = sample_analysis[1] if sample_analysis else None

        # Grab a sample HMM state from event_market_analysis
        sample_ema = session.execute(select(EventMarketAnalysis.hmm_state).limit(1)).first()
        sample_hmm_state = sample_ema[0] if sample_ema else 0

    print("Initial PostgreSQL Table Row Counts:")
    for table_name, count in initial_counts.items():
        print(f"  - {table_name}: {count}")
    print("")

    # 2. Test GET /api/news
    print("1. Testing GET /api/news?limit=5...")
    res_news = client.get("/api/news?limit=5")
    print(f"   Response Status: {res_news.status_code}")
    assert res_news.status_code == 200, f"Expected 200, got {res_news.status_code}"
    news_items = res_news.json()
    assert len(news_items) == 5, f"Expected 5 records, got {len(news_items)}"

    # Check published_at descending order
    news_dates = [item["published_at"] for item in news_items]
    assert news_dates == sorted(news_dates, reverse=True), f"News not in descending order: {news_dates}"
    print(f"   [OK] Returned {len(news_items)} news records in descending order: {news_dates[0]} -> {news_dates[-1]}")

    # 3. Test GET /api/news ticker filtering
    if sample_ticker:
        print(f"\n2. Testing GET /api/news ticker filter (ticker='{sample_ticker}')...")
        res_news_ticker = client.get(f"/api/news?ticker={sample_ticker}&limit=5")
        print(f"   Response Status: {res_news_ticker.status_code}")
        assert res_news_ticker.status_code == 200, f"Expected 200, got {res_news_ticker.status_code}"
        ticker_items = res_news_ticker.json()
        for item in ticker_items:
            assert item["query_ticker"] == sample_ticker, f"Expected ticker '{sample_ticker}', got '{item['query_ticker']}'"
        print(f"   [OK] Ticker filtering verified for '{sample_ticker}' ({len(ticker_items)} records returned).")

    # 4. Test GET /api/news/{news_id}/analysis
    if sample_analyzed_id:
        print(f"\n3. Testing GET /api/news/{{news_id}}/analysis for news_id='{sample_analyzed_id[:25]}...'")
        res_analysis = client.get(f"/api/news/{sample_analyzed_id}/analysis")
        print(f"   Response Status: {res_analysis.status_code}")
        assert res_analysis.status_code == 200, f"Expected 200, got {res_analysis.status_code}"
        analysis_data = res_analysis.json()
        assert analysis_data.get("news_id") == sample_analyzed_id, "news_id mismatch!"
        assert "sentiment_label" in analysis_data, "Missing sentiment_label!"
        assert "event_type" in analysis_data, "Missing event_type!"
        print(f"   [OK] Persisted analysis verified: Sentiment '{analysis_data['sentiment_label']}', Event '{analysis_data['event_type']}'")

    # 5. Test GET /api/event-market-analysis
    print("\n4. Testing GET /api/event-market-analysis?limit=5...")
    res_ema = client.get("/api/event-market-analysis?limit=5")
    print(f"   Response Status: {res_ema.status_code}")
    assert res_ema.status_code == 200, f"Expected 200, got {res_ema.status_code}"
    ema_items = res_ema.json()
    assert len(ema_items) == 5, f"Expected 5 records, got {len(ema_items)}"

    # Check event_date descending order & context fields
    ema_dates = [item["event_date"] for item in ema_items]
    assert ema_dates == sorted(ema_dates, reverse=True), f"Event dates not in descending order: {ema_dates}"
    for item in ema_items:
        assert "headline" in item, "Missing headline field in EMA response!"
        assert "sentiment_label" in item, "Missing sentiment_label in EMA response!"
        assert "event_type" in item, "Missing event_type in EMA response!"
    print(f"   [OK] Returned {len(ema_items)} event-market records in descending order with joined context.")

    # 6. Test GET /api/event-market-analysis filtering (event_type & hmm_state)
    if sample_event_type:
        print(f"\n5. Testing GET /api/event-market-analysis filtering (event_type='{sample_event_type}', hmm_state={sample_hmm_state})...")
        res_ema_filter = client.get(f"/api/event-market-analysis?event_type={sample_event_type}&hmm_state={sample_hmm_state}&limit=5")
        print(f"   Response Status: {res_ema_filter.status_code}")
        assert res_ema_filter.status_code == 200, f"Expected 200, got {res_ema_filter.status_code}"
        filtered_ema = res_ema_filter.json()
        for item in filtered_ema:
            assert item["event_type"] == sample_event_type, f"Expected event_type '{sample_event_type}', got '{item['event_type']}'"
            assert item["hmm_state"] == sample_hmm_state, f"Expected hmm_state {sample_hmm_state}, got {item['hmm_state']}"
        print(f"   [OK] Filtering by event_type & hmm_state verified ({len(filtered_ema)} records returned).")

    # 7. Test HTTP 404 responses for invalid parameters
    print("\n6. Testing HTTP 404 response handling...")
    res_invalid_news_id = client.get("/api/news/news_NON_EXISTENT_ID_9999/analysis")
    print(f"   /api/news/{{invalid_id}}/analysis status: {res_invalid_news_id.status_code}")
    assert res_invalid_news_id.status_code == 404, f"Expected 404, got {res_invalid_news_id.status_code}"

    res_invalid_ema_filter = client.get("/api/event-market-analysis?event_type=NON_EXISTENT_EVENT_TYPE_XYZ")
    print(f"   /api/event-market-analysis status for invalid filter: {res_invalid_ema_filter.status_code}")
    assert res_invalid_ema_filter.status_code == 404, f"Expected 404, got {res_invalid_ema_filter.status_code}"
    print("   [OK] HTTP 404 response handling verified for invalid requests!")

    # 8. Verify database row count immutability across all tables
    with get_session() as session:
        final_counts = {
            "market_data": session.scalar(select(func.count()).select_from(MarketData)),
            "regime_predictions": session.scalar(select(func.count()).select_from(RegimePrediction)),
            "news": session.scalar(select(func.count()).select_from(News)),
            "news_analysis": session.scalar(select(func.count()).select_from(NewsAnalysis)),
            "event_market_analysis": session.scalar(select(func.count()).select_from(EventMarketAnalysis)),
        }

    print("\nFinal PostgreSQL Table Row Counts:")
    for table_name, count in final_counts.items():
        print(f"  - {table_name}: {count}")
        assert count == initial_counts[table_name], f"Row count changed for table '{table_name}'!"

    print("[OK] All database row counts completely unchanged!")

    print("\n[OK] PHASE 10F-2B TEST SUITE PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    test_fastapi_news_and_events()
