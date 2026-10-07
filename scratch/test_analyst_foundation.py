"""
Financial Analyst Service Foundation Test Suite
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
from backend.database.connection import get_session
from backend.database.models import MarketData, RegimePrediction, News, NewsAnalysis, EventMarketAnalysis


def test_analyst_foundation():
    load_dotenv()
    print("==========================================")
    print("FINANCIAL ANALYST FOUNDATION TEST SUITE")
    print("==========================================")

    client = TestClient(app)
    print("FastAPI application and TestClient initialized successfully.\n")

    # 1. Record initial database row counts across all 5 tables
    with get_session() as session:
        initial_counts = {
            "market_data": session.scalar(select(func.count()).select_from(MarketData)),
            "regime_predictions": session.scalar(select(func.count()).select_from(RegimePrediction)),
            "news": session.scalar(select(func.count()).select_from(News)),
            "news_analysis": session.scalar(select(func.count()).select_from(NewsAnalysis)),
            "event_market_analysis": session.scalar(select(func.count()).select_from(EventMarketAnalysis)),
        }

    print("Initial PostgreSQL Table Row Counts:")
    for table_name, count in initial_counts.items():
        print(f"  - {table_name}: {count}")
    print("")

    # 2. Test Question 1: Latest Market Regime
    q1 = "What is the latest market regime?"
    print(f"1. Testing POST /api/analyst/query ('{q1}')...")
    res1 = client.post("/api/analyst/query", json={"question": q1})
    print(f"   Status: {res1.status_code}")
    assert res1.status_code == 200, f"Expected 200, got {res1.status_code}"
    data1 = res1.json()
    print(f"   Intent: {data1.get('intent')}")
    print(f"   Answer: {data1.get('answer')}")
    assert data1.get("intent") == "latest_regime"
    assert "Low-Vol Bull" in data1.get("answer") or "HMM State" in data1.get("answer") or "regime" in data1.get("answer").lower()
    assert "supporting_data" in data1 and "hmm_state" in data1["supporting_data"]
    print("   [OK] Latest market regime query verified.\n")

    # 3. Test Question 2: Latest Market Data
    q2 = "What is the latest market data?"
    print(f"2. Testing POST /api/analyst/query ('{q2}')...")
    res2 = client.post("/api/analyst/query", json={"question": q2})
    print(f"   Status: {res2.status_code}")
    assert res2.status_code == 200, f"Expected 200, got {res2.status_code}"
    data2 = res2.json()
    print(f"   Intent: {data2.get('intent')}")
    print(f"   Answer: {data2.get('answer')}")
    assert data2.get("intent") == "latest_market_data"
    assert "close" in data2.get("supporting_data", {})
    assert "daily_return" in data2.get("supporting_data", {})
    print("   [OK] Latest market data query verified.\n")

    # 4. Test Question 3: Recent Financial News
    q3 = "What recent financial news/events are available?"
    print(f"3. Testing POST /api/analyst/query ('{q3}')...")
    res3 = client.post("/api/analyst/query", json={"question": q3})
    print(f"   Status: {res3.status_code}")
    assert res3.status_code == 200, f"Expected 200, got {res3.status_code}"
    data3 = res3.json()
    print(f"   Intent: {data3.get('intent')}")
    print(f"   Answer: {data3.get('answer')}")
    assert data3.get("intent") == "recent_news"
    assert "articles" in data3.get("supporting_data", {})
    print("   [OK] Recent news query verified.\n")

    # 5. Test Question 4: Event Performance
    q4 = "How did Inflation perform?"
    print(f"4. Testing POST /api/analyst/query ('{q4}')...")
    res4 = client.post("/api/analyst/query", json={"question": q4})
    print(f"   Status: {res4.status_code}")
    assert res4.status_code == 200, f"Expected 200, got {res4.status_code}"
    data4 = res4.json()
    print(f"   Intent: {data4.get('intent')}")
    print(f"   Answer: {data4.get('answer')}")
    assert data4.get("intent") == "event_performance"
    assert data4.get("supporting_data", {}).get("event_type") == "Inflation"
    print("   [OK] Event performance query verified.\n")

    # 6. Test Question 5: Market Regime During Event
    q5 = "What was the market regime during Monetary Policy events?"
    print(f"5. Testing POST /api/analyst/query ('{q5}')...")
    res5 = client.post("/api/analyst/query", json={"question": q5})
    print(f"   Status: {res5.status_code}")
    assert res5.status_code == 200, f"Expected 200, got {res5.status_code}"
    data5 = res5.json()
    print(f"   Intent: {data5.get('intent')}")
    print(f"   Answer: {data5.get('answer')}")
    assert data5.get("intent") == "event_regime"
    assert data5.get("supporting_data", {}).get("event_type") == "Monetary Policy"
    assert "state_breakdown" in data5.get("supporting_data", {})
    print("   [OK] Event regime query verified.\n")

    # 7. Test Unsupported Question
    q6 = "What is the capital of France?"
    print(f"6. Testing POST /api/analyst/query ('{q6}') [Unsupported Query]...")
    res6 = client.post("/api/analyst/query", json={"question": q6})
    print(f"   Status: {res6.status_code}")
    assert res6.status_code == 200, f"Expected 200, got {res6.status_code}"
    data6 = res6.json()
    print(f"   Intent: {data6.get('intent')}")
    print(f"   Answer: {data6.get('answer')}")
    assert data6.get("intent") == "unsupported"
    assert "could not match your question" in data6.get("answer").lower()
    print("   [OK] Unsupported query handling verified.\n")

    # 8. Verify database row count immutability across all tables
    with get_session() as session:
        final_counts = {
            "market_data": session.scalar(select(func.count()).select_from(MarketData)),
            "regime_predictions": session.scalar(select(func.count()).select_from(RegimePrediction)),
            "news": session.scalar(select(func.count()).select_from(News)),
            "news_analysis": session.scalar(select(func.count()).select_from(NewsAnalysis)),
            "event_market_analysis": session.scalar(select(func.count()).select_from(EventMarketAnalysis)),
        }

    print("Final PostgreSQL Table Row Counts:")
    for table_name, count in final_counts.items():
        print(f"  - {table_name}: {count}")
        assert count == initial_counts[table_name], f"Row count changed for table '{table_name}'!"

    print("\n[OK] All database row counts completely unchanged!")
    print("\n[OK] FINANCIAL ANALYST FOUNDATION TEST SUITE PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    test_analyst_foundation()
