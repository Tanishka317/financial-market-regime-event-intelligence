"""
Financial Analyst Service Expanded Natural Language Test Suite
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


def test_analyst_expanded_queries():
    load_dotenv()
    print("=========================================================")
    print("EXPANDED FINANCIAL ANALYST NATURAL LANGUAGE TEST SUITE")
    print("=========================================================\n")

    client = TestClient(app)

    # 1. Record initial database row counts
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

    # Test Cases: (Question, Expected Intent)
    test_cases = [
        # Original Base Queries
        ("What is the latest market regime?", "latest_regime"),
        ("What is the latest market data?", "latest_market_data"),
        ("What recent financial news/events are available?", "recent_news"),
        ("How did Inflation perform?", "event_performance"),
        ("What was the market regime during Monetary Policy events?", "event_regime"),
        
        # New Natural Language Regime Variations
        ("What's the current regime?", "latest_regime"),
        ("Which regime is the market in?", "latest_regime"),
        ("Tell me about the latest S&P 500 regime.", "latest_regime"),
        ("Show regime distribution", "regime_distribution"),
        ("How often is the market in a bull regime?", "regime_distribution"),
        ("Market performance by regime", "regime_market_behavior"),
        
        # New Natural Language Market Metric Variations
        ("How volatile has the market been?", "market_metrics"),
        ("What is the market momentum?", "market_metrics"),
        ("What is the maximum drawdown?", "market_metrics"),
        ("Show cross-asset correlations", "market_metrics"),
        
        # New Natural Language News & Sentiment Variations
        ("Show recent negative news.", "news_by_sentiment"),
        ("Positive news headlines", "news_by_sentiment"),
        ("News about inflation", "recent_news"),
        
        # New Natural Language Event Variations
        ("What happened after inflation events?", "event_performance"),
        ("Which events performed best?", "event_ranking"),
        ("How did monetary policy events behave during different regimes?", "event_regime"),
        
        # Out-of-Scope / Unsupported Variations
        ("What is tomorrow's price of Apple stock?", "unsupported"),
        ("Who won the football match?", "unsupported"),
    ]

    print("Executing Natural Language Query Tests...")
    for idx, (question, expected_intent) in enumerate(test_cases, 1):
        response = client.post("/api/analyst/query", json={"question": question})
        assert response.status_code == 200, f"HTTP failure on query '{question}': {response.status_code}"
        data = response.json()
        
        actual_intent = data.get("intent")
        answer = data.get("answer", "")
        
        assert actual_intent == expected_intent, (
            f"Test #{idx} Failed for '{question}': expected intent '{expected_intent}', got '{actual_intent}'"
        )
        assert answer, f"Test #{idx} Failed: Empty answer for '{question}'"
        assert "supporting_data" in data, f"Test #{idx} Failed: Missing supporting_data for '{question}'"

        print(f"  #{idx:02d} [OK] '{question}'\n       -> Intent: {actual_intent} | Answer: {answer[:95]}...")

    # 3. Verify Database Immutability
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

    print("\n[OK] All database row counts completely unchanged!")
    print("\n[OK] EXPANDED FINANCIAL ANALYST TEST SUITE PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    test_analyst_expanded_queries()
