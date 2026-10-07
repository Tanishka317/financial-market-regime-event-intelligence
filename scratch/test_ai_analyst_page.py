"""
AI Analyst Streamlit Page Smoke & Integration Test Suite
Financial Market Regime & Event Intelligence Engine
"""

import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from dotenv import load_dotenv
from fastapi.testclient import TestClient

from backend.api.main import app as fastapi_app
from backend.database.connection import get_session
from app.components.sidebar import get_pages
from app.pages import ai_analyst


def test_ai_analyst_page_integration():
    load_dotenv()
    print("==========================================")
    print("AI ANALYST STREAMLIT PAGE SMOKE TEST")
    print("==========================================")

    # 1. Verify Page Registry
    print("1. Verifying Streamlit Navigation Page Registry...")
    pages = get_pages()
    print(f"   Registered Pages ({len(pages)}): {list(pages.keys())}")
    
    expected_pages = [
        "Overview", "Market Regimes", "News & Events",
        "Explainability", "Model & Data Health", "AI Analyst"
    ]
    for page_name in expected_pages:
        assert page_name in pages, f"Missing registered page: '{page_name}'"
    print("   [OK] Navigation registry contains all 6 expected Streamlit pages!\n")

    # 2. Verify AI Analyst page module structure
    print("2. Verifying AI Analyst page module and functions...")
    assert hasattr(ai_analyst, "render"), "ai_analyst module missing render() function!"
    assert hasattr(ai_analyst, "query_analyst_api"), "ai_analyst module missing query_analyst_api() function!"
    print("   [OK] ai_analyst page render and API query functions verified!\n")

    # 3. Test API integration between Streamlit query handler and FastAPI backend
    print("3. Testing Streamlit page query handler against FastAPI backend via TestClient...")
    client = TestClient(fastapi_app)

    # Simulate query call made by Streamlit page
    test_questions = [
        ("What is the latest market regime?", "latest_regime"),
        ("What is the latest market data?", "latest_market_data"),
        ("What recent financial news/events are available?", "recent_news"),
        ("How did Inflation perform?", "event_performance"),
        ("What was the market regime during Monetary Policy events?", "event_regime"),
    ]

    for q, expected_intent in test_questions:
        response = client.post("/api/analyst/query", json={"question": q, "ticker": "^GSPC"})
        assert response.status_code == 200, f"Query failed for '{q}': status {response.status_code}"
        data = response.json()
        assert data.get("intent") == expected_intent, f"Intent mismatch for '{q}': expected '{expected_intent}', got '{data.get('intent')}'"
        assert data.get("answer"), f"Missing answer for '{q}'"
        assert "supporting_data" in data, f"Missing supporting_data for '{q}'"
        print(f"   [OK] '{q}' -> Intent: {data['intent']} | Answer length: {len(data['answer'])} chars")

    print("\n[OK] AI ANALYST STREAMLIT PAGE SMOKE TEST PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    test_ai_analyst_page_integration()
