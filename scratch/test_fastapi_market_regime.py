"""
Market Data & Regime Predictions API Test Suite
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
from backend.database.models import MarketData, RegimePrediction


def test_fastapi_market_and_regimes():
    load_dotenv()
    print("==========================================")
    print("PHASE 10F-2A: MARKET & REGIME API TEST SUITE")
    print("==========================================")

    client = TestClient(app)
    print("FastAPI application and TestClient initialized successfully.\n")

    # 1. Measure initial database row counts
    with get_session() as session:
        initial_market_count = session.scalar(select(func.count()).select_from(MarketData))
        initial_regime_count = session.scalar(select(func.count()).select_from(RegimePrediction))
    print(f"Initial PostgreSQL 'market_data' row count:        {initial_market_count}")
    print(f"Initial PostgreSQL 'regime_predictions' row count: {initial_regime_count}\n")

    # 2. Test GET /api/market/history
    print("1. Testing GET /api/market/history?ticker=^GSPC&limit=5...")
    res_mkt_hist = client.get("/api/market/history?ticker=^GSPC&limit=5")
    print(f"   Response Status: {res_mkt_hist.status_code}")
    assert res_mkt_hist.status_code == 200, f"Expected 200, got {res_mkt_hist.status_code}"
    mkt_items = res_mkt_hist.json()
    assert len(mkt_items) == 5, f"Expected 5 records, got {len(mkt_items)}"

    # Check ticker & descending date order
    mkt_dates = [item["date"] for item in mkt_items]
    for item in mkt_items:
        assert item["ticker"] == "^GSPC", f"Expected ticker '^GSPC', got {item['ticker']}"
    assert mkt_dates == sorted(mkt_dates, reverse=True), f"Dates not in descending order: {mkt_dates}"
    print(f"   [OK] Returned {len(mkt_items)} '^GSPC' market records in descending date order: {mkt_dates[0]} -> {mkt_dates[-1]}")

    # 3. Test GET /api/regimes/latest
    print("\n2. Testing GET /api/regimes/latest?ticker=^GSPC...")
    res_reg_latest = client.get("/api/regimes/latest?ticker=^GSPC")
    print(f"   Response Status: {res_reg_latest.status_code}")
    assert res_reg_latest.status_code == 200, f"Expected 200, got {res_reg_latest.status_code}"
    reg_latest_data = res_reg_latest.json()
    print(f"   Response JSON:   {reg_latest_data}")
    assert reg_latest_data.get("ticker") == "^GSPC", f"Expected '^GSPC', got {reg_latest_data.get('ticker')}"
    assert "hmm_state" in reg_latest_data, "Missing 'hmm_state' field!"
    assert "regime_label" in reg_latest_data, "Missing 'regime_label' field!"
    assert "model_version" in reg_latest_data, "Missing 'model_version' field!"
    print(f"   [OK] Latest regime prediction for ^GSPC verified: State {reg_latest_data['hmm_state']} ({reg_latest_data['regime_label']})")

    # 4. Test GET /api/regimes
    print("\n3. Testing GET /api/regimes?ticker=^GSPC&limit=5...")
    res_reg_hist = client.get("/api/regimes?ticker=^GSPC&limit=5")
    print(f"   Response Status: {res_reg_hist.status_code}")
    assert res_reg_hist.status_code == 200, f"Expected 200, got {res_reg_hist.status_code}"
    reg_items = res_reg_hist.json()
    assert len(reg_items) == 5, f"Expected 5 records, got {len(reg_items)}"

    # Check ticker & descending date order
    reg_dates = [item["date"] for item in reg_items]
    for item in reg_items:
        assert item["ticker"] == "^GSPC", f"Expected ticker '^GSPC', got {item['ticker']}"
    assert reg_dates == sorted(reg_dates, reverse=True), f"Regime dates not in descending order: {reg_dates}"
    print(f"   [OK] Returned {len(reg_items)} '^GSPC' regime records in descending date order: {reg_dates[0]} -> {reg_dates[-1]}")

    # 5. Test 404 handling for invalid/non-existent ticker
    print("\n4. Testing HTTP 404 response for invalid/nonexistent ticker...")
    res_invalid_mkt = client.get("/api/market/history?ticker=NON_EXISTENT_999")
    print(f"   /api/market/history status for invalid ticker:  {res_invalid_mkt.status_code}")
    assert res_invalid_mkt.status_code == 404, f"Expected 404, got {res_invalid_mkt.status_code}"

    res_invalid_reg_latest = client.get("/api/regimes/latest?ticker=NON_EXISTENT_999")
    print(f"   /api/regimes/latest status for invalid ticker:   {res_invalid_reg_latest.status_code}")
    assert res_invalid_reg_latest.status_code == 404, f"Expected 404, got {res_invalid_reg_latest.status_code}"

    res_invalid_reg_hist = client.get("/api/regimes?ticker=NON_EXISTENT_999")
    print(f"   /api/regimes status for invalid ticker:          {res_invalid_reg_hist.status_code}")
    assert res_invalid_reg_hist.status_code == 404, f"Expected 404, got {res_invalid_reg_hist.status_code}"
    print("   [OK] HTTP 404 handling verified across all endpoints!")

    # 6. Verify row count immutability
    with get_session() as session:
        final_market_count = session.scalar(select(func.count()).select_from(MarketData))
        final_regime_count = session.scalar(select(func.count()).select_from(RegimePrediction))

    print(f"\nFinal PostgreSQL 'market_data' row count:        {final_market_count}")
    print(f"Final PostgreSQL 'regime_predictions' row count: {final_regime_count}")

    assert initial_market_count == final_market_count, "market_data row count changed!"
    assert initial_regime_count == final_regime_count, "regime_predictions row count changed!"
    print("[OK] Database row counts completely unchanged!")

    print("\n[OK] PHASE 10F-2A TEST SUITE PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    test_fastapi_market_and_regimes()
