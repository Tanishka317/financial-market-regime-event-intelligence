"""
FastAPI Backend Foundation Test
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
from backend.database.models import MarketData


def test_fastapi_foundation():
    load_dotenv()
    print("==========================================")
    print("PHASE 10F-1: FASTAPI FOUNDATION TEST")
    print("==========================================")

    # 1. Test app import & TestClient initialization
    client = TestClient(app)
    print("FastAPI application and TestClient imported successfully.")

    # 2. Test GET /api/health
    print("\nTesting GET /api/health...")
    response = client.get("/api/health")
    print(f"Response status: {response.status_code}")
    print(f"Response JSON:   {response.json()}")
    assert response.status_code == 200, f"Expected HTTP 200, got {response.status_code}"
    data = response.json()
    assert data.get("status") == "ok", f"Expected status 'ok', got {data.get('status')}"
    assert data.get("service") == "financial-intelligence-api", f"Unexpected service name: {data.get('service')}"
    print("[OK] Health endpoint check passed!")

    # 3. Record initial database market_data row count
    with get_session() as session:
        initial_count = session.scalar(select(func.count()).select_from(MarketData))
        latest_db_record = session.scalar(
            select(MarketData).where(MarketData.ticker == "^GSPC").order_by(MarketData.date.desc()).limit(1)
        )
    print(f"\nInitial PostgreSQL 'market_data' row count: {initial_count}")

    # 4. Test GET /api/market/latest
    print("Testing GET /api/market/latest...")
    response = client.get("/api/market/latest")
    print(f"Response status: {response.status_code}")

    if latest_db_record is not None:
        assert response.status_code == 200, f"Expected HTTP 200, got {response.status_code}"
        res_data = response.json()
        print(f"Response JSON:   {res_data}")

        assert res_data.get("ticker") == "^GSPC", f"Expected ticker '^GSPC', got {res_data.get('ticker')}"
        assert str(res_data.get("date")) == str(latest_db_record.date), (
            f"Expected date '{latest_db_record.date}', got '{res_data.get('date')}'"
        )
        assert res_data.get("close") == float(latest_db_record.close), "Close price mismatch!"
        print("[OK] Latest market data endpoint check passed!")
    else:
        print("No market_data records found in database. Expecting 404 response.")
        assert response.status_code == 404, f"Expected HTTP 404, got {response.status_code}"
        print("[OK] 404 Not Found response verified!")

    # 5. Verify database row count was NOT modified by read operations
    with get_session() as session:
        final_count = session.scalar(select(func.count()).select_from(MarketData))
    print(f"\nFinal PostgreSQL 'market_data' row count: {final_count}")
    assert initial_count == final_count, (
        f"Database modified! Initial count {initial_count} != Final count {final_count}"
    )
    print("[OK] Database row count unchanged after API calls!")

    print("\n[OK] FASTAPI FOUNDATION TEST PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    test_fastapi_foundation()
