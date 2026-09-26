"""
HMM Regime Prediction Persistence & Idempotency Verification Test (Non-Destructive)
Financial Market Regime & Event Intelligence Engine
"""

import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from dotenv import load_dotenv
from sqlalchemy import func
from app.utils.regime_model import fit_and_decode_hmm
from backend.database.connection import get_session
from backend.database.regime_repo import persist_regime_predictions
from backend.database.models import RegimePrediction


def test_regime_persistence():
    load_dotenv()
    print("==================================================")
    print("HMM REGIME PREDICTION PERSISTENCE TEST")
    print("==================================================\n")

    # 1. Fetch real HMM decoded predictions from regime_model pipeline
    print("1. Executing real Gaussian HMM regime pipeline...")
    hmm_res = fit_and_decode_hmm(period="5y")
    market_df = hmm_res["market_df"]
    df_len = len(market_df)
    latest_state = hmm_res["latest_state"]
    latest_label = hmm_res["latest_label"]
    
    print(f"   [OK] HMM pipeline executed successfully ({df_len:,} sessions decoded).")
    print(f"   Latest decoded state: State {latest_state} ({latest_label}) as of {hmm_res['latest_date']}\n")

    # 2. Record initial DB row count before test
    session = get_session()
    initial_db_count = session.query(func.count(RegimePrediction.id)).scalar()
    session.close()
    print(f"2. Initial PostgreSQL 'regime_predictions' table row count: {initial_db_count:,}")

    # 3. Run FIRST persistence operation
    print("3. Executing FIRST persistence operation (model_version: 'hmm_v1')...")
    count_1 = persist_regime_predictions(hmm_res, ticker="^GSPC", model_version="hmm_v1")
    print(f"   Processed {count_1:,} regime prediction records in Run 1.")

    session = get_session()
    db_count_run_1 = session.query(func.count(RegimePrediction.id)).scalar()
    session.close()
    print(f"   PostgreSQL 'regime_predictions' row count after Run 1: {db_count_run_1:,}\n")

    # 4. Run SECOND persistence operation (Idempotency Check)
    print("4. Executing SECOND persistence operation (Idempotency Check)...")
    count_2 = persist_regime_predictions(hmm_res, ticker="^GSPC", model_version="hmm_v1")
    print(f"   Processed {count_2:,} regime prediction records in Run 2.")

    session = get_session()
    db_count_run_2 = session.query(func.count(RegimePrediction.id)).scalar()
    
    # Query latest sample row from PostgreSQL to verify persisted fields
    sample_row = session.query(RegimePrediction).filter_by(ticker="^GSPC", model_version="hmm_v1").order_by(RegimePrediction.date.desc()).first()
    session.close()

    print(f"   PostgreSQL 'regime_predictions' row count after Run 2: {db_count_run_2:,}")

    # 5. Verify row count did not increase during Run 2
    assert db_count_run_2 == db_count_run_1, f"Idempotency failed! DB count changed from {db_count_run_1} to {db_count_run_2}"
    print("   [OK] Idempotency verified! Zero duplicate rows created during second ingestion.\n")

    # 6. Verify sample persisted row fields
    assert sample_row is not None, "Failed to retrieve sample persisted row from PostgreSQL!"
    print("5. Sample Persisted Regime Row Verification:")
    print(f"   - Ticker: {sample_row.ticker}")
    print(f"   - Date: {sample_row.date}")
    print(f"   - HMM State: {sample_row.hmm_state}")
    print(f"   - Regime Label: {sample_row.regime_label}")
    print(f"   - Model Version: {sample_row.model_version}")
    print(f"   - Created At: {sample_row.created_at}\n")

    assert sample_row.hmm_state == latest_state, f"Mismatch in state! Expected {latest_state}, got {sample_row.hmm_state}"
    assert sample_row.regime_label == latest_label, f"Mismatch in label! Expected {latest_label}, got {sample_row.regime_label}"
    print("[OK] Persisted state and regime label match HMM pipeline output exactly!\n")

    print("ALL HMM REGIME PERSISTENCE TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    test_regime_persistence()
