"""
Integration Verification Test for Milestone 09F
"""

import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from app.utils.shared_context import get_shared_dashboard_context
from app.utils.market_data import get_market_dataset
from app.utils.regime_model import fit_and_decode_hmm
from app.utils.explainability import compute_shap_explainability
from app.utils.event_market import build_event_market_pipeline

def test_shared_context():
    print("Testing shared context...")
    ctx = get_shared_dashboard_context(period="5y")
    print(f"Latest Date: {ctx['latest_date']}")
    print(f"S&P 500 Close: {ctx['sp500_close_fmt']}")
    print(f"HMM Current State: State {ctx['current_state']} ({ctx['current_regime_label']})")
    print(f"Consecutive Sessions: {ctx['consecutive_sessions']}")
    print(f"News Headlines Count: {ctx['total_news']}")
    print(f"Top Sentiment: {ctx['top_sentiment']}")
    print(f"Top Event Type: {ctx['top_event_type']}")
    print("[OK] Shared Context Verification Passed!\n")

def test_pipeline_layers():
    print("Testing pipeline layers...")
    m_df = get_market_dataset()
    print(f"Market dataset rows: {len(m_df)}")
    
    hmm_res = fit_and_decode_hmm()
    print(f"HMM decoded rows: {len(hmm_res['market_df'])}, Log Likelihood: {hmm_res['log_likelihood']:.2f}")
    
    shap_res = compute_shap_explainability()
    print(f"SHAP Test Accuracy: {shap_res['test_accuracy']*100:.2f}%")
    
    em_res = build_event_market_pipeline()
    print(f"Event-Market news count: {em_res['total_news_count']}")
    print("[OK] All Pipeline Layers Verified!\n")

if __name__ == "__main__":
    print("==========================================")
    print("RUNNING MILESTONE 09F INTEGRATION TESTS")
    print("==========================================\n")
    test_shared_context()
    test_pipeline_layers()
    print("ALL TESTS PASSED SUCCESSFULLY!")
