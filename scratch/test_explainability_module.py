import sys
import os
sys.path.insert(0, os.getcwd())

from app.utils.explainability import compute_shap_explainability

print("Testing compute_shap_explainability...")
res = compute_shap_explainability(period="5y")

print(f"Test Accuracy            : {res['test_accuracy']:.4f}")
print(f"Macro F1 (Active States) : {res['macro_f1_active']:.4f}")
print(f"Macro F1 (All 4 States)  : {res['macro_f1_all']:.4f}")
print(f"Test Samples             : {res['test_samples']}")
print(f"Active Test States       : {res['active_test_states']}")
print(f"Test State Counts        : {res['test_state_counts']}")

print("\n--- Global SHAP Feature Importance ---")
print(res['global_shap_df'].to_string(index=False))

print(f"\n--- Latest Observation SHAP ({res['latest_date']}) ---")
print(f"Actual State    : State {res['actual_hmm_state']}")
print(f"Surrogate State : State {res['surrogate_pred_state']}")
print(f"Match           : {res['is_match']}")
print("\nLocal SHAP Contributions:")
print(res['local_shap_df'].to_string(index=False))
