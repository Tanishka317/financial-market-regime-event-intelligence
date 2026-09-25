import sys
import os
sys.path.insert(0, os.getcwd())

from app.utils.explainability import compute_shap_explainability
from app.utils.charts import create_global_shap_chart, create_local_shap_chart

print("Testing Explainability backend execution...")
res = compute_shap_explainability(period="5y")

print("SHAP Results:")
print(f"  Test Accuracy: {res['test_accuracy']:.4f}")
print(f"  Macro F1 (Active): {res['macro_f1_active']:.4f}")
print(f"  Latest Date: {res['latest_date']}")
print(f"  Actual HMM State: State {res['actual_hmm_state']}")
print(f"  Surrogate Pred: State {res['surrogate_pred_state']}")
print(f"  Match: {res['is_match']}")

print("\nGenerating SHAP Charts...")
fig1 = create_global_shap_chart(res["global_shap_df"])
fig2 = create_local_shap_chart(res["local_shap_df"], res["latest_date"], res["surrogate_pred_state"])

print("Explainability backend data and chart pipeline verified successfully!")
