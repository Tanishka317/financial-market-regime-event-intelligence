import sys
import os
sys.path.insert(0, os.getcwd())

from app.utils.regime_model import fit_and_decode_hmm
from app.utils.charts import create_hmm_regime_timeline_chart, create_transition_matrix_heatmap

print("Testing Market Regimes backend execution...")
res = fit_and_decode_hmm(period="5y")

print("HMM Model Decoded Results:")
print(f"  Latest Date: {res['latest_date']}")
print(f"  Latest State: State {res['latest_state']} ({res['latest_label']})")
print(f"  Consecutive Sessions: {res['consecutive_days']}")
print(f"  Log-Likelihood: {res['log_likelihood']:.2f}")

print("\nGenerating HMM Charts...")
fig_timeline = create_hmm_regime_timeline_chart(res["market_df"])
fig_transmat = create_transition_matrix_heatmap(res["transmat"])

print("Market Regimes backend data and chart pipeline verified successfully!")
