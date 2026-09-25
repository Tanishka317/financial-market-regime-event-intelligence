import sys
import os
sys.path.insert(0, os.getcwd())

from app.utils.regime_model import fit_and_decode_hmm

print("Testing fit_and_decode_hmm...")
res = fit_and_decode_hmm()

print(f"Latest Date       : {res['latest_date']}")
print(f"Latest State      : State {res['latest_state']} ({res['latest_label']})")
print(f"Consecutive Days  : {res['consecutive_days']}")
print(f"Log-Likelihood    : {res['log_likelihood']:.2f}")

print("\n--- State Stats Table ---")
print(res['state_stats_df'][['state', 'count', 'percentage', 'label', 'mean_daily_return', 'mean_volatility_20', 'mean_drawdown']])

print("\n--- Transition Matrix (P_ij) ---")
print(res['transmat'].round(4))

print("\n--- State Durations (Days) ---")
print(res['duration_df'])
