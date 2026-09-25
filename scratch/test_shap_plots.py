import shap
import pandas as pd
import numpy as np
import yfinance as yf

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM
from sklearn.ensemble import RandomForestClassifier

sp500_close = yf.download("^GSPC", period="5y", interval="1d")["Close"]
tlt_close = yf.download("TLT", period="5y", interval="1d")["Close"]
gld_close = yf.download("GLD", period="5y", interval="1d")["Close"]

if isinstance(sp500_close, pd.DataFrame): sp500_close = sp500_close.squeeze()
if isinstance(tlt_close, pd.DataFrame): tlt_close = tlt_close.squeeze()
if isinstance(gld_close, pd.DataFrame): gld_close = gld_close.squeeze()

daily_return = sp500_close.pct_change()
vol_20 = daily_return.rolling(window=20).std()
mom_20 = sp500_close.pct_change(periods=20)
peak = sp500_close.cummax()
drawdown = (sp500_close - peak) / peak

tlt_return = tlt_close.pct_change()
gld_return = gld_close.pct_change()

corr_sp_tlt = daily_return.rolling(window=20).corr(tlt_return)
corr_sp_gld = daily_return.rolling(window=20).corr(gld_return)

feature_names = [
    "Daily_Return", "Rolling_Volatility_20", "Momentum_20",
    "Drawdown", "SP500_TLT_Corr_20", "SP500_GLD_Corr_20"
]

market_raw_df = pd.DataFrame({
    "Close": sp500_close,
    "Daily_Return": daily_return,
    "Rolling_Volatility_20": vol_20,
    "Momentum_20": mom_20,
    "Drawdown": drawdown,
    "SP500_TLT_Corr_20": corr_sp_tlt,
    "SP500_GLD_Corr_20": corr_sp_gld
})

market_clean_df = market_raw_df.dropna(subset=feature_names).copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(market_clean_df[feature_names])

hmm_model = GaussianHMM(n_components=4, covariance_type="full", n_iter=200, random_state=42)
hmm_model.fit(X_scaled)
market_clean_df["HMM_State"] = hmm_model.predict(X_scaled)

X = market_clean_df[feature_names]
y = market_clean_df["HMM_State"]

train_size = int(len(X) * 0.8)
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

rf_surrogate = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")
rf_surrogate.fit(X_train, y_train)

explainer = shap.TreeExplainer(rf_surrogate)
shap_exp = explainer(X_test)
print("SHAP Explanation shape:", shap_exp.shape)

# Test 1: Global Mean Absolute SHAP Importance per Feature Table
mean_abs_shap = np.abs(shap_exp.values).mean(axis=(0, 2))
global_importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Mean_Abs_SHAP": mean_abs_shap
}).sort_values(by="Mean_Abs_SHAP", ascending=False).reset_index(drop=True)
print("--- Global SHAP Importance Table ---")
print(global_importance_df)

# Test 2: Multi-class summary plot / beeswarm
print("Testing summary_plot (beeswarm)...")
plt.figure(figsize=(10, 6))
# For beeswarm across a specific state or list of states:
# shap_exp[:, :, state_k] gives an Explanation object for state_k
target_state = int(y_test.iloc[0])
print(f"Plotting beeswarm for State {target_state}...")
shap.plots.beeswarm(shap_exp[:, :, target_state], show=False)
plt.title(f"SHAP Beeswarm Plot for HMM State {target_state}")
plt.tight_layout()
plt.close()

# Test 3: SHAP Bar Plot for Global Importance / Per State
print("Testing bar plot...")
plt.figure(figsize=(8, 5))
shap.plots.bar(shap_exp[:, :, target_state], show=False)
plt.title(f"SHAP Feature Importance Bar Plot (State {target_state})")
plt.tight_layout()
plt.close()

# Test 4: Local Waterfall Plot for 1 Observation
print("Testing waterfall plot...")
sample_idx = 0
pred_state = rf_surrogate.predict(X_test.iloc[[sample_idx]])[0]
actual_state = y_test.iloc[sample_idx]
print(f"Sample {sample_idx} on date {X_test.index[sample_idx].strftime('%Y-%m-%d')}: Actual={actual_state}, Pred={pred_state}")

plt.figure(figsize=(8, 5))
shap.plots.waterfall(shap_exp[sample_idx, :, pred_state], show=False)
plt.title(f"Local SHAP Explanation for Date {X_test.index[sample_idx].strftime('%Y-%m-%d')} (Predicted State {pred_state})")
plt.tight_layout()
plt.close()

print("ALL SHAP PLOTS TESTED SUCCESSFULLY!")
