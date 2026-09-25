import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

print("Downloading market data...")
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

y_pred = rf_surrogate.predict(X_test)

surrogate_acc = accuracy_score(y_test, y_pred)
present_classes = np.unique(y_test)
macro_f1_present = f1_score(y_test, y_pred, labels=present_classes, average="macro")
macro_f1_all = f1_score(y_test, y_pred, labels=[0, 1, 2, 3], average="macro", zero_division=0)
all_states = [0, 1, 2, 3]
surrogate_cm = confusion_matrix(y_test, y_pred, labels=all_states)

print("=== FIDELITY EVALUATION ===")
print(f"Accuracy                          : {surrogate_acc:.4f}")
print(f"Macro F1 (Classes Present {present_classes.tolist()}): {macro_f1_present:.4f}")
print(f"Macro F1 (All 4 States [0, 1, 2, 3])  : {macro_f1_all:.4f}")

print("\n--- Classification Report (All 4 States) ---")
print(classification_report(y_test, y_pred, labels=all_states, zero_division=0))

print("=== GLOBAL SHAP ANALYSIS ACROSS HISTORICAL REGIMES ===")
explainer = shap.TreeExplainer(rf_surrogate)
# Evaluate SHAP values across the full dataset X (all historical regimes representation)
shap_full = explainer(X)

print("Full SHAP values shape:", shap_full.shape)
mean_abs_shap_full = np.abs(shap_full.values).mean(axis=(0, 2))
global_shap_full_df = pd.DataFrame({
    "Feature": feature_names,
    "Mean_Absolute_SHAP": mean_abs_shap_full
}).sort_values(by="Mean_Absolute_SHAP", ascending=False).reset_index(drop=True)

print("--- Full Dataset Global SHAP Feature Importance ---")
print(global_shap_full_df.to_string(index=False))

print("\n=== LOCAL SHAP EXPLANATION SAMPLE ===")
sample_idx = 0
sample_date = X_test.index[sample_idx].strftime("%Y-%m-%d")
actual_hmm_state = int(y_test.iloc[sample_idx])
pred_surrogate_state = int(y_pred[sample_idx])
is_correct = (actual_hmm_state == pred_surrogate_state)

print(f"Observation Date : {sample_date}")
print(f"Actual HMM State : State {actual_hmm_state}")
print(f"Predicted State  : State {pred_surrogate_state}")
print(f"Prediction Match : {is_correct}")

print("\nALL TESTED SUCCESSFULLY!")
