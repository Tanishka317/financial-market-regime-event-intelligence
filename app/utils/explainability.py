"""
SHAP Explainability & Surrogate Model Utility
Financial Market Regime & Event Intelligence Engine
"""

import pandas as pd
import numpy as np
import shap
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from app.utils.market_data import FEATURE_NAMES
from app.utils.regime_model import fit_and_decode_hmm


@st.cache_data(ttl=3600, show_spinner=False)
def compute_shap_explainability(period: str = "5y") -> dict:
    """
    Fits the Random Forest surrogate model on HMM state decodes (80/20 chronological split)
    and computes global and local SHAP feature attributions using TreeExplainer.
    
    Recreates exact methodology from Notebook 08.
    """
    # 1. Obtain market dataset with decoded HMM_State from regime_model.py
    hmm_res = fit_and_decode_hmm(period=period)
    market_df = hmm_res["market_df"]
    
    if market_df.empty or "HMM_State" not in market_df.columns:
        raise ValueError("Market dataset with decoded HMM_State is unavailable.")
        
    X = market_df[FEATURE_NAMES]
    y = market_df["HMM_State"]
    
    # 2. Chronological 80/20 Train/Test Split
    train_size = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
    
    # 3. Train Random Forest Surrogate Model
    rf_surrogate = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced"
    )
    rf_surrogate.fit(X_train, y_train)
    
    # 4. Evaluate Out-of-Sample Surrogate Fidelity
    y_pred = rf_surrogate.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    
    present_classes = np.unique(y_test)
    macro_f1_active = f1_score(y_test, y_pred, labels=present_classes, average="macro")
    macro_f1_all = f1_score(y_test, y_pred, labels=[0, 1, 2, 3], average="macro", zero_division=0)
    
    test_state_counts = pd.Series(y_test).value_counts().to_dict()
    
    # 5. Initialize SHAP TreeExplainer
    explainer = shap.TreeExplainer(rf_surrogate)
    
    # 6. Global SHAP Feature Importance across full historical dataset X
    shap_full = explainer(X)
    mean_abs_shap = np.abs(shap_full.values).mean(axis=(0, 2))
    
    global_shap_df = pd.DataFrame({
        "Feature": FEATURE_NAMES,
        "Mean_Absolute_SHAP": mean_abs_shap
    }).sort_values(by="Mean_Absolute_SHAP", ascending=False).reset_index(drop=True)
    
    # 7. Local SHAP Explanation for the latest available market observation
    latest_idx = len(market_df) - 1
    latest_date = market_df.index[latest_idx].strftime("%Y-%m-%d")
    actual_hmm_state = int(market_df["HMM_State"].iloc[latest_idx])
    
    latest_X = X.iloc[[latest_idx]]
    surrogate_pred_state = int(rf_surrogate.predict(latest_X)[0])
    is_match = (actual_hmm_state == surrogate_pred_state)
    
    # Extract feature values for latest session
    latest_features = {feat: float(latest_X[feat].iloc[0]) for feat in FEATURE_NAMES}
    
    # Extract SHAP value vector for predicted state class at latest_idx
    # shap_full.values has shape (N_samples, N_features, N_classes)
    latest_shap_vector = shap_full.values[latest_idx, :, surrogate_pred_state]
    
    local_shap_df = pd.DataFrame({
        "Feature": FEATURE_NAMES,
        "SHAP_Value": latest_shap_vector,
        "Feature_Value": [latest_features[f] for f in FEATURE_NAMES]
    }).sort_values(by="SHAP_Value", key=abs, ascending=False).reset_index(drop=True)
    
    return {
        "test_accuracy": float(test_acc),
        "macro_f1_active": float(macro_f1_active),
        "macro_f1_all": float(macro_f1_all),
        "test_samples": len(y_test),
        "test_state_counts": test_state_counts,
        "active_test_states": present_classes.tolist(),
        "global_shap_df": global_shap_df,
        "latest_date": latest_date,
        "actual_hmm_state": actual_hmm_state,
        "surrogate_pred_state": surrogate_pred_state,
        "is_match": is_match,
        "latest_features": latest_features,
        "local_shap_df": local_shap_df
    }
