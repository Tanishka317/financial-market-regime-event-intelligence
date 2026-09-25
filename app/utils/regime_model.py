"""
Gaussian Hidden Markov Model (HMM) Market Regime Engine
Financial Market Regime & Event Intelligence Engine
"""

import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM
from app.utils.market_data import FEATURE_NAMES, get_market_dataset


@st.cache_data(ttl=3600, show_spinner=False)
def fit_and_decode_hmm(period: str = "5y") -> dict:
    """
    Fits the 4-state Gaussian HMM on standard quantitative features
    and decodes market regime states.
    
    Recreates exact methodology from Notebook 02:
    - 6 features: Daily_Return, Rolling_Volatility_20, Momentum_20, Drawdown, SP500_TLT_Corr_20, SP500_GLD_Corr_20
    - StandardScaler preprocessing
    - GaussianHMM(n_components=4, covariance_type="full", n_iter=200, random_state=42)
    """
    # 1. Obtain cleaned market dataset from market_data.py
    market_df = get_market_dataset(period=period)
    
    if market_df.empty or len(market_df) < 50:
        raise ValueError("Insufficient market data available for HMM fitting.")
        
    X = market_df[FEATURE_NAMES]
    
    # 2. Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 3. Fit 4-State Gaussian HMM
    hmm_model = GaussianHMM(
        n_components=4,
        covariance_type="full",
        n_iter=200,
        random_state=42
    )
    hmm_model.fit(X_scaled)
    
    # 4. Decode latent states
    market_df["HMM_State"] = hmm_model.predict(X_scaled)
    
    # 5. Extract transition matrix P (4x4)
    transmat = hmm_model.transmat_
    
    # 6. Empirical state statistics
    total_obs = len(market_df)
    state_stats = []
    
    for state in range(4):
        state_df = market_df[market_df["HMM_State"] == state]
        count = len(state_df)
        pct = (count / total_obs) * 100
        
        mean_ret = state_df["Daily_Return"].mean() if count > 0 else 0.0
        mean_vol = state_df["Rolling_Volatility_20"].mean() if count > 0 else 0.0
        mean_mom = state_df["Momentum_20"].mean() if count > 0 else 0.0
        mean_dd = state_df["Drawdown"].mean() if count > 0 else 0.0
        mean_tlt_corr = state_df["SP500_TLT_Corr_20"].mean() if count > 0 else 0.0
        mean_gld_corr = state_df["SP500_GLD_Corr_20"].mean() if count > 0 else 0.0
        
        # Qualitative regime mapping derived from empirical feature statistics
        if mean_vol < 0.009 and mean_dd > -0.02 and mean_mom > 0:
            regime_label = "Low-Vol Bull"
            regime_desc = "Calm market with low volatility, positive momentum, and minimal drawdown."
        elif mean_dd < -0.10 and mean_mom < 0:
            regime_label = "High-Vol Bear"
            regime_desc = "Elevated volatility, negative momentum, and deep drawdown stress."
        elif mean_dd < -0.08 and mean_mom > 0:
            regime_label = "Volatile Recovery"
            regime_desc = "High volatility with positive momentum recovering from cumulative drawdowns."
        else:
            regime_label = "Consolidation"
            regime_desc = "Moderate volatility, neutral momentum, and minor range-bound pullbacks."
            
        state_stats.append({
            "state": state,
            "count": count,
            "percentage": pct,
            "mean_daily_return": mean_ret,
            "mean_volatility_20": mean_vol,
            "mean_momentum_20": mean_mom,
            "mean_drawdown": mean_dd,
            "mean_tlt_corr": mean_tlt_corr,
            "mean_gld_corr": mean_gld_corr,
            "label": regime_label,
            "description": regime_desc
        })
        
    state_stats_df = pd.DataFrame(state_stats)
    
    # 7. State Duration Calculations
    durations = []
    
    # Calculate empirical consecutive block lengths
    market_df["_state_change"] = (market_df["HMM_State"] != market_df["HMM_State"].shift(1)).astype(int)
    market_df["_block_id"] = market_df["_state_change"].cumsum()
    block_lengths = market_df.groupby("_block_id").agg(
        state=("HMM_State", "first"),
        length=("HMM_State", "count")
    )
    empirical_durations = block_lengths.groupby("state")["length"].mean().to_dict()
    market_df.drop(columns=["_state_change", "_block_id"], inplace=True)
    
    for state in range(4):
        p_ii = transmat[state, state]
        # Model-implied expected duration = 1 / (1 - P_ii)
        model_duration = 1.0 / (1.0 - p_ii) if p_ii < 1.0 else np.inf
        emp_duration = empirical_durations.get(state, np.nan)
        
        durations.append({
            "state": state,
            "p_ii": p_ii,
            "model_expected_duration": model_duration,
            "empirical_avg_duration": emp_duration
        })
        
    duration_df = pd.DataFrame(durations)
    
    # 8. Latest session snapshot
    latest_state = int(market_df["HMM_State"].iloc[-1])
    latest_date = market_df.index[-1].strftime("%Y-%m-%d")
    latest_label = state_stats_df.loc[state_stats_df["state"] == latest_state, "label"].values[0]
    latest_desc = state_stats_df.loc[state_stats_df["state"] == latest_state, "description"].values[0]
    
    # Compute consecutive sessions in current state
    state_series = market_df["HMM_State"]
    consecutive_days = 0
    for s in reversed(state_series.values):
        if s == latest_state:
            consecutive_days += 1
        else:
            break
            
    return {
        "market_df": market_df,
        "transmat": transmat,
        "state_stats_df": state_stats_df,
        "duration_df": duration_df,
        "latest_state": latest_state,
        "latest_date": latest_date,
        "latest_label": latest_label,
        "latest_desc": latest_desc,
        "consecutive_days": consecutive_days,
        "log_likelihood": float(hmm_model.score(X_scaled))
    }
