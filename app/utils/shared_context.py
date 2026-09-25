"""
Shared Dashboard Context Helper
Financial Market Regime & Event Intelligence Engine
"""

import pandas as pd
import numpy as np
import streamlit as st
from app.utils.market_data import get_market_dataset, get_latest_market_snapshot
from app.utils.regime_model import fit_and_decode_hmm
from app.utils.event_market import build_event_market_pipeline


@st.cache_data(ttl=1800, show_spinner=False)
def get_shared_dashboard_context(period: str = "5y") -> dict:
    """
    Returns a unified, consistent current-market & regime intelligence snapshot
    for use across all dashboard pages without duplicating downloads or model fits.
    """
    # 1. Market Data Snapshot
    market_df = get_market_dataset(period=period)
    snapshot = get_latest_market_snapshot(market_df)
    
    # 2. HMM Regime Snapshot
    hmm_res = fit_and_decode_hmm(period=period)
    
    # 3. News & Event Pipeline Snapshot (safely handled)
    try:
        pipeline_res = build_event_market_pipeline(period=period)
        em_df = pipeline_res.get("event_market_df")
        total_news = pipeline_res.get("total_news_count", 0)
        
        if em_df is not None and not em_df.empty:
            s_counts = em_df["sentiment_label"].value_counts()
            top_sentiment = s_counts.index[0] if not s_counts.empty else "neutral"
            top_sentiment_cnt = int(s_counts.iloc[0]) if not s_counts.empty else 0
            
            if "event_type" in em_df.columns and not em_df["event_type"].empty:
                e_counts = em_df["event_type"].value_counts()
                top_event_type = str(e_counts.index[0])
                top_event_cnt = int(e_counts.iloc[0])
            else:
                top_event_type = "N/A"
                top_event_cnt = 0
                
            latest_headline = str(em_df.iloc[0]["headline"]) if "headline" in em_df.columns else "N/A"
            latest_publisher = str(em_df.iloc[0]["publisher"]) if "publisher" in em_df.columns else "N/A"
        else:
            top_sentiment = "neutral"
            top_sentiment_cnt = 0
            top_event_type = "N/A"
            top_event_cnt = 0
            latest_headline = "N/A"
            latest_publisher = "N/A"
    except Exception:
        total_news = 0
        top_sentiment = "N/A"
        top_sentiment_cnt = 0
        top_event_type = "N/A"
        top_event_cnt = 0
        latest_headline = "N/A"
        latest_publisher = "N/A"

    # State duration & transition info for current state
    curr_state = hmm_res["latest_state"]
    dur_df = hmm_res["duration_df"]
    transmat = hmm_res["transmat"]
    
    state_dur_row = dur_df.loc[dur_df["state"] == curr_state].iloc[0] if not dur_df.empty else None
    expected_dur = float(state_dur_row["model_expected_duration"]) if state_dur_row is not None else 0.0
    empirical_dur = float(state_dur_row["empirical_avg_duration"]) if state_dur_row is not None else 0.0
    p_ii = float(state_dur_row["p_ii"]) if state_dur_row is not None else 0.0

    # Self-transition & next most likely state transition
    row_trans = transmat[curr_state].copy()
    row_trans[curr_state] = -1.0  # exclude self
    next_state = int(np.argmax(row_trans))
    next_state_prob = float(transmat[curr_state, next_state])

    return {
        "market_df": market_df,
        "latest_date": snapshot.get("date", "N/A"),
        "sp500_close": snapshot.get("close", 0.0),
        "sp500_close_fmt": snapshot.get("close_fmt", "$0.00"),
        "daily_return": snapshot.get("daily_return", 0.0),
        "daily_return_fmt": snapshot.get("daily_return_fmt", "0.00%"),
        "vol_20": snapshot.get("vol_20", 0.0),
        "vol_20_fmt": snapshot.get("vol_20_fmt", "0.00%"),
        "mom_20": snapshot.get("mom_20", 0.0),
        "mom_20_fmt": snapshot.get("mom_20_fmt", "0.00%"),
        "drawdown": snapshot.get("drawdown", 0.0),
        "drawdown_fmt": snapshot.get("drawdown_fmt", "0.00%"),
        "corr_tlt_fmt": snapshot.get("corr_tlt_fmt", "0.000"),
        "corr_gld_fmt": snapshot.get("corr_gld_fmt", "0.000"),
        
        # HMM Regime Context
        "current_state": curr_state,
        "current_regime_label": hmm_res["latest_label"],
        "current_regime_desc": hmm_res["latest_desc"],
        "consecutive_sessions": hmm_res["consecutive_days"],
        "expected_duration": expected_dur,
        "empirical_duration": empirical_dur,
        "p_ii": p_ii,
        "next_likely_state": next_state,
        "next_likely_prob": next_state_prob,
        "total_obs": len(market_df),
        "log_likelihood": hmm_res["log_likelihood"],
        
        # News & Event Context
        "total_news": total_news,
        "top_sentiment": top_sentiment,
        "top_sentiment_cnt": top_sentiment_cnt,
        "top_event_type": top_event_type,
        "top_event_cnt": top_event_cnt,
        "latest_headline": latest_headline,
        "latest_publisher": latest_publisher,
    }
