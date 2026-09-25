"""
Event-Market Alignment & Observational Analysis Layer
Financial Market Regime & Event Intelligence Engine
"""

import pandas as pd
import numpy as np
import streamlit as st
from app.utils.news_data import fetch_financial_news
from app.utils.news_nlp import apply_finbert_sentiment
from app.utils.event_classifier import apply_event_classification
from app.utils.regime_model import fit_and_decode_hmm


@st.cache_data(ttl=1800, show_spinner=False)
def build_event_market_pipeline(period: str = "5y") -> dict:
    """
    Executes the end-to-end event-market pipeline from Notebook 07:
    news_data -> sentiment -> event_classifier -> calendar alignment -> HMM join.
    
    Observational analysis ONLY. Does not claim causality.
    """
    # 1. Fetch news, apply FinBERT sentiment & event classification
    raw_news = fetch_financial_news()
    sentiment_news = apply_finbert_sentiment(raw_news)
    enriched_news = apply_event_classification(sentiment_news)
    
    if enriched_news.empty:
        return {
            "event_market_df": pd.DataFrame(),
            "event_stats_df": pd.DataFrame(),
            "sentiment_stats_df": pd.DataFrame(),
            "crosstab_df": pd.DataFrame(),
            "total_news_count": 0
        }
        
    # 2. Fetch HMM decoded market dataset from regime_model.py
    hmm_res = fit_and_decode_hmm(period=period)
    market_clean_df = hmm_res["market_df"]
    
    # Calculate 1-day and 5-day forward returns on market_clean_df if not already present
    if "event_day_return" not in market_clean_df.columns:
        market_clean_df["event_day_return"] = market_clean_df["Daily_Return"]
    if "next_day_return" not in market_clean_df.columns:
        market_clean_df["next_day_return"] = market_clean_df["Close"].pct_change(periods=1).shift(-1)
    if "5_day_forward_return" not in market_clean_df.columns:
        market_clean_df["5_day_forward_return"] = (market_clean_df["Close"].shift(-5) / market_clean_df["Close"]) - 1
        
    # 3. Calendar Date Alignment (Notebook 07 Methodology)
    # Convert tz-aware published_at to tz-naive date matching market index
    enriched_news["news_date"] = enriched_news["published_at"].dt.tz_localize(None).dt.floor("D")
    
    market_dates = market_clean_df.index
    min_news_date = enriched_news["news_date"].min()
    max_market_date = market_dates.max()
    
    if pd.isna(min_news_date) or min_news_date > max_market_date:
        min_news_date = market_dates.min()
        
    full_date_range = pd.date_range(start=min_news_date, end=max_market_date, freq="D")
    calendar_df = pd.DataFrame(index=full_date_range)
    calendar_df["trading_date"] = pd.Series(market_dates, index=market_dates)
    
    # Forward-backfill non-trading dates to next available trading day
    calendar_df["trading_date"] = calendar_df["trading_date"].bfill()
    
    # Map news to trading dates
    enriched_news["event_date"] = enriched_news["news_date"].map(calendar_df["trading_date"])
    
    # 4. Merge news with market regime & forward returns
    event_market_df = pd.merge(
        enriched_news,
        market_clean_df[["HMM_State", "event_day_return", "next_day_return", "5_day_forward_return", "Rolling_Volatility_20"]],
        left_on="event_date",
        right_index=True,
        how="inner"
    ).reset_index(drop=True)
    
    if event_market_df.empty:
        return {
            "event_market_df": enriched_news,
            "event_stats_df": pd.DataFrame(),
            "sentiment_stats_df": pd.DataFrame(),
            "crosstab_df": pd.DataFrame(),
            "total_news_count": len(enriched_news)
        }
        
    # 5. Aggregated Statistics by Event Category (%)
    event_stats = event_market_df.groupby("event_type").agg(
        Event_Count=("headline", "count"),
        Avg_Next_Day_Return=("next_day_return", lambda x: x.mean() * 100 if len(x.dropna()) > 0 else np.nan),
        Median_Next_Day_Return=("next_day_return", lambda x: x.median() * 100 if len(x.dropna()) > 0 else np.nan),
        Avg_5Day_Forward_Return=("5_day_forward_return", lambda x: x.mean() * 100 if len(x.dropna()) > 0 else np.nan),
        Median_5Day_Forward_Return=("5_day_forward_return", lambda x: x.median() * 100 if len(x.dropna()) > 0 else np.nan)
    ).reset_index().sort_values(by="Event_Count", ascending=False).reset_index(drop=True)
    
    # 6. Aggregated Statistics by Sentiment Label (%)
    sentiment_stats = event_market_df.groupby("sentiment_label").agg(
        Event_Count=("headline", "count"),
        Avg_Next_Day_Return=("next_day_return", lambda x: x.mean() * 100 if len(x.dropna()) > 0 else np.nan),
        Median_Next_Day_Return=("next_day_return", lambda x: x.median() * 100 if len(x.dropna()) > 0 else np.nan),
        Avg_5Day_Forward_Return=("5_day_forward_return", lambda x: x.mean() * 100 if len(x.dropna()) > 0 else np.nan),
        Median_5Day_Forward_Return=("5_day_forward_return", lambda x: x.median() * 100 if len(x.dropna()) > 0 else np.nan)
    ).reset_index()
    
    # 7. Event Type x HMM State Frequency Crosstab
    crosstab_df = pd.crosstab(
        event_market_df["event_type"],
        event_market_df["HMM_State"],
        margins=True
    ).reset_index()
    
    return {
        "event_market_df": event_market_df,
        "event_stats_df": event_stats,
        "sentiment_stats_df": sentiment_stats,
        "crosstab_df": crosstab_df,
        "total_news_count": len(event_market_df)
    }
