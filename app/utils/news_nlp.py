"""
FinBERT Sentiment Inference Layer
Financial Market Regime & Event Intelligence Engine
"""

import pandas as pd
import streamlit as st
from transformers import pipeline


@st.cache_resource(show_spinner=False)
def load_finbert_pipeline():
    """
    Loads and caches the pretrained ProsusAI/finbert sentiment analysis model.
    Recreates the exact NLP inference setup from Notebook 05.
    """
    return pipeline(
        "sentiment-analysis",
        model="ProsusAI/finbert",
        tokenizer="ProsusAI/finbert"
    )


@st.cache_data(ttl=1800, show_spinner=False)
def apply_finbert_sentiment(news_df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies FinBERT sentiment inference to financial news headlines.
    Creates sentiment_label and sentiment_score (confidence probability).
    """
    if news_df.empty or "headline" not in news_df.columns:
        df_out = news_df.copy()
        df_out["sentiment_label"] = []
        df_out["sentiment_score"] = []
        return df_out
        
    df_out = news_df.copy()
    sentiment_pipe = load_finbert_pipeline()
    
    headlines = df_out["headline"].tolist()
    
    try:
        nlp_results = sentiment_pipe(headlines)
        labels = [r["label"].lower() for r in nlp_results]
        scores = [round(float(r["score"]), 4) for r in nlp_results]
    except Exception as e:
        # Fallback if inference encounters unexpected error
        labels = ["neutral"] * len(headlines)
        scores = [0.0] * len(headlines)
        
    df_out["sentiment_label"] = labels
    df_out["sentiment_score"] = scores
    
    return df_out
