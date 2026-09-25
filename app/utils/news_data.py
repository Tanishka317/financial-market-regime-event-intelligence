"""
Financial News Ingestion Layer
Financial Market Regime & Event Intelligence Engine
"""

import pandas as pd
import yfinance as yf
import streamlit as st

TARGET_TICKERS = [
    "^GSPC", "SPY", "^VIX", "TLT", "GLD", "QQQ",
    "AAPL", "MSFT", "NVDA", "AMZN", "JPM", "GS"
]


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_financial_news() -> pd.DataFrame:
    """
    Fetches real financial news items across 12 target market tickers using yfinance.
    Recreates the exact ingestion & cleaning pipeline from Notebook 04.
    
    Fields preserved:
    published_at, headline, publisher, query_ticker, summary, url
    """
    raw_records = []
    
    for ticker in TARGET_TICKERS:
        try:
            news_items = yf.Ticker(ticker).news
            if not news_items:
                continue
                
            for item in news_items:
                content = item.get("content", item)
                headline = content.get("title")
                pub_date = content.get("pubDate")
                summary = content.get("summary") or content.get("description", "")
                
                provider_info = content.get("provider", {})
                publisher = provider_info.get("displayName", "Unknown") if isinstance(provider_info, dict) else "Unknown"
                
                canonical_info = content.get("canonicalUrl", {})
                click_info = content.get("clickThroughUrl", {})
                
                url = ""
                if isinstance(canonical_info, dict) and canonical_info.get("url"):
                    url = canonical_info.get("url")
                elif isinstance(click_info, dict) and click_info.get("url"):
                    url = click_info.get("url")
                    
                if headline:
                    raw_records.append({
                        "query_ticker": ticker,
                        "headline": headline,
                        "pub_date_raw": pub_date,
                        "summary": summary if summary else "N/A",
                        "publisher": publisher if publisher else "Unknown",
                        "url": url if url else ""
                    })
        except Exception:
            continue

    if not raw_records:
        return pd.DataFrame(columns=[
            "published_at", "headline", "publisher", "query_ticker", "summary", "url"
        ])
        
    news_df = pd.DataFrame(raw_records)
    
    # Normalize publication timestamps consistently
    news_df["published_at"] = pd.to_datetime(news_df["pub_date_raw"], utc=True, errors="coerce")
    
    # Remove null/empty headlines and deduplicate by headline text
    news_df = news_df.dropna(subset=["headline", "published_at"]).copy()
    news_df = news_df.drop_duplicates(subset=["headline"]).reset_index(drop=True)
    
    # Sort descending by published_at
    news_df = news_df.sort_values(by="published_at", ascending=False).reset_index(drop=True)
    
    return news_df[["published_at", "headline", "publisher", "query_ticker", "summary", "url"]]
