"""
Market Data Acquisition & Feature Engineering Utility
Financial Market Regime & Event Intelligence Engine
"""

import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st

FEATURE_NAMES = [
    "Daily_Return", "Rolling_Volatility_20", "Momentum_20",
    "Drawdown", "SP500_TLT_Corr_20", "SP500_GLD_Corr_20"
]


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_raw_market_data(period: str = "5y") -> pd.DataFrame:
    """
    Downloads historical closing prices for ^GSPC, TLT, GLD using yfinance.
    Handles MultiIndex/Series formatting robustly.
    Returns a DataFrame indexed by Date with columns: Close, TLT_Close, GLD_Close.
    """
    try:
        sp500 = yf.download("^GSPC", period=period, interval="1d")["Close"]
        tlt = yf.download("TLT", period=period, interval="1d")["Close"]
        gld = yf.download("GLD", period=period, interval="1d")["Close"]

        if isinstance(sp500, pd.DataFrame): sp500 = sp500.squeeze()
        if isinstance(tlt, pd.DataFrame): tlt = tlt.squeeze()
        if isinstance(gld, pd.DataFrame): gld = gld.squeeze()

        raw_df = pd.DataFrame({
            "Close": sp500,
            "TLT_Close": tlt,
            "GLD_Close": gld
        }).dropna(how="all")

        if raw_df.empty or "Close" not in raw_df.columns:
            raise ValueError("Downloaded market data is empty or missing 'Close' column.")

        return raw_df
    except Exception as e:
        raise RuntimeError(f"Failed to fetch market data from yfinance: {str(e)}")


def calculate_market_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Recreates the exact 6 quantitative features from Notebook 01:
    - Daily_Return
    - Rolling_Volatility_20
    - Momentum_20
    - Drawdown
    - SP500_TLT_Corr_20
    - SP500_GLD_Corr_20
    
    Drops incomplete rolling NaN rows and preserves the Date index.
    """
    df = raw_df.copy()
    
    sp500_close = df["Close"]
    tlt_close = df["TLT_Close"]
    gld_close = df["GLD_Close"]

    daily_return = sp500_close.pct_change()
    vol_20 = daily_return.rolling(window=20).std()
    mom_20 = sp500_close.pct_change(periods=20)
    peak = sp500_close.cummax()
    drawdown = (sp500_close - peak) / peak

    tlt_return = tlt_close.pct_change()
    gld_return = gld_close.pct_change()

    corr_sp_tlt = daily_return.rolling(window=20).corr(tlt_return)
    corr_sp_gld = daily_return.rolling(window=20).corr(gld_return)

    df["Daily_Return"] = daily_return
    df["Rolling_Volatility_20"] = vol_20
    df["Momentum_20"] = mom_20
    df["Drawdown"] = drawdown
    df["SP500_TLT_Corr_20"] = corr_sp_tlt
    df["SP500_GLD_Corr_20"] = corr_sp_gld

    # Clean rows with incomplete rolling window calculations
    clean_df = df.dropna(subset=FEATURE_NAMES).copy()
    return clean_df


def get_market_dataset(period: str = "5y") -> pd.DataFrame:
    """
    Convenience function that fetches and processes the cleaned market dataset.
    Returns the cleaned DataFrame without database side effects.
    """
    raw_df = fetch_raw_market_data(period=period)
    clean_df = calculate_market_features(raw_df)
    return clean_df




def get_latest_market_snapshot(market_df: pd.DataFrame) -> dict:
    """
    Extracts formatted metric values from the latest available market session row.
    """
    if market_df.empty:
        return {}

    latest_row = market_df.iloc[-1]
    latest_date = market_df.index[-1].strftime("%Y-%m-%d")

    close_val = latest_row["Close"]
    daily_ret = latest_row["Daily_Return"]
    vol_20 = latest_row["Rolling_Volatility_20"]
    mom_20 = latest_row["Momentum_20"]
    drawdown_val = latest_row["Drawdown"]
    corr_tlt = latest_row["SP500_TLT_Corr_20"]
    corr_gld = latest_row["SP500_GLD_Corr_20"]

    return {
        "date": latest_date,
        "close": close_val,
        "close_fmt": f"{close_val:,.2f}",
        "daily_return": daily_ret,
        "daily_return_fmt": f"{daily_ret * 100:+.2f}%",
        "vol_20": vol_20,
        "vol_20_fmt": f"{vol_20 * 100:.2f}% (Daily) / {vol_20 * np.sqrt(252) * 100:.1f}% (Ann.)",
        "mom_20": mom_20,
        "mom_20_fmt": f"{mom_20 * 100:+.2f}%",
        "drawdown": drawdown_val,
        "drawdown_fmt": f"{drawdown_val * 100:.2f}%",
        "corr_tlt": corr_tlt,
        "corr_tlt_fmt": f"{corr_tlt:+.3f}",
        "corr_gld": corr_gld,
        "corr_gld_fmt": f"{corr_gld:+.3f}"
    }
