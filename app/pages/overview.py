"""
Overview Dashboard Page — Connected to Real Market Data Layer
"""

import streamlit as st
from app.components.layout import render_page_header, render_section_header
from app.components.cards import render_kpi_card, render_empty_state
from app.utils.market_data import get_market_dataset, get_latest_market_snapshot
from app.utils.charts import (
    create_sp500_price_chart,
    create_daily_returns_chart,
    create_rolling_volatility_chart,
    create_drawdown_chart,
    create_cross_asset_correlation_chart
)


def render():
    """Renders the Overview Page with real market data and Plotly charts."""
    
    # 1. Fetch real market data with Streamlit caching & error handling
    try:
        with st.spinner("Fetching market data..."):
            market_df = get_market_dataset(period="5y")
            snapshot = get_latest_market_snapshot(market_df)
            latest_date = snapshot.get("date", "Unknown")
            status_text = f"Latest session: {latest_date}"
            status_type = "success"
            data_error = None
    except Exception as e:
        market_df = None
        snapshot = {}
        status_text = "Market Data Unavailable"
        status_type = "warning"
        data_error = str(e)

    # 2. Render Page Header
    render_page_header(
        title="Financial Market Intelligence Overview",
        subtitle="Real-time quantitative market snapshot, S&P 500 price trends, volatility risk signals, and cross-asset correlations.",
        status_badge_text=status_text,
        status_type=status_type
    )

    if data_error:
        st.error(f"Market Data Ingestion Error: {data_error}")
        render_empty_state(
            title="Unable to Load Market Data",
            subtitle="Check network connection or yfinance API status.",
            icon="⚠️"
        )
        return

    # 3. MARKET SNAPSHOT: Top KPI Cards Row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_kpi_card(
            title="S&P 500 Close (^GSPC)",
            value=f"${snapshot['close_fmt']}",
            subtitle=f"Session date: {snapshot['date']}",
            badge_text="Real Market Data",
            badge_type="success"
        )
    with col2:
        daily_ret = snapshot['daily_return']
        ret_type = "success" if daily_ret >= 0 else "warning"
        render_kpi_card(
            title="Daily Return",
            value=snapshot['daily_return_fmt'],
            subtitle="1-day close-to-close change",
            badge_text="Daily Return",
            badge_type=ret_type
        )
    with col3:
        render_kpi_card(
            title="20D Rolling Volatility",
            value=snapshot['vol_20_fmt'],
            subtitle="20-day rolling std dev",
            badge_text="Vol Signal",
            badge_type="research"
        )
    with col4:
        render_kpi_card(
            title="Current Drawdown",
            value=snapshot['drawdown_fmt'],
            subtitle="Peak-to-trough decline",
            badge_text="Drawdown",
            badge_type="warning"
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # 4. PRICE TREND: S&P 500 Interactive Price Chart
    render_section_header(
        title="1. S&P 500 Price History & Trend",
        subtitle=f"5-year daily closing prices from {market_df.index.min().strftime('%Y-%m-%d')} to {latest_date} ({len(market_df):,} sessions)"
    )
    fig_price = create_sp500_price_chart(market_df)
    st.plotly_chart(fig_price, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 5. MARKET RISK / VOLATILITY: 3 Feature Risk Charts
    render_section_header(
        title="2. Market Risk & Quantitative Feature Signals",
        subtitle="Daily return distributions, 20-day rolling volatility, and peak-to-trough drawdown depth"
    )

    f_col1, f_col2, f_col3 = st.columns(3)
    with f_col1:
        fig_ret = create_daily_returns_chart(market_df)
        st.plotly_chart(fig_ret, use_container_width=True)
    with f_col2:
        fig_vol = create_rolling_volatility_chart(market_df)
        st.plotly_chart(fig_vol, use_container_width=True)
    with f_col3:
        fig_dd = create_drawdown_chart(market_df)
        st.plotly_chart(fig_dd, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 6. CROSS-ASSET CONTEXT: S&P 500 ↔ TLT & GLD Correlations
    render_section_header(
        title="3. Cross-Asset Macro Dynamics",
        subtitle="20-day rolling correlations between S&P 500, U.S. Treasuries (TLT), and Gold (GLD)"
    )

    c_col1, c_col2 = st.columns([1, 2])
    with c_col1:
        st.markdown(
            """
            <div style="font-size: 0.85rem; color: #8B949E; margin-bottom: 0.8rem;">
                Cross-asset correlations evaluate macroeconomic flight-to-safety dynamics (Treasuries) and inflation/hedging pressures (Gold).
            </div>
            """,
            unsafe_allow_html=True
        )
        render_kpi_card(
            title="S&P 500 ↔ TLT Corr (20D)",
            value=snapshot['corr_tlt_fmt'],
            subtitle="Treasury bond correlation",
            badge_text="20D Rolling",
            badge_type="research"
        )
        render_kpi_card(
            title="S&P 500 ↔ GLD Corr (20D)",
            value=snapshot['corr_gld_fmt'],
            subtitle="Gold commodity correlation",
            badge_text="20D Rolling",
            badge_type="research"
        )

    with c_col2:
        fig_corr = create_cross_asset_correlation_chart(market_df)
        st.plotly_chart(fig_corr, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 7. UNCONNECTED PIPELINE PLACEHOLDERS (HMM, News & SHAP)
    render_section_header(
        title="4. Downstream Intelligence Pipelines (Research Placeholders)",
        subtitle="Market regime decoding, financial event intelligence, and SHAP explainability states"
    )

    p_col1, p_col2 = st.columns(2)
    with p_col1:
        render_empty_state(
            title="Current HMM Regime: Awaiting Model Connection",
            subtitle="Connect 4-state Gaussian HMM model (Notebook 02) to display active market regime state.",
            icon="📈"
        )
    with p_col2:
        render_empty_state(
            title="Financial News Feed: No News Loaded",
            subtitle="Connect Yahoo Finance news pipeline (Notebooks 04–06) to display enriched event streams.",
            icon="📰"
        )
