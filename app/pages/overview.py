"""
Overview Dashboard Page — Command Center & Integrated Market Intelligence
Financial Market Regime & Event Intelligence Engine
"""

import streamlit as st
from app.components.layout import render_page_header, render_section_header
from app.components.cards import render_kpi_card, render_empty_state
from app.components.sidebar import navigate_to
from app.utils.shared_context import get_shared_dashboard_context
from app.utils.charts import (
    create_sp500_price_chart,
    create_daily_returns_chart,
    create_rolling_volatility_chart,
    create_drawdown_chart,
    create_cross_asset_correlation_chart
)


def render():
    """Renders the Overview Page as the integrated application command center."""
    
    # 1. Fetch unified shared dashboard context
    try:
        with st.spinner("Loading market & regime context..."):
            ctx = get_shared_dashboard_context(period="5y")
            market_df = ctx["market_df"]
            latest_date = ctx["latest_date"]
            status_text = f"Latest session: {latest_date}"
            status_type = "success"
            data_error = None
    except Exception as e:
        market_df = None
        ctx = {}
        status_text = "Market Data Unavailable"
        status_type = "warning"
        data_error = str(e)

    # 2. Render Page Header
    render_page_header(
        title="Financial Market Intelligence Overview",
        subtitle="Command center for real-time quantitative market snapshot, HMM regime context, financial news sentiment, and risk signals.",
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

    # 3. CURRENT MARKET INTELLIGENCE (Top Command Center Summary)
    render_section_header(
        title="Current Market Intelligence",
        subtitle=f"Real-time snapshot answering: 'What is happening in the market right now?' (As of {latest_date})"
    )

    c_col1, c_col2, c_col3, c_col4, c_col5, c_col6 = st.columns(6)
    
    with c_col1:
        render_kpi_card(
            title="Current Regime",
            value=f"State {ctx['current_state']}",
            subtitle=ctx['current_regime_label'],
            badge_text="Active HMM",
            badge_type="success"
        )
    with c_col2:
        render_kpi_card(
            title="Regime Streak",
            value=f"{ctx['consecutive_sessions']} Days",
            subtitle=f"Exp duration: {ctx['expected_duration']:.1f}d",
            badge_text="Persistence",
            badge_type="research"
        )
    with c_col3:
        render_kpi_card(
            title="S&P 500 Close",
            value=f"${ctx['sp500_close_fmt']}",
            subtitle=f"1D Return: {ctx['daily_return_fmt']}",
            badge_text="Price Trend",
            badge_type="success" if ctx['daily_return'] >= 0 else "warning"
        )
    with c_col4:
        render_kpi_card(
            title="20D Volatility",
            value=ctx['vol_20_fmt'].split(" ")[0],
            subtitle=f"Drawdown: {ctx['drawdown_fmt']}",
            badge_text="Risk Signal",
            badge_type="research"
        )
    with c_col5:
        render_kpi_card(
            title="Sentiment Snapshot",
            value=ctx['top_sentiment'].capitalize(),
            subtitle=f"{ctx['total_news']} news headlines ingested",
            badge_text="FinBERT",
            badge_type="success" if ctx['top_sentiment'] == "positive" else ("warning" if ctx['top_sentiment'] == "negative" else "research")
        )
    with c_col6:
        render_kpi_card(
            title="Top Event Category",
            value=ctx['top_event_type'],
            subtitle=f"{ctx['top_event_cnt']} category headlines",
            badge_text="Rule Match",
            badge_type="research"
        )

    # Narrative Intelligence Card with Navigation Links
    st.markdown(
        f"""
        <div class="fmie-card" style="border-left: 4px solid #58A6FF; margin-top: 0.5rem; margin-bottom: 1.5rem;">
            <div style="font-weight: 600; color: #E6EDF3; font-size: 0.95rem; margin-bottom: 0.3rem;">
                Executive Market Summary:
            </div>
            <div style="font-size: 0.85rem; color: #8B949E; line-height: 1.5;">
                The market is currently in <strong>State {ctx['current_state']} ({ctx['current_regime_label']})</strong> for {ctx['consecutive_sessions']} consecutive trading sessions. 
                {ctx['current_regime_desc']} 
                Latest ingested headlines reflect a predominantly <strong>{ctx['top_sentiment']}</strong> sentiment, with <strong>{ctx['top_event_type']}</strong> as the leading event driver.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Quick Navigation / Context Action Links
    st.markdown(
        """
        <div style="font-weight: 600; color: #E6EDF3; font-size: 0.9rem; margin-bottom: 0.6rem;">
            Deep-Dive Intelligence Shortcuts:
        </div>
        """,
        unsafe_allow_html=True
    )
    nav_col1, nav_col2, nav_col3 = st.columns(3)
    with nav_col1:
        if st.button("📈 Explore Market Regimes & Transition Matrix →", use_container_width=True, key="btn_nav_regimes"):
            navigate_to("Market Regimes")
    with nav_col2:
        if st.button("📰 Inspect News & FinBERT Sentiment Stream →", use_container_width=True, key="btn_nav_news"):
            navigate_to("News & Events")
    with nav_col3:
        if st.button("🔍 Analyze SHAP Feature Attributions →", use_container_width=True, key="btn_nav_shap"):
            navigate_to("Explainability")

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
            value=ctx['corr_tlt_fmt'],
            subtitle="Treasury bond correlation",
            badge_text="20D Rolling",
            badge_type="research"
        )
        render_kpi_card(
            title="S&P 500 ↔ GLD Corr (20D)",
            value=ctx['corr_gld_fmt'],
            subtitle="Gold commodity correlation",
            badge_text="20D Rolling",
            badge_type="research"
        )

    with c_col2:
        fig_corr = create_cross_asset_correlation_chart(market_df)
        st.plotly_chart(fig_corr, use_container_width=True)


if __name__ == "__main__":
    render()

