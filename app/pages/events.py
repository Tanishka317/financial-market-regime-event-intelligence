"""
Financial News & Event Intelligence Dashboard Page — Connected to Notebooks 04–07 Pipelines
Financial Market Regime & Event Intelligence Engine
"""

import streamlit as st
import pandas as pd
import numpy as np
from app.components.layout import render_page_header, render_section_header
from app.components.cards import render_kpi_card, render_empty_state, render_status_badge
from app.utils.event_market import build_event_market_pipeline
from app.utils.shared_context import get_shared_dashboard_context
from app.utils.event_classifier import EVENT_CATEGORIES
from app.utils.charts import (
    create_sentiment_distribution_chart,
    create_event_type_distribution_chart,
    create_event_hmm_crosstab_heatmap
)


def render():
    """Renders the News & Events Page connected to real financial news, FinBERT sentiment, and event-market pipeline."""
    
    # 1. Fetch & build event-market dataset & shared context
    try:
        with st.spinner("Ingesting financial news & running FinBERT sentiment analysis..."):
            ctx = get_shared_dashboard_context(period="5y")
            pipeline_res = build_event_market_pipeline(period="5y")
            em_df = pipeline_res["event_market_df"]
            event_stats = pipeline_res["event_stats_df"]
            sentiment_stats = pipeline_res["sentiment_stats_df"]
            crosstab_df = pipeline_res["crosstab_df"]
            total_news = pipeline_res["total_news_count"]
            
            curr_state = ctx["current_state"]
            curr_regime = ctx["current_regime_label"]
            
            status_text = f"Live News Ingested: {total_news} items"
            status_type = "success"
            data_error = None
    except Exception as e:
        em_df = pd.DataFrame()
        total_news = 0
        curr_state = 0
        curr_regime = "Unknown"
        status_text = "News Ingestion Error"
        status_type = "warning"
        data_error = str(e)

    # 2. Render Page Header
    render_page_header(
        title="Financial News & Event Intelligence",
        subtitle="Financial news stream, FinBERT sentiment inference, rule-based event classification, and historical event-market return associations.",
        status_badge_text=status_text,
        status_type=status_type
    )

    if data_error:
        st.error(f"Financial News Pipeline Error: {data_error}")
        render_empty_state(
            title="Unable to Ingest Financial News",
            subtitle="Check network connection or Yahoo Finance news API status.",
            icon="⚠️"
        )
        return

    if em_df.empty:
        render_empty_state(
            title="No Financial News Data Loaded",
            subtitle="The news ingestion pipeline returned zero news records.",
            icon="📰"
        )
        return

    # 3. CURRENT MARKET REGIME BANNER
    st.markdown(
        f"""
        <div class="fmie-card" style="border-left: 4px solid #3FB950; margin-bottom: 1.2rem; padding: 0.85rem 1.1rem;">
            <div style="font-size: 0.85rem; color: #8B949E;">
                Current Active Market Regime: <strong style="color: #E6EDF3;">State {curr_state} ({curr_regime})</strong> • All ingested events below are chronologically aligned with market regime states and daily price action.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 4. SECTION 1: NEWS OVERVIEW (KPI Cards)
    s_counts = em_df["sentiment_label"].value_counts()
    pos_count = s_counts.get("positive", 0)
    neu_count = s_counts.get("neutral", 0)
    neg_count = s_counts.get("negative", 0)
    
    pos_pct = (pos_count / total_news * 100) if total_news > 0 else 0
    neu_pct = (neu_count / total_news * 100) if total_news > 0 else 0
    neg_pct = (neg_count / total_news * 100) if total_news > 0 else 0
    
    unique_categories = em_df["event_type"].nunique()

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        render_kpi_card(
            title="Ingested News Items",
            value=f"{total_news}",
            subtitle="12 target market tickers",
            badge_text="Live News",
            badge_type="success"
        )
    with col2:
        render_kpi_card(
            title="Positive Sentiment",
            value=f"{pos_pct:.1f}%",
            subtitle=f"{pos_count} headlines",
            badge_text="FinBERT Positive",
            badge_type="success"
        )
    with col3:
        render_kpi_card(
            title="Neutral Sentiment",
            value=f"{neu_pct:.1f}%",
            subtitle=f"{neu_count} headlines",
            badge_text="FinBERT Neutral",
            badge_type="research"
        )
    with col4:
        render_kpi_card(
            title="Negative Sentiment",
            value=f"{neg_pct:.1f}%",
            subtitle=f"{neg_count} headlines",
            badge_text="FinBERT Negative",
            badge_type="warning"
        )
    with col5:
        render_kpi_card(
            title="Event Categories",
            value=f"{unique_categories} Active",
            subtitle="Rule-based matching",
            badge_text="Event Rules",
            badge_type="research"
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # 5. SECTION 2: RECENT FINANCIAL NEWS STREAM (Filtered Table View)
    render_section_header(
        title="2. Recent Ingested Financial News Feed",
        subtitle="Headlines enriched with FinBERT sentiment labels, confidence probability scores, rule-based event triggers, and HMM market regime decodes"
    )

    # Filtering Controls
    f_col1, f_col2, f_col3 = st.columns(3)
    with f_col1:
        ticker_options = ["All Tickers"] + sorted(list(em_df["query_ticker"].unique()))
        selected_ticker = st.selectbox("Filter by Ticker:", options=ticker_options, index=0)
    with f_col2:
        sentiment_options = ["All Sentiments", "positive", "neutral", "negative"]
        selected_sentiment = st.selectbox("Filter by Sentiment:", options=sentiment_options, index=0)
    with f_col3:
        event_options = ["All Event Types"] + sorted(list(em_df["event_type"].unique()))
        selected_event = st.selectbox("Filter by Event Type:", options=event_options, index=0)

    # Filter dataframe
    filtered_df = em_df.copy()
    if selected_ticker != "All Tickers":
        filtered_df = filtered_df[filtered_df["query_ticker"] == selected_ticker]
    if selected_sentiment != "All Sentiments":
        filtered_df = filtered_df[filtered_df["sentiment_label"] == selected_sentiment]
    if selected_event != "All Event Types":
        filtered_df = filtered_df[filtered_df["event_type"] == selected_event]

    st.markdown(f"<div style='font-size: 0.8rem; color: #8B949E; margin-bottom: 0.8rem;'>Showing {len(filtered_df)} of {total_news} news items</div>", unsafe_allow_html=True)

    # Render Clean News List/Feed Card View
    for idx, row in filtered_df.head(25).iterrows():
        pub_str = row["published_at"].strftime("%Y-%m-%d %H:%M UTC") if pd.notna(row["published_at"]) else "Unknown Date"
        headline = row["headline"]
        publisher = row["publisher"]
        query_ticker = row["query_ticker"]
        sentiment_lbl = row["sentiment_label"]
        sentiment_score = row["sentiment_score"]
        event_type = row["event_type"]
        url = row["url"]
        hmm_state = row.get("HMM_State", "N/A")
        
        # Headline HTML link if URL available
        if url:
            headline_html = f'<a href="{url}" target="_blank" style="color: #E6EDF3; text-decoration: none; font-weight: 600;">{headline} ↗</a>'
        else:
            headline_html = f'<span style="color: #E6EDF3; font-weight: 600;">{headline}</span>'
            
        sentiment_badge_type = "success" if sentiment_lbl == "positive" else ("warning" if sentiment_lbl == "negative" else "research")
        sent_badge = render_status_badge(f"{sentiment_lbl} ({sentiment_score*100:.1f}%)", sentiment_badge_type)
        event_badge = render_status_badge(event_type, "research")
        regime_badge = render_status_badge(f"HMM State {hmm_state}", "success" if hmm_state == curr_state else "research")

        st.markdown(
            f"""
            <div class="fmie-card" style="padding: 0.85rem 1.1rem; margin-bottom: 0.55rem;">
                <div style="display: flex; justify-content: space-between; align-items: flex-start; gap: 0.8rem; flex-wrap: wrap;">
                    <div style="flex: 1; min-width: 280px;">
                        <div style="font-size: 0.92rem; margin-bottom: 0.3rem;">{headline_html}</div>
                        <div style="font-size: 0.76rem; color: #8B949E;">
                            <strong>{publisher}</strong> • Ticker: <code style="color: #58A6FF; background: rgba(88,166,255,0.1); padding: 0.1rem 0.3rem; border-radius: 4px;">{query_ticker}</code> • {pub_str}
                        </div>
                    </div>
                    <div style="display: flex; gap: 0.4rem; align-items: center; flex-wrap: wrap;">
                        {regime_badge}
                        {event_badge}
                        {sent_badge}
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # 6. SECTIONS 3 & 4: SENTIMENT & EVENT DISTRIBUTION CHARTS (2 columns)
    d_col1, d_col2 = st.columns(2)
    with d_col1:
        render_section_header(
            title="3. FinBERT Sentiment Distribution",
            subtitle="Headline classification output from ProsusAI/finbert model"
        )
        fig_sent = create_sentiment_distribution_chart(s_counts)
        st.plotly_chart(fig_sent, use_container_width=True)

    with d_col2:
        render_section_header(
            title="4. Event Category Volume Distribution",
            subtitle="Headline counts across the 9 rule-based event categories"
        )
        fig_event = create_event_type_distribution_chart(event_stats)
        st.plotly_chart(fig_event, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 7. SECTION 5: EVENT-MARKET ASSOCIATION (Aggregated Return Statistics Table)
    render_section_header(
        title="5. Observational Event-Market Return Associations",
        subtitle="Historical average and median 1-day and 5-trading-day forward cumulative returns following event categories"
    )

    st.markdown(
        """
        <div style="font-size: 0.78rem; color: #D29922; margin-bottom: 0.6rem;">
            ⚠️ <strong>Methodological Notice</strong>: The return statistics below represent <strong>historical observational associations</strong> across the market dataset. 
            They are strictly non-causal and do not represent predictive trading signals.
        </div>
        """,
        unsafe_allow_html=True
    )

    if not event_stats.empty:
        formatted_event_stats = pd.DataFrame({
            "Event Category": event_stats["event_type"],
            "Article Count": event_stats["Event_Count"],
            "Mean Next-Day Return": [f"{v:+.2f}%" if pd.notna(v) else "N/A" for v in event_stats["Avg_Next_Day_Return"]],
            "Median Next-Day Return": [f"{v:+.2f}%" if pd.notna(v) else "N/A" for v in event_stats["Median_Next_Day_Return"]],
            "Mean 5-Day Forward Return": [f"{v:+.2f}%" if pd.notna(v) else "N/A" for v in event_stats["Avg_5Day_Forward_Return"]],
            "Median 5-Day Forward Return": [f"{v:+.2f}%" if pd.notna(v) else "N/A" for v in event_stats["Median_5Day_Forward_Return"]]
        })
        st.dataframe(formatted_event_stats, use_container_width=True, hide_index=True)
    else:
        render_empty_state("No Event Statistics Available", "Insufficient aligned event-market records.")

    st.markdown("<br>", unsafe_allow_html=True)

    # 8. SECTION 6: REGIME × EVENT CONTEXT (Heatmap)
    render_section_header(
        title="6. Event Category × HMM Market State Crosstab",
        subtitle="Observational frequency cross-tabulation mapping event categories against active HMM market regime states"
    )

    if not crosstab_df.empty:
        fig_ct = create_event_hmm_crosstab_heatmap(crosstab_df)
        st.plotly_chart(fig_ct, use_container_width=True)
        st.caption(
            "This cross-tabulation provides qualitative historical context on which market regimes were active when specific news categories occurred. "
            "It represents observational association, not causal impact."
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # 9. SECTION 7: METHODOLOGY & LIMITATIONS
    render_section_header(
        title="7. Pipeline Methodology & Disclaimers",
        subtitle="10-point framework documenting ingestion, FinBERT sentiment, event classification, and observational limits"
    )

    with st.expander("📖 View Detailed Event Pipeline Methodology & Limitations", expanded=True):
        st.markdown(
            """
            1. **News Data Source**: Ingested from Yahoo Finance news feed across 12 U.S. market tickers (`^GSPC`, `SPY`, `^VIX`, `TLT`, `GLD`, `QQQ`, `AAPL`, `MSFT`, `NVDA`, `AMZN`, `JPM`, `GS`).
            2. **FinBERT Sentiment Inference**: Applied pretrained `ProsusAI/finbert` model via Hugging Face/PyTorch pipeline without fine-tuning. Sentiment scores represent class confidence probabilities.
            3. **Rule-Based Event Classification**: Assigned event types using regex keyword matching across 9 categories in priority order (Monetary Policy, Inflation, Employment/Labor, Earnings, M&A, Geopolitical, Commodities, Market/Index, Other).
            4. **Calendar Date Alignment**: Mapped news publication dates to trading dates using a forward-backfill (`bfill`) rule for non-trading weekend/holiday sessions.
            5. **HMM Regime Integration**: Decoded 4-state Gaussian HMM market regimes joined by aligned trading date.
            6. **Forward Returns Calculation**: Computed next-day return ($t+1$) and 5-trading-day cumulative forward return ($(Close_{t+5} / Close_t) - 1$).
            7. **Strict Observational Disclaimer**: All event-market associations are purely observational. We do NOT claim headlines caused market returns or that sentiment predicts future prices.
            8. **Dataset Coverage Limits**: Live news feed provides a modest sample of recent articles rather than multi-year historical news archives.
            9. **Headline Clustering**: Major macroeconomic announcements generate multiple headlines across publishers, causing article clustering.
            10. **Loss of Intraday Timing**: Date-level mapping maps news to daily close-to-close returns, losing intraday timestamp execution details.
            """,
            unsafe_allow_html=True
        )


if __name__ == "__main__":
    render()

