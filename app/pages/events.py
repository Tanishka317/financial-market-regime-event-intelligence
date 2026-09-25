"""
News & Events Intelligence Page
"""

import streamlit as st
from app.components.layout import render_page_header, render_section_header
from app.components.cards import render_kpi_card, render_empty_state


def render():
    """Renders the News & Events Page visual structure."""
    render_page_header(
        title="Financial News & Event Intelligence",
        subtitle="Live news ingestion, FinBERT financial sentiment inference, and rule-based event classification.",
        status_badge_text="Research Environment",
        status_type="research"
    )

    # Top KPI Row: News Stream Summary
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_kpi_card(
            title="Total Headlines",
            value="Awaiting news data",
            subtitle="Ingested financial news items",
            badge_text="Pending Data",
            badge_type="awaiting"
        )
    with col2:
        render_kpi_card(
            title="Primary Sentiment",
            value="Awaiting news data",
            subtitle="FinBERT headline sentiment mode",
            badge_text="Pending Data",
            badge_type="awaiting"
        )
    with col3:
        render_kpi_card(
            title="Top Event Category",
            value="Awaiting news data",
            subtitle="Rule-based keyword pattern match",
            badge_text="Pending Data",
            badge_type="awaiting"
        )
    with col4:
        render_kpi_card(
            title="Mapped Trading Days",
            value="Awaiting news data",
            subtitle="Forward-backfilled calendar alignment",
            badge_text="Pending Data",
            badge_type="awaiting"
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Main Financial News Table Container
    render_section_header(
        title="Ingested Financial News Stream",
        subtitle="Headlines enriched with FinBERT sentiment labels, confidence scores, and rule-based event triggers"
    )
    render_empty_state(
        title="No Financial News Data Loaded",
        subtitle="Connect Yahoo Finance news pipeline (Notebooks 04–06) to inspect headlines, sentiment, and event classifications.",
        icon="📰"
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # Section 2: Sentiment & Event Distributions (2 columns)
    left_col, right_col = st.columns(2)

    with left_col:
        render_section_header(
            title="FinBERT Sentiment Breakdown",
            subtitle="Distribution of Positive, Neutral, and Negative headline sentiments"
        )
        render_empty_state(
            title="Awaiting Sentiment Distribution Data",
            subtitle="Connect FinBERT inference outputs (Notebook 05) to display sentiment distribution charts.",
            icon="🧠"
        )

    with right_col:
        render_section_header(
            title="Event Category Taxonomy",
            subtitle="Monetary Policy, Inflation, Labor, Earnings, M&A, Geopolitical & Market events"
        )
        render_empty_state(
            title="Awaiting Event Category Counts",
            subtitle="Connect event classification outputs (Notebook 06) to display category frequency bar charts.",
            icon="🏷️"
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Section 3: Event-Market Forward Return Analysis
    render_section_header(
        title="Event vs. Subsequent Forward Returns",
        subtitle="Observational evaluation of 1-day and 5-trading-day forward cumulative returns following news events"
    )
    render_empty_state(
        title="Awaiting Event-Market Merged Dataset",
        subtitle="Connect Notebook 07 observational dataset to inspect forward-return statistics across event categories and HMM market states.",
        icon="📊"
    )
