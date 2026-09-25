"""
Explainability & SHAP Surrogate Page
"""

import streamlit as st
from app.components.layout import render_page_header, render_section_header
from app.components.cards import render_kpi_card, render_empty_state


def render():
    """Renders the Explainability Page visual structure."""
    render_page_header(
        title="Regime Explainability (SHAP Surrogate)",
        subtitle="Game-theoretic SHAP feature attributions explaining decision boundaries of the Random Forest surrogate model.",
        status_badge_text="Research Environment",
        status_type="research"
    )

    # Top KPI Row: Surrogate Fidelity Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        render_kpi_card(
            title="Surrogate Test Accuracy",
            value="Awaiting SHAP data",
            subtitle="Out-of-sample Random Forest state fidelity",
            badge_text="Pending Data",
            badge_type="awaiting"
        )
    with col2:
        render_kpi_card(
            title="Macro F1 (Active Test States)",
            value="Awaiting SHAP data",
            subtitle="F1-score across active test regimes",
            badge_text="Pending Data",
            badge_type="awaiting"
        )
    with col3:
        render_kpi_card(
            title="Top Global Driver",
            value="Awaiting SHAP data",
            subtitle="Highest mean absolute SHAP contribution",
            badge_text="Pending Data",
            badge_type="awaiting"
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Section 1: Global SHAP Feature Importance Across Historical Regimes
    render_section_header(
        title="Global SHAP Analysis Across Historical Regimes",
        subtitle="Mean absolute SHAP value ranking across 5-year multi-regime dataset (Drawdown, Momentum, Volatility, Correlations)"
    )
    
    left_col, right_col = st.columns([1, 1])
    with left_col:
        render_empty_state(
            title="Awaiting Global SHAP Feature Ranking Table",
            subtitle="Connect Notebook 08 SHAP TreeExplainer output to render feature ranking table.",
            icon="🏆"
        )
    with right_col:
        render_empty_state(
            title="Awaiting Global SHAP Bar Plot",
            subtitle="Connect SHAP values to display global feature importance bar chart.",
            icon="📊"
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Section 2: SHAP Beeswarm Summary Plot
    render_section_header(
        title="SHAP Beeswarm Summary Plot",
        subtitle="Feature value impact distributions (high vs. low feature values pushing regime assignments)"
    )
    render_empty_state(
        title="Awaiting SHAP Beeswarm Plot",
        subtitle="Connect TreeExplainer shap_values matrix to display interactive beeswarm plot.",
        icon="🐝"
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # Section 3: "Why This Regime?" Local Observation Breakdown
    render_section_header(
        title='Local Observation Breakdown ("Why This Regime?")',
        subtitle="Single-session SHAP waterfall attribution explaining specific trading date regime predictions"
    )
    render_empty_state(
        title="Awaiting Local Waterfall Attribution",
        subtitle="Connect Notebook 08 local SHAP waterfall output to inspect individual date decision paths.",
        icon="🔍"
    )
