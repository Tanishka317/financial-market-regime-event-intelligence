"""
Model & Data Health Dashboard Page
"""

import streamlit as st
from app.components.layout import render_page_header, render_section_header
from app.components.cards import render_kpi_card, render_empty_state, render_status_badge


def render():
    """Renders the Model & Data Health Page visual structure."""
    render_page_header(
        title="Engine Architecture, Data & Model Health",
        subtitle="System telemetry, dataset coverage, pipeline execution states, and component version tracking.",
        status_badge_text="Research Environment",
        status_type="research"
    )

    # Section 1: Data Coverage & Telemetry
    render_section_header(
        title="Dataset Coverage & Ingestion Telemetry",
        subtitle="Historical market dataset and news feed storage status"
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_kpi_card(
            title="Data Coverage",
            value="Awaiting data",
            subtitle="S&P 500, TLT & GLD period",
            badge_text="Pending Data",
            badge_type="awaiting"
        )
    with col2:
        render_kpi_card(
            title="Observations Count",
            value="Awaiting data",
            subtitle="Usable daily trading sessions",
            badge_text="Pending Data",
            badge_type="awaiting"
        )
    with col3:
        render_kpi_card(
            title="Latest Timestamp",
            value="Awaiting data",
            subtitle="Most recent market close",
            badge_text="Pending Data",
            badge_type="awaiting"
        )
    with col4:
        render_kpi_card(
            title="Feature Dimension",
            value="6 Features",
            subtitle="Daily return, Vol, Mom, DD, Corrs",
            badge_text="Configured",
            badge_type="success"
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Section 2: Pipeline Component Health Matrix
    render_section_header(
        title="Pipeline Component Status Matrix",
        subtitle="Execution environment states across notebooks 01 through 08"
    )

    components_data = [
        {"name": "Market Data Acquisition", "stage": "Notebook 01", "status": "Research Ready", "type": "success", "details": "Yahoo Finance ^GSPC, TLT, GLD 5-year daily feed"},
        {"name": "Gaussian HMM Engine", "stage": "Notebook 02/03", "status": "Research Ready", "type": "success", "details": "4-state full covariance GaussianHMM model"},
        {"name": "Financial News Pipeline", "stage": "Notebook 04", "status": "Research Ready", "type": "success", "details": "Yahoo Finance news ticker search API"},
        {"name": "FinBERT Sentiment Layer", "stage": "Notebook 05", "status": "Research Ready", "type": "success", "details": "ProsusAI/finbert Transformer inference pipeline"},
        {"name": "Rule Event Classifier", "stage": "Notebook 06", "status": "Research Ready", "type": "success", "details": "Keyword pattern dictionaries across 8 event classes"},
        {"name": "Event-Regime Merger", "stage": "Notebook 07", "status": "Research Ready", "type": "success", "details": "Calendar date alignment & forward return join"},
        {"name": "SHAP Surrogate Explainer", "stage": "Notebook 08", "status": "Research Ready", "type": "success", "details": "RandomForestClassifier surrogate TreeExplainer"}
    ]

    for comp in components_data:
        badge_html = render_status_badge(comp["status"], comp["type"])
        st.markdown(
            f"""
            <div class="fmie-card" style="padding: 0.9rem 1.25rem; margin-bottom: 0.6rem;">
                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 0.5rem;">
                    <div>
                        <span style="font-weight: 600; color: #E6EDF3; font-size: 0.95rem;">{comp['name']}</span>
                        <span style="font-size: 0.8rem; color: #8B949E; margin-left: 0.75rem;">({comp['stage']})</span>
                    </div>
                    <div>
                        {badge_html}
                    </div>
                </div>
                <div style="font-size: 0.8rem; color: #6E7681; margin-top: 0.35rem;">
                    {comp['details']}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Section 3: System Environment & Dependencies
    render_section_header(
        title="Environment Telemetry",
        subtitle="Python runtime and library version environment details"
    )

    env_col1, env_col2 = st.columns(2)
    with env_col1:
        st.markdown(
            """
            <div class="fmie-card">
                <div style="font-weight: 600; color: #E6EDF3; margin-bottom: 0.5rem;">Virtual Environment</div>
                <div style="font-size: 0.82rem; color: #8B949E;">• Runtime: Python 3.11.5 (.venv)</div>
                <div style="font-size: 0.82rem; color: #8B949E;">• Framework: Streamlit 1.64.0</div>
                <div style="font-size: 0.82rem; color: #8B949E;">• Visualization: Plotly 6.x / Matplotlib</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with env_col2:
        st.markdown(
            """
            <div class="fmie-card">
                <div style="font-weight: 600; color: #E6EDF3; margin-bottom: 0.5rem;">ML & NLP Packages</div>
                <div style="font-size: 0.82rem; color: #8B949E;">• ML Ecosystem: scikit-learn, hmmlearn</div>
                <div style="font-size: 0.82rem; color: #8B949E;">• Explainability: SHAP 0.51.0</div>
                <div style="font-size: 0.82rem; color: #8B949E;">• NLP Model: Hugging Face Transformers (FinBERT)</div>
            </div>
            """,
            unsafe_allow_html=True
        )
