"""
Sidebar Navigation Component for Financial Market Intelligence Engine
"""

import streamlit as st

NAV_OPTIONS = [
    "Overview",
    "Market Regimes",
    "News & Events",
    "Explainability",
    "Model & Data Health"
]

NAV_ICONS = {
    "Overview": "📊",
    "Market Regimes": "📈",
    "News & Events": "📰",
    "Explainability": "🔍",
    "Model & Data Health": "⚙️"
}


def render_sidebar() -> str:
    """
    Renders the compact sidebar navigation and returns the selected page name.
    """
    with st.sidebar:
        st.markdown(
            """
            <div style="padding: 0.5rem 0 1rem 0; border-bottom: 1px solid #21262D; margin-bottom: 1rem;">
                <div style="font-size: 1.05rem; font-weight: 700; color: #E6EDF3; letter-spacing: -0.01em;">
                    FMIE Analytics
                </div>
                <div style="font-size: 0.75rem; color: #8B949E; margin-top: 0.15rem;">
                    Regime & Event Engine
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        formatted_options = [f"{NAV_ICONS[opt]}  {opt}" for opt in NAV_OPTIONS]
        
        selected_fmt = st.radio(
            "Navigation",
            options=formatted_options,
            index=0,
            label_visibility="collapsed"
        )
        
        selected_page = selected_fmt.split("  ", 1)[1] if "  " in selected_fmt else selected_fmt

        st.markdown(
            """
            <div style="margin-top: 6rem; padding-top: 1rem; border-top: 1px solid #21262D;">
                <div style="font-size: 0.75rem; font-weight: 600; color: #8B949E;">
                    Financial Market Intelligence Engine
                </div>
                <div style="font-size: 0.7rem; color: #6E7681; margin-top: 0.2rem;">
                    Research Dashboard • v1.0.0
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    return selected_page
