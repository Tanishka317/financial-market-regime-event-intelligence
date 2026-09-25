"""
Sidebar Navigation Component & Page Registry for Financial Market Intelligence Engine
"""

import streamlit as st


def get_pages() -> dict:
    """
    Lazy-loads and returns the centralized Streamlit Page objects mapping.
    Prevents circular import issues between sidebar and page modules.
    """
    from app.pages import overview, regimes, events, explainability, model_health

    return {
        "Overview": st.Page(overview.render, title="Overview", icon="📊", default=True, url_path="overview"),
        "Market Regimes": st.Page(regimes.render, title="Market Regimes", icon="📈", url_path="regimes"),
        "News & Events": st.Page(events.render, title="News & Events", icon="📰", url_path="events"),
        "Explainability": st.Page(explainability.render, title="Explainability", icon="🔍", url_path="explainability"),
        "Model & Data Health": st.Page(model_health.render, title="Model & Data Health", icon="⚙️", url_path="health"),
    }


def navigate_to(page_name: str):
    """Programmatically switches to the specified page using Streamlit's native router."""
    pages = get_pages()
    if page_name in pages:
        st.switch_page(pages[page_name])


def render_sidebar_header():
    """Renders the top brand header in the sidebar."""
    st.markdown(
        """
        <div style="padding: 0.5rem 0 1rem 0; border-bottom: 1px solid #21262D; margin-bottom: 0.8rem;">
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


def render_sidebar_footer():
    """Renders the bottom brand footer in the sidebar."""
    st.markdown(
        """
        <div style="margin-top: 4rem; padding-top: 1rem; border-top: 1px solid #21262D;">
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
