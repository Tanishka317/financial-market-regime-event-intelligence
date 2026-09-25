"""
Reusable Layout Components for Financial Market Intelligence Engine
"""

import streamlit as st
from app.components.cards import render_status_badge


def render_page_header(
    title: str = "Financial Market Regime & Event Intelligence Engine",
    subtitle: str = "Market regimes, financial events, sentiment and explainable intelligence.",
    status_badge_text: str = "Research Environment",
    status_type: str = "research"
):
    """
    Renders the polished top-level application header.
    Includes system/environment badge without fabricating live statuses.
    """
    badge_html = render_status_badge(status_badge_text, status_type)
    
    st.markdown(
        f"""
        <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 1.2rem; flex-wrap: wrap; gap: 1rem;">
            <div>
                <h1 style="font-size: 1.65rem; font-weight: 700; color: #E6EDF3; margin: 0 0 0.25rem 0; letter-spacing: -0.02em;">
                    {title}
                </h1>
                <p style="font-size: 0.9rem; color: #8B949E; margin: 0; font-weight: 400;">
                    {subtitle}
                </p>
            </div>
            <div style="padding-top: 0.2rem;">
                {badge_html}
            </div>
        </div>
        <hr style="margin: 0.5rem 0 1.5rem 0;" />
        """,
        unsafe_allow_html=True
    )


def render_section_header(title: str, subtitle: str = None):
    """
    Renders a clean section header.
    """
    sub_html = f'<div class="fmie-section-subtitle">{subtitle}</div>' if subtitle else '<div style="margin-bottom: 0.75rem;"></div>'
    st.markdown(
        f"""
        <div class="fmie-section-header">{title}</div>
        {sub_html}
        """,
        unsafe_allow_html=True
    )
