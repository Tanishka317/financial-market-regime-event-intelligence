"""
Reusable Card Components for Financial Market Intelligence Engine
"""

import streamlit as st


def render_status_badge(text: str, badge_type: str = "research") -> str:
    """
    Returns HTML for a status badge pill.
    badge_type options: 'research', 'awaiting', 'warning', 'success'
    """
    valid_types = ["research", "awaiting", "warning", "success"]
    t = badge_type.lower() if badge_type.lower() in valid_types else "research"
    return f'<span class="fmie-badge fmie-badge-{t}">{text}</span>'


def render_kpi_card(
    title: str,
    value: str = "Awaiting market data",
    subtitle: str = None,
    badge_text: str = None,
    badge_type: str = "awaiting"
):
    """
    Renders a responsive metric/KPI card with dark financial styling.
    Strictly avoids fabricating fake numerical metrics.
    """
    badge_html = f'<div style="float: right;">{render_status_badge(badge_text, badge_type)}</div>' if badge_text else ''
    subtitle_html = f'<div class="fmie-kpi-sub">{subtitle}</div>' if subtitle else ''
    
    html_content = f"""
    <div class="fmie-kpi-card">
        {badge_html}
        <div class="fmie-kpi-title">{title}</div>
        <div class="fmie-kpi-value">{value}</div>
        {subtitle_html}
    </div>
    """
    st.markdown(html_content, unsafe_allow_html=True)


def render_empty_state(
    title: str = "Awaiting market data",
    subtitle: str = "Connect model/pipeline outputs to populate this view.",
    icon: str = "📊"
):
    """
    Renders a clean empty-state component for unpopulated dashboard sections.
    """
    html_content = f"""
    <div class="fmie-empty-state">
        <div style="font-size: 1.8rem; margin-bottom: 0.5rem; opacity: 0.8;">{icon}</div>
        <div class="fmie-empty-title">{title}</div>
        <div class="fmie-empty-sub">{subtitle}</div>
    </div>
    """
    st.markdown(html_content, unsafe_allow_html=True)
