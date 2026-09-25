"""
Financial Market Regime & Event Intelligence Engine
Design System & Theme Tokens
"""

import streamlit as st

# Color Palette Tokens (Dark Financial Analytics System)
COLOR_BG = "#090C10"
COLOR_SIDEBAR_BG = "#0D1117"
COLOR_SURFACE = "#161B22"
COLOR_SURFACE_HOVER = "#21262D"
COLOR_BORDER = "#30363D"
COLOR_BORDER_LIGHT = "#21262D"

COLOR_TEXT_PRIMARY = "#E6EDF3"
COLOR_TEXT_SECONDARY = "#8B949E"
COLOR_TEXT_MUTED = "#6E7681"

COLOR_ACCENT_BLUE = "#58A6FF"
COLOR_ACCENT_GREEN = "#3FB950"
COLOR_ACCENT_RED = "#F85149"
COLOR_ACCENT_AMBER = "#D29922"
COLOR_ACCENT_PURPLE = "#A371F7"

CSS_STYLES = f"""
<style>
/* Reset & Global Container */
.stApp {{
    background-color: {COLOR_BG} !important;
    color: {COLOR_TEXT_PRIMARY} !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}}

/* Sidebar Overrides */
[data-testid="stSidebar"] {{
    background-color: {COLOR_SIDEBAR_BG} !important;
    border-right: 1px solid {COLOR_BORDER_LIGHT} !important;
}}

[data-testid="stSidebar"] .stMarkdown h1, 
[data-testid="stSidebar"] .stMarkdown h2, 
[data-testid="stSidebar"] .stMarkdown h3 {{
    color: {COLOR_TEXT_PRIMARY} !important;
}}

/* Main Block Padding */
.block-container {{
    padding-top: 1.5rem !important;
    padding-bottom: 3rem !important;
    max-width: 100% !important;
}}

/* Custom Panel / Card Container */
.fmie-card {{
    background-color: {COLOR_SURFACE};
    border: 1px solid {COLOR_BORDER};
    border-radius: 8px;
    padding: 1.25rem;
    margin-bottom: 1rem;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.3);
}}

/* KPI Metric Card */
.fmie-kpi-card {{
    background-color: {COLOR_SURFACE};
    border: 1px solid {COLOR_BORDER};
    border-radius: 8px;
    padding: 1.1rem 1.25rem;
    margin-bottom: 0.8rem;
    height: 100%;
}}

.fmie-kpi-title {{
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: {COLOR_TEXT_SECONDARY};
    font-weight: 600;
    margin-bottom: 0.25rem;
}}

.fmie-kpi-value {{
    font-size: 1.35rem;
    font-weight: 700;
    color: {COLOR_TEXT_PRIMARY};
    margin: 0.2rem 0;
}}

.fmie-kpi-sub {{
    font-size: 0.78rem;
    color: {COLOR_TEXT_MUTED};
    margin-top: 0.25rem;
}}

/* Status Badge */
.fmie-badge {{
    display: inline-block;
    padding: 0.22rem 0.65rem;
    font-size: 0.72rem;
    font-weight: 600;
    border-radius: 12px;
    letter-spacing: 0.03em;
    text-transform: uppercase;
}}

.fmie-badge-research {{
    background-color: rgba(88, 166, 255, 0.15);
    color: {COLOR_ACCENT_BLUE};
    border: 1px solid rgba(88, 166, 255, 0.4);
}}

.fmie-badge-awaiting {{
    background-color: rgba(139, 148, 158, 0.15);
    color: {COLOR_TEXT_SECONDARY};
    border: 1px solid rgba(139, 148, 158, 0.3);
}}

.fmie-badge-warning {{
    background-color: rgba(210, 153, 34, 0.15);
    color: {COLOR_ACCENT_AMBER};
    border: 1px solid rgba(210, 153, 34, 0.4);
}}

.fmie-badge-success {{
    background-color: rgba(63, 185, 80, 0.15);
    color: {COLOR_ACCENT_GREEN};
    border: 1px solid rgba(63, 185, 80, 0.4);
}}

/* Section Headers */
.fmie-section-header {{
    font-size: 1.15rem;
    font-weight: 600;
    color: {COLOR_TEXT_PRIMARY};
    margin-bottom: 0.25rem;
}}

.fmie-section-subtitle {{
    font-size: 0.85rem;
    color: {COLOR_TEXT_SECONDARY};
    margin-bottom: 1rem;
}}

/* Empty State Container */
.fmie-empty-state {{
    background-color: rgba(22, 27, 34, 0.6);
    border: 1px dashed {COLOR_BORDER};
    border-radius: 8px;
    padding: 2rem 1.5rem;
    text-align: center;
    color: {COLOR_TEXT_SECONDARY};
    margin: 0.5rem 0 1rem 0;
}}

.fmie-empty-title {{
    font-size: 0.95rem;
    font-weight: 600;
    color: {COLOR_TEXT_PRIMARY};
    margin-bottom: 0.4rem;
}}

.fmie-empty-sub {{
    font-size: 0.82rem;
    color: {COLOR_TEXT_MUTED};
}}

/* Radio / Navigation Customization */
div[data-testid="stSidebar"] div.stRadio > label {{
    display: none !important;
}}

div[data-testid="stSidebar"] div.stRadio > div {{
    gap: 0.3rem;
}}

div[data-testid="stSidebar"] div.stRadio > div > label {{
    background-color: transparent !important;
    border-radius: 6px !important;
    padding: 0.5rem 0.75rem !important;
    color: {COLOR_TEXT_SECONDARY} !important;
    border: 1px solid transparent !important;
    font-size: 0.9rem !important;
    font-weight: 500 !important;
    transition: all 0.15s ease-in-out;
}}

div[data-testid="stSidebar"] div.stRadio > div > label:hover {{
    background-color: {COLOR_SURFACE_HOVER} !important;
    color: {COLOR_TEXT_PRIMARY} !important;
}}

div[data-testid="stSidebar"] div.stRadio > div > label[data-checked="true"] {{
    background-color: {COLOR_SURFACE} !important;
    color: {COLOR_ACCENT_BLUE} !important;
    border: 1px solid {COLOR_BORDER} !important;
    font-weight: 600 !important;
}}

/* Clean Dividers */
hr {{
    border-color: {COLOR_BORDER_LIGHT} !important;
    margin: 1.2rem 0 !important;
}}
</style>
"""


def apply_theme():
    """Injects the dark financial design system CSS into Streamlit."""
    st.markdown(CSS_STYLES, unsafe_allow_html=True)


def get_plotly_dark_layout():
    """
    Returns a Plotly layout dict configured for the application's dark theme.
    Use this helper to style Plotly charts consistently across pages.
    """
    return {
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {
            "color": COLOR_TEXT_SECONDARY,
            "family": "-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif",
            "size": 12
        },
        "title": {
            "font": {"color": COLOR_TEXT_PRIMARY, "size": 14, "weight": "bold"},
            "x": 0.0,
            "xanchor": "left"
        },
        "xaxis": {
            "gridcolor": COLOR_BORDER_LIGHT,
            "zerolinecolor": COLOR_BORDER_LIGHT,
            "tickfont": {"color": COLOR_TEXT_MUTED}
        },
        "yaxis": {
            "gridcolor": COLOR_BORDER_LIGHT,
            "zerolinecolor": COLOR_BORDER_LIGHT,
            "tickfont": {"color": COLOR_TEXT_MUTED}
        },
        "margin": {"l": 40, "r": 20, "t": 40, "b": 40},
        "hoverlabel": {
            "bgcolor": COLOR_SURFACE,
            "bordercolor": COLOR_BORDER,
            "font": {"color": COLOR_TEXT_PRIMARY}
        }
    }
