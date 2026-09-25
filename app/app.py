"""
Financial Market Regime & Event Intelligence Engine
Main Streamlit Application Router & Entry Point
"""

import sys
import os
import streamlit as st

# Ensure root workspace directory is in python path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from app.utils.theme import apply_theme
from app.components.sidebar import render_sidebar
from app.pages import overview, regimes, events, explainability, model_health

# Streamlit Page Configuration
st.set_page_config(
    page_title="Financial Market Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply Centralized Dark Theme System
apply_theme()


def main():
    """Main application navigation router."""
    selected_page = render_sidebar()

    if selected_page == "Overview":
        overview.render()
    elif selected_page == "Market Regimes":
        regimes.render()
    elif selected_page == "News & Events":
        events.render()
    elif selected_page == "Explainability":
        explainability.render()
    elif selected_page == "Model & Data Health":
        model_health.render()
    else:
        overview.render()


if __name__ == "__main__":
    main()
