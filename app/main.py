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
from app.components.sidebar import get_pages, render_sidebar_header, render_sidebar_footer

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
    """Main application navigation router using st.navigation."""
    # 1. Obtain Centralized Page Registry
    pages_map = get_pages()

    # 2. Render Top Sidebar Brand Header
    with st.sidebar:
        render_sidebar_header()

    # 3. Initialize Single Explicit Streamlit Navigation Router
    pg = st.navigation(list(pages_map.values()), position="sidebar")

    # 4. Render Bottom Sidebar Brand Footer
    with st.sidebar:
        render_sidebar_footer()

    # 5. Execute Active Selected Page
    pg.run()


if __name__ == "__main__":
    main()
