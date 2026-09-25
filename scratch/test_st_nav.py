"""
Test st.navigation behavior with Streamlit
"""

import sys
import os
import streamlit as st

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from app.utils.theme import apply_theme
from app.pages import overview, regimes, events, explainability, model_health

st.set_page_config(
    page_title="Financial Market Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

apply_theme()

def run_test():
    overview_page = st.Page(overview.render, title="Overview", icon="📊", default=True)
    regimes_page = st.Page(regimes.render, title="Market Regimes", icon="📈")
    events_page = st.Page(events.render, title="News & Events", icon="📰")
    explainability_page = st.Page(explainability.render, title="Explainability", icon="🔍")
    health_page = st.Page(model_health.render, title="Model & Data Health", icon="⚙️")

    pg = st.navigation({
        "FMIE Analytics": [overview_page, regimes_page, events_page, explainability_page, health_page]
    })

    pg.run()

if __name__ == "__main__":
    run_test()
