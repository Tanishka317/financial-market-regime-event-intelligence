import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

print("Python path:", sys.path)

print("Importing app modules...")
from app.utils.theme import apply_theme, get_plotly_dark_layout
from app.components.cards import render_kpi_card, render_empty_state, render_status_badge
from app.components.layout import render_page_header, render_section_header
from app.components.sidebar import render_sidebar
from app.pages import overview, regimes, events, explainability, model_health

print("Modules imported successfully!")
