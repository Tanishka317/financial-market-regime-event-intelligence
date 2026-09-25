"""
Verification script for navigation bug fix
"""
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import streamlit as st
from app.components.sidebar import get_pages, navigate_to, render_sidebar_header, render_sidebar_footer
from streamlit.runtime.pages_manager import PagesManager

def test_pages_manager():
    print("=== Testing Streamlit Pages Discovery ===")
    pm = PagesManager("app/main.py")
    discovered = pm.get_pages()
    print("Discovered pages count:", len(discovered))
    for page_hash, page_info in discovered.items():
        print(f"  Script: {page_info.get('script_path')} | Name: {page_info.get('page_name')}")
    
    assert len(discovered) == 1, f"Expected 1 entrypoint page script in PagesManager, got {len(discovered)}"
    print("[OK] PagesManager auto-discovery duplicates successfully suppressed!\n")

def test_pages_registry():
    print("=== Testing Centralized PAGES Registry ===")
    pages = get_pages()
    expected_titles = ["Overview", "Market Regimes", "News & Events", "Explainability", "Model & Data Health"]
    for title in expected_titles:
        assert title in pages, f"Missing page {title} in PAGES registry!"
        page_obj = pages[title]
        assert isinstance(page_obj, st.Page), f"{title} is not a valid st.Page object!"
        print(f"  [OK] Registered Page: '{title}' -> {page_obj}")
    print("[OK] All 5 dashboard pages properly registered as st.Page objects!\n")

if __name__ == "__main__":
    print("==========================================")
    print("RUNNING NAVIGATION BUG FIX VERIFICATION")
    print("==========================================\n")
    test_pages_manager()
    test_pages_registry()
    print("ALL NAVIGATION VERIFICATION TESTS PASSED!")
