"""
Inspect Streamlit Runtime Page Registry
"""
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import streamlit as st
from streamlit.source_util import get_pages

def inspect_pages():
    print("=== Testing Streamlit Pages Discovery ===")
    try:
        pages = get_pages("app/main.py")
        print("Discovered pages dictionary keys:")
        for page_hash, page_info in pages.items():
            print(f"  Hash: {page_hash} | Name: {page_info.get('page_name')} | Script: {page_info.get('script_path')}")
    except Exception as e:
        print("get_pages error:", e)

if __name__ == "__main__":
    inspect_pages()
