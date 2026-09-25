"""
Check Streamlit page discovery state
"""
import urllib.request
import json

def check_streamlit_health():
    try:
        url = "http://localhost:8506/_stcore/health"
        req = urllib.request.urlopen(url)
        print("Health status code:", req.getcode())
        print("Health response:", req.read().decode())
    except Exception as e:
        print("Health check error:", e)

if __name__ == "__main__":
    check_streamlit_health()
