"""
Test st.navigation rendered sidebar structure
"""
import urllib.request

def fetch_page_html():
    try:
        url = "http://localhost:8506/"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        html = urllib.request.urlopen(req).read().decode("utf-8")
        print("Fetched HTML length:", len(html))
        # Check if 'FMIE Analytics' is present
        print("Contains FMIE Analytics:", "FMIE Analytics" in html)
        print("Contains main:", "main" in html)
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    fetch_page_html()
