"""
Rule-Based Financial Event Classifier Layer
Financial Market Regime & Event Intelligence Engine
"""

import re
import pandas as pd

# Exact priority order and regex rules from Notebook 06
EVENT_RULES = [
    ("Monetary Policy", [r"\bfed\b", r"\bfederal reserve\b", r"\binterest rate(s)?\b", r"\brate cut(s)?\b", r"\brate hike(s)?\b", r"\bfomc\b", r"\bmonetary policy\b", r"\bcentral bank(s)?\b"]),
    ("Inflation", [r"\binflation\b", r"\bcpi\b", r"\bppi\b", r"\bconsumer price(s)?\b", r"\bproducer price(s)?\b"]),
    ("Employment / Labor", [r"\bjob(s)?\b", r"\bpayroll(s)?\b", r"\bunemployment\b", r"\bemployment\b", r"\bnonfarm\b", r"\blabor\b"]),
    ("Earnings", [r"\bearning(s)?\b", r"\brevenue(s)?\b", r"\bprofit(s)?\b", r"\bquarterly result(s)?\b", r"\bguidance\b", r"\beps\b"]),
    ("M&A / Corporate Action", [r"\bacquisition(s)?\b", r"\bmerger(s)?\b", r"\btakeover(s)?\b", r"\bbuyout(s)?\b", r"\bspin-off(s)?\b", r"\bdeal(s)?\b"]),
    ("Geopolitical", [r"\bwar\b", r"\bsanction(s)?\b", r"\btariff(s)?\b", r"\btrade tension(s)?\b", r"\bconflict(s)?\b", r"\bgeopoliti\w*"]),
    ("Commodities", [r"\boil\b", r"\bcrude\b", r"\bgold\b", r"\bnatural gas\b", r"\bcommodity\b", r"\bbrent\b"]),
    ("Market / Index", [r"\bs&p 500\b", r"\bs&p\b", r"\bnasdaq\b", r"\bdow\b", r"\bstock(s)?\b", r"\bequity\b", r"\bequities\b", r"\bmarket rally\b", r"\bmarket selloff\b", r"\bwall street\b"])
]

EVENT_CATEGORIES = [
    "Monetary Policy",
    "Inflation",
    "Employment / Labor",
    "Earnings",
    "M&A / Corporate Action",
    "Geopolitical",
    "Commodities",
    "Market / Index",
    "Other"
]


def classify_headline(headline: str) -> tuple[str, str]:
    """
    Classifies a financial news headline using rule-based keyword pattern matching.
    Returns (event_type, event_trigger).
    """
    if not isinstance(headline, str) or not headline.strip():
        return "Other", "N/A"
        
    clean_text = headline.lower().strip()
    
    for category, patterns in EVENT_RULES:
        for pattern in patterns:
            match = re.search(pattern, clean_text)
            if match:
                return category, match.group(0)
                
    return "Other", "N/A"


def apply_event_classification(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies transparent rule-based event classification to a DataFrame of financial headlines.
    """
    if df.empty or "headline" not in df.columns:
        df_out = df.copy()
        df_out["event_type"] = []
        df_out["event_trigger"] = []
        return df_out
        
    df_out = df.copy()
    class_results = [classify_headline(h) for h in df_out["headline"]]
    
    df_out["event_type"] = [r[0] for r in class_results]
    df_out["event_trigger"] = [r[1] for r in class_results]
    
    return df_out
