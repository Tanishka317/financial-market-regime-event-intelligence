"""
Analyst Query Intent & Entity Recognition Layer
Financial Market Regime & Event Intelligence Engine
"""

import re
from typing import Dict, Any, Optional

# Supported Analyst Intent Constants
INTENT_LATEST_REGIME = "latest_regime"
INTENT_LATEST_MARKET_DATA = "latest_market_data"
INTENT_RECENT_NEWS = "recent_news"
INTENT_EVENT_PERFORMANCE = "event_performance"
INTENT_EVENT_REGIME = "event_regime"
INTENT_UNSUPPORTED = "unsupported"

# Event categories mapping for entity extraction
EVENT_CATEGORY_PATTERNS = {
    "Monetary Policy": [r"\bmonetary policy\b", r"\bfed\b", r"\bfederal reserve\b", r"\brate(s)?\b", r"\bfomc\b"],
    "Inflation": [r"\binflation\b", r"\bcpi\b", r"\bppi\b", r"\bprice(s)?\b"],
    "Employment / Labor": [r"\bemployment\b", r"\blabor\b", r"\bjob(s)?\b", r"\bpayroll(s)?\b", r"\bunemployment\b"],
    "Earnings": [r"\bearning(s)?\b", r"\brevenue\b", r"\bprofit(s)?\b", r"\beps\b"],
    "M&A / Corporate Action": [r"\bm&a\b", r"\bmerger(s)?\b", r"\bacquisition(s)?\b", r"\bdeal(s)?\b"],
    "Geopolitical": [r"\bgeopoliti\w*", r"\bwar\b", r"\bsanction(s)?\b", r"\btariff(s)?\b"],
    "Commodities": [r"\bcommodity\b", r"\bcommodities\b", r"\boil\b", r"\bgold\b"],
    "Market / Index": [r"\bmarket / index\b", r"\bs&p\b", r"\bnasdaq\b", r"\bdow\b", r"\bequity\b"]
}


def extract_event_type(question_text: str) -> Optional[str]:
    """
    Extracts explicit event category from user question text using pattern matching.
    """
    text_lower = question_text.lower()
    for cat_name, patterns in EVENT_CATEGORY_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, text_lower):
                return cat_name
    return None


def extract_ticker(question_text: str, default_ticker: Optional[str] = "^GSPC") -> str:
    """
    Extracts asset ticker from question text if specified, otherwise returns default_ticker.
    """
    text_upper = question_text.upper()
    known_tickers = ["^GSPC", "SPY", "^VIX", "TLT", "GLD", "QQQ", "AAPL", "MSFT", "NVDA", "AMZN", "JPM", "GS"]
    for t in known_tickers:
        if t in text_upper or (t.startswith("^") and t[1:] in text_upper):
            return t
    return default_ticker or "^GSPC"


def classify_intent(question: str) -> Dict[str, Any]:
    """
    Classifies the user question into one of the supported analyst intents
    and extracts relevant entities (ticker, event_type).
    """
    q_lower = question.strip().lower()

    if not q_lower:
        return {
            "intent": INTENT_UNSUPPORTED,
            "event_type": None,
            "ticker": "^GSPC"
        }

    extracted_event = extract_event_type(question)
    extracted_ticker = extract_ticker(question)

    # 1. Event Regime Intent ("What was the market regime during Monetary Policy events?")
    if (("regime" in q_lower and ("during" in q_lower or "when" in q_lower or "for event" in q_lower or "with event" in q_lower or "in event" in q_lower))
        or ("regime" in q_lower and extracted_event is not None and "latest" not in q_lower and "current" not in q_lower)):
        return {
            "intent": INTENT_EVENT_REGIME,
            "event_type": extracted_event,
            "ticker": extracted_ticker
        }

    # 2. Latest Regime Intent ("What is the latest market regime?")
    if ("regime" in q_lower or "state" in q_lower) and ("latest" in q_lower or "current" in q_lower or "today" in q_lower or "now" in q_lower or q_lower.startswith("what is the regime") or q_lower.startswith("what is market regime")):
        return {
            "intent": INTENT_LATEST_REGIME,
            "event_type": None,
            "ticker": extracted_ticker
        }

    # 3. Latest Market Data Intent ("What is the latest market data?")
    if ("market data" in q_lower or "market price" in q_lower or "stock data" in q_lower or "latest market" in q_lower or ("market" in q_lower and "data" in q_lower)):
        return {
            "intent": INTENT_LATEST_MARKET_DATA,
            "event_type": None,
            "ticker": extracted_ticker
        }

    # 4. Event Performance Intent ("How did Inflation perform?" / "How did Monetary Policy perform?")
    if (("perform" in q_lower or "performance" in q_lower or "return" in q_lower or "impact" in q_lower)
        and (extracted_event is not None or "event" in q_lower)):
        return {
            "intent": INTENT_EVENT_PERFORMANCE,
            "event_type": extracted_event,
            "ticker": extracted_ticker
        }

    # 5. Recent News Intent ("What recent financial news/events are available?")
    if ("news" in q_lower or "headline" in q_lower or "article" in q_lower or "events available" in q_lower or "recent events" in q_lower):
        return {
            "intent": INTENT_RECENT_NEWS,
            "event_type": extracted_event,
            "ticker": extracted_ticker
        }

    # Fallback checking for regime queries
    if "regime" in q_lower:
        return {
            "intent": INTENT_LATEST_REGIME,
            "event_type": None,
            "ticker": extracted_ticker
        }

    return {
        "intent": INTENT_UNSUPPORTED,
        "event_type": None,
        "ticker": extracted_ticker
    }
