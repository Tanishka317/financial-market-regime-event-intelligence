"""
Analyst Query Intent & Entity Recognition Layer
Financial Market Regime & Event Intelligence Engine
"""

import re
from typing import Dict, Any, Optional

# Supported Analyst Intent Constants
INTENT_LATEST_REGIME = "latest_regime"
INTENT_LATEST_MARKET_DATA = "latest_market_data"
INTENT_MARKET_METRICS = "market_metrics"
INTENT_REGIME_DISTRIBUTION = "regime_distribution"
INTENT_REGIME_MARKET_BEHAVIOR = "regime_market_behavior"
INTENT_RECENT_NEWS = "recent_news"
INTENT_NEWS_BY_SENTIMENT = "news_by_sentiment"
INTENT_EVENT_PERFORMANCE = "event_performance"
INTENT_EVENT_RANKING = "event_ranking"
INTENT_EVENT_REGIME = "event_regime"
INTENT_UNSUPPORTED = "unsupported"

# Event categories mapping for entity extraction
EVENT_CATEGORY_PATTERNS = {
    "Monetary Policy": [r"\bmonetary policy\b", r"\bfed\b", r"\bfederal reserve\b", r"\brate(s)?\b", r"\bfomc\b", r"\bcentral bank\b"],
    "Inflation": [r"\binflation\b", r"\bcpi\b", r"\bppi\b", r"\bprice(s)?\b"],
    "Employment / Labor": [r"\bemployment\b", r"\blabor\b", r"\bjob(s)?\b", r"\bpayroll(s)?\b", r"\bunemployment\b", r"\bnonfarm\b"],
    "Earnings": [r"\bearning(s)?\b", r"\brevenue\b", r"\bprofit(s)?\b", r"\beps\b", r"\bguidance\b"],
    "M&A / Corporate Action": [r"\bm&a\b", r"\bmerger(s)?\b", r"\bacquisition(s)?\b", r"\bdeal(s)?\b", r"\btakeover\b"],
    "Geopolitical": [r"\bgeopoliti\w*", r"\bwar\b", r"\bsanction(s)?\b", r"\btariff(s)?\b", r"\btrade tension(s)?\b"],
    "Commodities": [r"\bcommodity\b", r"\bcommodities\b", r"\boil\b", r"\bgold\b", r"\bcrude\b", r"\bgas\b"],
    "Market / Index": [r"\bmarket / index\b", r"\bs&p\b", r"\bnasdaq\b", r"\bdow\b", r"\bequity\b", r"\bequities\b", r"\bstock(s)?\b"]
}

# Ticker alias mapping
TICKER_MAP = {
    "^GSPC": [r"\b\^gspc\b", r"\bs&p 500\b", r"\bs&p500\b", r"\bs&p\b", r"\bsp500\b", r"\bindex\b"],
    "SPY": [r"\bspy\b", r"\bspdr\b"],
    "^VIX": [r"\b\^vix\b", r"\bvix\b", r"\bvolatility index\b"],
    "TLT": [r"\btlt\b", r"\btreasury\b", r"\bbond(s)?\b"],
    "GLD": [r"\bgld\b", r"\bgold\b"],
    "QQQ": [r"\bqqq\b", r"\bnasdaq\b", r"\btech ETF\b"],
    "AAPL": [r"\baapl\b", r"\bapple\b"],
    "MSFT": [r"\bmsft\b", r"\bmicrosoft\b"],
    "NVDA": [r"\bnvda\b", r"\bnvidia\b"],
    "AMZN": [r"\bamzn\b", r"\bamazon\b"],
    "JPM": [r"\bjpm\b", r"\bjpmorgan\b", r"\bchase\b"],
    "GS": [r"\bgs\b", r"\bgoldman\b", r"\bgoldman sachs\b"]
}


def extract_event_type(question_text: str) -> Optional[str]:
    """
    Extracts explicit event category from question text.
    """
    text_lower = question_text.lower()
    for cat_name, patterns in EVENT_CATEGORY_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, text_lower):
                return cat_name
    return None


def extract_ticker(question_text: str, default_ticker: Optional[str] = "^GSPC") -> str:
    """
    Extracts asset ticker from question text using symbol & alias matching.
    """
    text_lower = question_text.lower()
    for symbol, patterns in TICKER_MAP.items():
        for pat in patterns:
            if re.search(pat, text_lower):
                return symbol
    return default_ticker or "^GSPC"


def extract_sentiment_label(question_text: str) -> Optional[str]:
    """
    Extracts sentiment label filter (positive, negative, neutral) from question text.
    """
    text_lower = question_text.lower()
    if re.search(r"\b(negative|bearish|bad|pessimistic)\b", text_lower):
        return "negative"
    if re.search(r"\b(positive|bullish|good|optimistic)\b", text_lower):
        return "positive"
    if re.search(r"\b(neutral|mixed)\b", text_lower):
        return "neutral"
    return None


def extract_metric(question_text: str) -> Optional[str]:
    """
    Extracts quantitative market metric from question text.
    """
    text_lower = question_text.lower()
    if re.search(r"\b(volatility|volatile|risk|std dev)\b", text_lower):
        return "volatility"
    if re.search(r"\b(momentum|trend)\b", text_lower):
        return "momentum"
    if re.search(r"\b(drawdown|drop from peak|max drop)\b", text_lower):
        return "drawdown"
    if re.search(r"\b(correlation|correlations|corr|cross-asset|relationship)\b", text_lower):
        return "correlation"
    if re.search(r"\b(return|returns|daily return|daily returns)\b", text_lower):
        return "returns"
    if re.search(r"\b(close|closing price|price|stock price)\b", text_lower):
        return "close"
    return None


def extract_ranking(question_text: str) -> Optional[str]:
    """
    Extracts ranking or comparison direction from question text.
    """
    text_lower = question_text.lower()
    if re.search(r"\b(best|highest|top|most profitable|highest return)\b", text_lower):
        return "best"
    if re.search(r"\b(worst|lowest|bottom|most negative)\b", text_lower):
        return "worst"
    if re.search(r"\b(compare|comparison|ranking|rank|most frequent)\b", text_lower):
        return "ranking"
    return None


def classify_intent(question: str) -> Dict[str, Any]:
    """
    Classifies user natural-language questions into analytical intent categories
    and extracts relevant entities (ticker, event_type, sentiment, metric, ranking).
    """
    q_lower = question.strip().lower()

    if not q_lower:
        return {
            "intent": INTENT_UNSUPPORTED,
            "event_type": None,
            "ticker": "^GSPC",
            "sentiment_label": None,
            "metric": None,
            "ranking": None
        }

    extracted_event = extract_event_type(question)
    extracted_ticker = extract_ticker(question)
    extracted_sentiment = extract_sentiment_label(question)
    extracted_metric = extract_metric(question)
    extracted_ranking = extract_ranking(question)

    # Detect out-of-scope non-financial / forward projection questions
    if re.search(r"\b(tomorrow|next week|next year|predict future|who won|capital of|weather|president|election|joke)\b", q_lower):
        return {
            "intent": INTENT_UNSUPPORTED,
            "event_type": None,
            "ticker": extracted_ticker,
            "sentiment_label": None,
            "metric": None,
            "ranking": None
        }

    # 1. Event Ranking / Comparison ("Which events performed best?", "Which events had the highest forward return?")
    if (extracted_ranking is not None or "compare" in q_lower or "comparison" in q_lower) and ("event" in q_lower or extracted_event is not None):
        return {
            "intent": INTENT_EVENT_RANKING,
            "event_type": extracted_event,
            "ticker": extracted_ticker,
            "sentiment_label": extracted_sentiment,
            "metric": extracted_metric,
            "ranking": extracted_ranking or "best"
        }

    # 2. Event Regime Breakdown ("What was the market regime during Monetary Policy events?", "How did events behave across regimes?")
    if (("regime" in q_lower or "hmm" in q_lower or "state" in q_lower)
        and ("during" in q_lower or "when" in q_lower or "across" in q_lower or "in event" in q_lower or "with event" in q_lower or "for event" in q_lower)
        and (extracted_event is not None or "event" in q_lower)):
        return {
            "intent": INTENT_EVENT_REGIME,
            "event_type": extracted_event,
            "ticker": extracted_ticker,
            "sentiment_label": extracted_sentiment,
            "metric": extracted_metric,
            "ranking": extracted_ranking
        }

    # 3. Regime Distribution / Frequency ("How often is the market in a bull regime?", "Show regime distribution", "Regime frequency")
    if ("regime" in q_lower or "state" in q_lower) and ("frequency" in q_lower or "distribution" in q_lower or "how often" in q_lower or "how many days" in q_lower or "historically" in q_lower or "time spent" in q_lower):
        return {
            "intent": INTENT_REGIME_DISTRIBUTION,
            "event_type": None,
            "ticker": extracted_ticker,
            "sentiment_label": None,
            "metric": None,
            "ranking": None
        }

    # 4. Market Behavior by Regime ("What is the market volatility during Low-Vol Bull regime?", "Market performance by regime")
    if ("regime" in q_lower or "state" in q_lower) and ("behavior" in q_lower or "performance by regime" in q_lower or "by regime" in q_lower or "stat" in q_lower or ("bull" in q_lower and "regime" in q_lower) or ("bear" in q_lower and "regime" in q_lower)):
        return {
            "intent": INTENT_REGIME_MARKET_BEHAVIOR,
            "event_type": None,
            "ticker": extracted_ticker,
            "sentiment_label": None,
            "metric": extracted_metric,
            "ranking": None
        }

    # 5. Latest Regime Queries ("What's the current regime?", "Which regime is the market in?", "Tell me about the latest S&P 500 regime")
    if ("regime" in q_lower or "hmm state" in q_lower) and ("latest" in q_lower or "current" in q_lower or "today" in q_lower or "now" in q_lower or "which regime" in q_lower or q_lower.startswith("what is the regime") or q_lower.startswith("what is market regime") or q_lower.endswith("regime")):
        return {
            "intent": INTENT_LATEST_REGIME,
            "event_type": None,
            "ticker": extracted_ticker,
            "sentiment_label": None,
            "metric": None,
            "ranking": None
        }

    # 6. Event Performance ("How did Inflation perform?", "What happened after inflation events?")
    if (("perform" in q_lower or "performance" in q_lower or "happen" in q_lower or "after" in q_lower or "impact" in q_lower or "return" in q_lower)
        and (extracted_event is not None or "event" in q_lower)):
        return {
            "intent": INTENT_EVENT_PERFORMANCE,
            "event_type": extracted_event,
            "ticker": extracted_ticker,
            "sentiment_label": extracted_sentiment,
            "metric": extracted_metric,
            "ranking": extracted_ranking
        }

    # 7. News by Sentiment ("Show recent negative news", "Positive news headlines")
    if ("news" in q_lower or "headline" in q_lower or "article" in q_lower) and (extracted_sentiment is not None):
        return {
            "intent": INTENT_NEWS_BY_SENTIMENT,
            "event_type": extracted_event,
            "ticker": extracted_ticker,
            "sentiment_label": extracted_sentiment,
            "metric": None,
            "ranking": None
        }

    # 8. Recent News ("What recent financial news/events are available?", "Show recent news", "News about inflation")
    if ("news" in q_lower or "headline" in q_lower or "article" in q_lower or "events available" in q_lower or "recent events" in q_lower):
        return {
            "intent": INTENT_RECENT_NEWS,
            "event_type": extracted_event,
            "ticker": extracted_ticker,
            "sentiment_label": extracted_sentiment,
            "metric": None,
            "ranking": None
        }

    # 9. Market Metrics Queries ("How volatile has the market been?", "What is the market momentum?", "Show cross-asset correlations")
    if extracted_metric is not None:
        return {
            "intent": INTENT_MARKET_METRICS,
            "event_type": None,
            "ticker": extracted_ticker,
            "sentiment_label": None,
            "metric": extracted_metric,
            "ranking": None
        }

    # 10. Latest Market Data Fallback ("What is the latest market data?", "Show market price")
    if ("market data" in q_lower or "price" in q_lower or "market" in q_lower or "close" in q_lower or "snapshot" in q_lower):
        return {
            "intent": INTENT_LATEST_MARKET_DATA,
            "event_type": None,
            "ticker": extracted_ticker,
            "sentiment_label": None,
            "metric": None,
            "ranking": None
        }

    # Fallback to Unsupported
    return {
        "intent": INTENT_UNSUPPORTED,
        "event_type": extracted_event,
        "ticker": extracted_ticker,
        "sentiment_label": extracted_sentiment,
        "metric": extracted_metric,
        "ranking": extracted_ranking
    }
