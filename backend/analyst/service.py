"""
Financial Intelligence Analyst Service
Financial Market Regime & Event Intelligence Engine
"""

import logging
from typing import Dict, Any, Optional
from sqlalchemy.orm import Session

from backend.database.market_data_repo import get_latest_market_data
from backend.database.regime_repo import get_latest_regime_prediction
from backend.database.news_repo import get_news_records
from backend.database.event_market_repo import get_event_market_analysis_records
from backend.analyst.intents import (
    classify_intent,
    INTENT_LATEST_REGIME,
    INTENT_LATEST_MARKET_DATA,
    INTENT_RECENT_NEWS,
    INTENT_EVENT_PERFORMANCE,
    INTENT_EVENT_REGIME,
    INTENT_UNSUPPORTED
)

logger = logging.getLogger("analyst_service")


def execute_analyst_query(
    db: Session,
    question: str,
    ticker: Optional[str] = "^GSPC"
) -> Dict[str, Any]:
    """
    Executes a deterministic, data-grounded Financial Analyst query.
    1. Classifies user question into supported analyst intent.
    2. Queries database using existing repositories.
    3. Synthesizes a structured answer and supporting context.
    """
    intent_info = classify_intent(question)
    intent = intent_info["intent"]
    extracted_event = intent_info["event_type"]
    query_ticker = ticker or intent_info["ticker"] or "^GSPC"

    if intent == INTENT_LATEST_REGIME:
        regime = get_latest_regime_prediction(db, ticker=query_ticker)
        if not regime:
            return {
                "question": question,
                "intent": intent,
                "answer": f"No market regime predictions found for ticker '{query_ticker}' in the database.",
                "supporting_data": {"ticker": query_ticker, "record_found": False}
            }

        answer = (
            f"The latest decoded market regime for {regime.ticker} on {regime.date} "
            f"is '{regime.regime_label}' (HMM State {regime.hmm_state}, model: {regime.model_version})."
        )
        supporting_data = {
            "ticker": regime.ticker,
            "date": str(regime.date),
            "hmm_state": regime.hmm_state,
            "regime_label": regime.regime_label,
            "model_version": regime.model_version,
            "created_at": regime.created_at.isoformat() if regime.created_at else None
        }
        return {
            "question": question,
            "intent": intent,
            "answer": answer,
            "supporting_data": supporting_data
        }

    elif intent == INTENT_LATEST_MARKET_DATA:
        market = get_latest_market_data(db, ticker=query_ticker)
        if not market:
            return {
                "question": question,
                "intent": intent,
                "answer": f"No market data records found for ticker '{query_ticker}' in the database.",
                "supporting_data": {"ticker": query_ticker, "record_found": False}
            }

        daily_ret_pct = f"{float(market.daily_return) * 100:+.2f}%" if market.daily_return is not None else "N/A"
        vol_pct = f"{float(market.volatility_20) * 100:.2f}%" if market.volatility_20 is not None else "N/A"
        
        answer = (
            f"The latest market observation for {market.ticker} on {market.date} shows a closing price of "
            f"${float(market.close):.2f} with a daily return of {daily_ret_pct} and 20-day volatility of {vol_pct}."
        )
        supporting_data = {
            "ticker": market.ticker,
            "date": str(market.date),
            "close": float(market.close),
            "daily_return": float(market.daily_return) if market.daily_return is not None else None,
            "volatility_20": float(market.volatility_20) if market.volatility_20 is not None else None,
            "momentum_20": float(market.momentum_20) if market.momentum_20 is not None else None,
            "drawdown": float(market.drawdown) if market.drawdown is not None else None
        }
        return {
            "question": question,
            "intent": intent,
            "answer": answer,
            "supporting_data": supporting_data
        }

    elif intent == INTENT_RECENT_NEWS:
        news_list = get_news_records(db, ticker=None if query_ticker == "^GSPC" else query_ticker, limit=5)
        if not news_list:
            return {
                "question": question,
                "intent": intent,
                "answer": "No financial news records were found in the database.",
                "supporting_data": {"count": 0, "articles": []}
            }

        articles = [
            {
                "news_id": item.news_id,
                "published_at": item.published_at.isoformat() if item.published_at else None,
                "headline": item.headline,
                "publisher": item.publisher,
                "query_ticker": item.query_ticker
            }
            for item in news_list
        ]
        top_headline = news_list[0].headline
        top_pub = news_list[0].publisher
        answer = (
            f"Retrieved {len(articles)} recent financial news items. "
            f"Latest headline: '{top_headline}' published by {top_pub}."
        )
        return {
            "question": question,
            "intent": intent,
            "answer": answer,
            "supporting_data": {"count": len(articles), "articles": articles}
        }

    elif intent == INTENT_EVENT_PERFORMANCE:
        records = get_event_market_analysis_records(db, event_type=extracted_event, limit=50)
        if not records:
            event_name = f"'{extracted_event}'" if extracted_event else "specified"
            return {
                "question": question,
                "intent": intent,
                "answer": f"No event-market analysis records found for {event_name} events in the database.",
                "supporting_data": {"event_type": extracted_event, "record_count": 0, "records": []}
            }

        next_day_rets = [r["next_day_return"] for r in records if r.get("next_day_return") is not None]
        five_day_rets = [r["five_day_forward_return"] for r in records if r.get("five_day_forward_return") is not None]

        avg_next_day = (sum(next_day_rets) / len(next_day_rets) * 100) if next_day_rets else None
        avg_5day = (sum(five_day_rets) / len(five_day_rets) * 100) if five_day_rets else None

        avg_next_str = f"{avg_next_day:+.2f}%" if avg_next_day is not None else "N/A"
        avg_5day_str = f"{avg_5day:+.2f}%" if avg_5day is not None else "N/A"
        target_event_lbl = extracted_event or "Financial"

        answer = (
            f"Analyzed {len(records)} '{target_event_lbl}' event records. "
            f"Average next-day return was {avg_next_str} and average 5-day forward return was {avg_5day_str}."
        )
        supporting_data = {
            "event_type": target_event_lbl,
            "record_count": len(records),
            "avg_next_day_return_pct": round(avg_next_day, 4) if avg_next_day is not None else None,
            "avg_5day_forward_return_pct": round(avg_5day, 4) if avg_5day is not None else None,
            "sample_records": records[:5]
        }
        return {
            "question": question,
            "intent": intent,
            "answer": answer,
            "supporting_data": supporting_data
        }

    elif intent == INTENT_EVENT_REGIME:
        records = get_event_market_analysis_records(db, event_type=extracted_event, limit=50)
        if not records:
            event_name = f"'{extracted_event}'" if extracted_event else "specified"
            return {
                "question": question,
                "intent": intent,
                "answer": f"No event-market analysis records found for {event_name} events in the database.",
                "supporting_data": {"event_type": extracted_event, "record_count": 0, "state_breakdown": {}}
            }

        state_counts = {}
        for r in records:
            st_val = r["hmm_state"]
            state_counts[st_val] = state_counts.get(st_val, 0) + 1

        breakdown_str = ", ".join([f"HMM State {st}: {cnt} occurrence(s)" for st, cnt in sorted(state_counts.items())])
        target_event_lbl = extracted_event or "Financial"

        answer = (
            f"During {len(records)} '{target_event_lbl}' events, the market regime breakdown was: {breakdown_str}."
        )
        supporting_data = {
            "event_type": target_event_lbl,
            "record_count": len(records),
            "state_breakdown": state_counts,
            "sample_records": records[:5]
        }
        return {
            "question": question,
            "intent": intent,
            "answer": answer,
            "supporting_data": supporting_data
        }

    else:
        answer = (
            "I could not match your question to a supported financial intelligence intent. "
            "Supported topics include: 1) Latest market regime, 2) Latest market data, "
            "3) Recent financial news, 4) Event performance analysis (e.g. 'How did Inflation perform?'), "
            "and 5) Market regimes during events (e.g. 'What was the market regime during Monetary Policy events?')."
        )
        return {
            "question": question,
            "intent": INTENT_UNSUPPORTED,
            "answer": answer,
            "supporting_data": {
                "supported_intents": [
                    INTENT_LATEST_REGIME,
                    INTENT_LATEST_MARKET_DATA,
                    INTENT_RECENT_NEWS,
                    INTENT_EVENT_PERFORMANCE,
                    INTENT_EVENT_REGIME
                ]
            }
        }
