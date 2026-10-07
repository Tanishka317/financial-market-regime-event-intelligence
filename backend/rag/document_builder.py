"""
RAG Document Builder Module
Financial Market Regime & Event Intelligence Engine

Converts existing PostgreSQL records (news, news_analysis, event_market_analysis)
into structured textual documents with preserved metadata for vector indexing.
"""

import logging
from typing import List, Dict, Any, Optional
from sqlalchemy import select
from sqlalchemy.orm import Session
from backend.database.connection import get_session
from backend.database.models import News, NewsAnalysis, EventMarketAnalysis

logger = logging.getLogger("rag.document_builder")


def build_rag_documents(db: Optional[Session] = None) -> List[Dict[str, Any]]:
    """
    Queries joined PostgreSQL news, news_analysis, and event_market_analysis records
    and builds structured textual documents with rich metadata.

    Accepts:
        db: Optional SQLAlchemy Session. If None, a context-managed session is created.

    Returns:
        List of document dictionaries formatted as:
        {
            "doc_id": str,
            "text": str,
            "metadata": dict
        }
    """
    if db is None:
        with get_session() as session:
            return _build_documents_from_session(session)
    else:
        return _build_documents_from_session(db)


def _build_documents_from_session(session: Session) -> List[Dict[str, Any]]:
    stmt = (
        select(
            News.news_id,
            News.published_at,
            News.headline,
            News.publisher,
            News.query_ticker,
            News.summary,
            News.url,
            NewsAnalysis.sentiment_label,
            NewsAnalysis.sentiment_score,
            NewsAnalysis.event_type,
            NewsAnalysis.event_trigger,
            EventMarketAnalysis.event_date,
            EventMarketAnalysis.event_day_return,
            EventMarketAnalysis.next_day_return,
            EventMarketAnalysis.five_day_forward_return,
            EventMarketAnalysis.volatility_20,
            EventMarketAnalysis.hmm_state,
        )
        .outerjoin(NewsAnalysis, News.news_id == NewsAnalysis.news_id)
        .outerjoin(EventMarketAnalysis, News.news_id == EventMarketAnalysis.news_id)
        .order_by(News.published_at.desc())
    )

    results = session.execute(stmt).all()
    documents = []

    for row in results:
        published_at_str = row.published_at.isoformat() if row.published_at else "Unknown"
        event_date_str = str(row.event_date) if row.event_date else None
        ticker = row.query_ticker if row.query_ticker else "UNKNOWN"
        publisher = row.publisher if row.publisher else "Unknown"
        headline = row.headline if row.headline else ""
        summary = row.summary if row.summary else None
        url = row.url if row.url else None

        sentiment_label = row.sentiment_label if row.sentiment_label else "neutral"
        sentiment_score = float(row.sentiment_score) if row.sentiment_score is not None else None
        event_type = row.event_type if row.event_type else "Other"
        event_trigger = row.event_trigger if row.event_trigger else None

        hmm_state = int(row.hmm_state) if row.hmm_state is not None else None
        event_day_return = float(row.event_day_return) if row.event_day_return is not None else None
        next_day_return = float(row.next_day_return) if row.next_day_return is not None else None
        five_day_forward_return = float(row.five_day_forward_return) if row.five_day_forward_return is not None else None
        volatility_20 = float(row.volatility_20) if row.volatility_20 is not None else None

        # Build clean textual document representation
        text_lines = [
            f"Headline: {headline}",
            f"Publisher: {publisher} | Ticker: {ticker} | Published Date: {published_at_str}",
        ]
        if summary:
            text_lines.append(f"Summary: {summary}")

        score_str = f"{sentiment_score:.4f}" if sentiment_score is not None else "N/A"
        text_lines.append(
            f"Event Analysis: Event Type: {event_type} | Sentiment: {sentiment_label} (Score: {score_str})"
        )
        if event_trigger:
            text_lines.append(f"Event Trigger Keywords: {event_trigger}")

        if event_date_str or hmm_state is not None:
            market_parts = [f"Event Date: {event_date_str or 'N/A'}"]
            if hmm_state is not None:
                market_parts.append(f"HMM Regime State: {hmm_state}")
            if next_day_return is not None:
                market_parts.append(f"Next-Day Return: {next_day_return:+.2f}%")
            if five_day_forward_return is not None:
                market_parts.append(f"5-Day Forward Return: {five_day_forward_return:+.2f}%")
            if volatility_20 is not None:
                market_parts.append(f"20-Day Volatility: {volatility_20:.2f}%")

            text_lines.append(f"Event Market Alignment: {' | '.join(market_parts)}")

        doc_text = "\n".join(text_lines)

        metadata = {
            "news_id": row.news_id,
            "published_at": published_at_str,
            "event_date": event_date_str,
            "ticker": ticker,
            "publisher": publisher,
            "event_type": event_type,
            "sentiment_label": sentiment_label,
            "sentiment_score": sentiment_score,
            "hmm_state": hmm_state,
            "headline": headline,
            "url": url,
        }

        documents.append({
            "doc_id": row.news_id,
            "text": doc_text,
            "metadata": metadata,
        })

    logger.info(f"Built {len(documents)} RAG documents from PostgreSQL records.")
    return documents
