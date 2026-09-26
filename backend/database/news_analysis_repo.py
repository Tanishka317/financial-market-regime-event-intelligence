"""
News Analysis Repository & PostgreSQL Persistence Module
Financial Market Regime & Event Intelligence Engine
"""

import logging
from datetime import datetime, timezone
import pandas as pd
from typing import Optional
from sqlalchemy.dialects.postgresql import insert
from backend.database.connection import get_engine
from backend.database.models import NewsAnalysis
from backend.database.news_repo import generate_news_id, persist_news

logger = logging.getLogger("news_analysis_repo")


def sanitize_score(val) -> float:
    """
    Sanitizes sentiment confidence score float value.
    """
    if pd.isna(val) or val is None:
        return 0.0
    try:
        return round(float(val), 6)
    except (ValueError, TypeError):
        return 0.0


def persist_news_analysis(
    analysis_df: pd.DataFrame,
    model_version: str = "finbert_v1"
) -> int:
    """
    Idempotently persists FinBERT sentiment inference and rule-based event classification results
    into PostgreSQL 'news_analysis' table.
    
    Uses PostgreSQL ON CONFLICT DO UPDATE on 'news_id' to prevent duplicate records.
    Automatically ensures underlying raw news rows exist in 'news' table to satisfy Foreign Key constraint.
    
    Accepts:
        analysis_df: DataFrame containing 'headline', 'sentiment_label', 'sentiment_score', 'event_type', 'event_trigger'
        model_version: String identifier for sentiment/event classifier version (default: 'finbert_v1')
        
    Returns:
        Number of processed analysis records.
    """
    if analysis_df is None or analysis_df.empty:
        logger.info("Empty or None DataFrame provided for news analysis persistence.")
        return 0

    # Ensure parent raw news records exist in 'news' table first to prevent FK violations
    persist_news(analysis_df)

    records = []
    seen_ids = set()

    for _, row in analysis_df.iterrows():
        headline_val = row.get("headline")
        if pd.isna(headline_val) or not headline_val or not str(headline_val).strip():
            continue

        # Extract or generate news_id
        if "news_id" in row and pd.notna(row["news_id"]) and str(row["news_id"]).strip():
            news_id = str(row["news_id"]).strip()
        else:
            row_dict = {
                "headline": str(headline_val).strip(),
                "published_at": row.get("published_at"),
                "publisher": row.get("publisher"),
                "query_ticker": row.get("query_ticker"),
                "summary": row.get("summary"),
                "url": row.get("url"),
            }
            news_id = generate_news_id(row_dict)

        if news_id in seen_ids:
            continue
        seen_ids.add(news_id)

        sentiment_lbl = row.get("sentiment_label")
        if pd.isna(sentiment_lbl) or not sentiment_lbl or str(sentiment_lbl).strip().lower() in ["none", "nan"]:
            sentiment_lbl_str = "neutral"
        else:
            sentiment_lbl_str = str(sentiment_lbl).strip().lower()

        sentiment_sc = sanitize_score(row.get("sentiment_score"))

        event_tp = row.get("event_type")
        if pd.isna(event_tp) or not event_tp or str(event_tp).strip().lower() in ["none", "nan"]:
            event_tp_str = "Other"
        else:
            event_tp_str = str(event_tp).strip()

        event_tr = row.get("event_trigger")
        if pd.isna(event_tr) or event_tr is None or str(event_tr).strip().lower() in ["", "none", "nan"]:
            event_tr_str = "N/A"
        else:
            event_tr_str = str(event_tr).strip()

        analyzed_at_val = row.get("analyzed_at")
        if pd.isna(analyzed_at_val) or analyzed_at_val is None:
            analyzed_dt = datetime.now(timezone.utc)
        elif isinstance(analyzed_at_val, str):
            analyzed_dt = pd.to_datetime(analyzed_at_val, utc=True).to_pydatetime()
        elif hasattr(analyzed_at_val, "to_pydatetime"):
            analyzed_dt = analyzed_at_val.to_pydatetime()
        elif isinstance(analyzed_at_val, datetime):
            analyzed_dt = analyzed_at_val
        else:
            analyzed_dt = datetime.now(timezone.utc)

        if analyzed_dt.tzinfo is None:
            analyzed_dt = analyzed_dt.replace(tzinfo=timezone.utc)

        records.append({
            "news_id": news_id,
            "sentiment_label": sentiment_lbl_str,
            "sentiment_score": sentiment_sc,
            "event_type": event_tp_str,
            "event_trigger": event_tr_str,
            "analyzed_at": analyzed_dt,
            "model_version": model_version,
        })

    if not records:
        return 0

    engine = get_engine()

    stmt = insert(NewsAnalysis).values(records)
    upsert_stmt = stmt.on_conflict_do_update(
        index_elements=["news_id"],
        set_={
            "sentiment_label": stmt.excluded.sentiment_label,
            "sentiment_score": stmt.excluded.sentiment_score,
            "event_type": stmt.excluded.event_type,
            "event_trigger": stmt.excluded.event_trigger,
            "analyzed_at": stmt.excluded.analyzed_at,
            "model_version": stmt.excluded.model_version,
        }
    )

    with engine.begin() as conn:
        conn.execute(upsert_stmt)

    logger.info(
        f"Successfully persisted {len(records)} news analysis records into PostgreSQL 'news_analysis' "
        f"table (version: '{model_version}')."
    )
    return len(records)
