"""
Event-Market Analysis Repository & PostgreSQL Persistence Module
Financial Market Regime & Event Intelligence Engine
"""

import logging
from datetime import datetime, timezone
import pandas as pd
from typing import Optional, List, Dict, Any
from sqlalchemy import select
from sqlalchemy.orm import Session
from backend.database.connection import get_engine, get_session
from backend.database.models import News, NewsAnalysis, EventMarketAnalysis
from backend.database.news_repo import generate_news_id

logger = logging.getLogger("event_market_repo")


def get_event_market_analysis_records(
    db: Session,
    event_type: Optional[str] = None,
    hmm_state: Optional[int] = None,
    limit: int = 100
) -> List[Dict[str, Any]]:
    """
    Retrieves event-market analysis records joined with news and news_analysis data.
    Allows optional filtering by event_type and hmm_state, ordered by event_date descending.
    """
    stmt = (
        select(
            EventMarketAnalysis.news_id,
            EventMarketAnalysis.event_date,
            EventMarketAnalysis.event_day_return,
            EventMarketAnalysis.next_day_return,
            EventMarketAnalysis.five_day_forward_return,
            EventMarketAnalysis.volatility_20,
            EventMarketAnalysis.hmm_state,
            News.headline,
            NewsAnalysis.event_type,
            NewsAnalysis.sentiment_label,
            NewsAnalysis.sentiment_score,
        )
        .join(News, EventMarketAnalysis.news_id == News.news_id)
        .outerjoin(NewsAnalysis, EventMarketAnalysis.news_id == NewsAnalysis.news_id)
    )

    if event_type is not None and event_type.strip():
        stmt = stmt.where(NewsAnalysis.event_type == event_type.strip())

    if hmm_state is not None:
        stmt = stmt.where(EventMarketAnalysis.hmm_state == hmm_state)

    stmt = stmt.order_by(EventMarketAnalysis.event_date.desc()).limit(limit)

    results = db.execute(stmt).all()

    output = []
    for row in results:
        output.append({
            "news_id": row.news_id,
            "event_date": row.event_date,
            "event_day_return": float(row.event_day_return) if row.event_day_return is not None else None,
            "next_day_return": float(row.next_day_return) if row.next_day_return is not None else None,
            "five_day_forward_return": float(row.five_day_forward_return) if row.five_day_forward_return is not None else None,
            "volatility_20": float(row.volatility_20) if row.volatility_20 is not None else None,
            "hmm_state": row.hmm_state,
            "headline": row.headline,
            "event_type": row.event_type if row.event_type else "Other",
            "sentiment_label": row.sentiment_label if row.sentiment_label else "neutral",
            "sentiment_score": float(row.sentiment_score) if row.sentiment_score is not None else None,
        })
    return output



def sanitize_float(val) -> Optional[float]:
    """
    Safely converts pandas/numpy NaN, NaT, None, or float values to float or None for SQL insertion.
    """
    if pd.isna(val) or val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def persist_event_market_analysis(
    analysis_df: pd.DataFrame,
    ticker: str = "^GSPC"
) -> int:
    """
    Idempotently persists observational event-market alignment statistics into PostgreSQL 'event_market_analysis' table.
    
    Foreign Key Constraint:
        Every persisted record must reference a valid parent news_id in the PostgreSQL 'news' table.
        Records without an existing parent in 'news' are skipped and reported explicitly.
        
    Idempotency:
        Repeated calls update existing records matching news_id without creating duplicate rows.
        
    Accepts:
        analysis_df: DataFrame containing event-market fields ('news_id' or raw news columns, 'event_date',
                     'event_day_return', 'next_day_return', '5_day_forward_return', 'Rolling_Volatility_20', 'HMM_State')
        ticker: String ticker symbol (default: '^GSPC')
        
    Returns:
        Number of successfully persisted (inserted + updated) records.
    """
    if analysis_df is None or analysis_df.empty:
        logger.info("Empty or None DataFrame provided for event-market analysis persistence.")
        return 0

    # 1. Extract and map row values
    candidate_records = []
    candidate_news_ids = []

    for _, row in analysis_df.iterrows():
        headline_val = row.get("headline")
        
        # Determine news_id
        if "news_id" in row and pd.notna(row["news_id"]) and str(row["news_id"]).strip():
            news_id = str(row["news_id"]).strip()
        elif pd.notna(headline_val) and str(headline_val).strip():
            row_dict = {
                "headline": str(headline_val).strip(),
                "published_at": row.get("published_at"),
                "publisher": row.get("publisher"),
                "query_ticker": row.get("query_ticker", ticker),
                "summary": row.get("summary"),
                "url": row.get("url"),
            }
            news_id = generate_news_id(row_dict)
        else:
            continue

        # Extract event_date
        event_date_val = row.get("event_date")
        if pd.isna(event_date_val) or event_date_val is None:
            continue

        if hasattr(event_date_val, "date"):
            evt_date = event_date_val.date()
        else:
            evt_date = pd.to_datetime(event_date_val).date()

        # Extract returns & features
        evt_day_ret = sanitize_float(row.get("event_day_return"))
        nxt_day_ret = sanitize_float(row.get("next_day_return"))
        
        five_day_ret = row.get("5_day_forward_return")
        if five_day_ret is None or pd.isna(five_day_ret):
            five_day_ret = row.get("five_day_forward_return")
        five_day_ret = sanitize_float(five_day_ret)

        vol_20 = row.get("Rolling_Volatility_20")
        if vol_20 is None or pd.isna(vol_20):
            vol_20 = row.get("volatility_20")
        vol_20 = sanitize_float(vol_20)

        hmm_st = row.get("HMM_State")
        if hmm_st is None or pd.isna(hmm_st):
            hmm_st = row.get("hmm_state")
        if pd.isna(hmm_st) or hmm_st is None:
            continue
        hmm_st = int(hmm_st)

        candidate_records.append({
            "news_id": news_id,
            "event_date": evt_date,
            "event_day_return": evt_day_ret,
            "next_day_return": nxt_day_ret,
            "five_day_forward_return": five_day_ret,
            "volatility_20": vol_20,
            "hmm_state": hmm_st,
        })
        candidate_news_ids.append(news_id)

    if not candidate_records:
        logger.info("No valid candidate records found in analysis_df.")
        return 0

    # 2. Verify parent news_id existence in PostgreSQL 'news' table
    with get_session() as session:
        valid_parent_news_ids = set(
            session.scalars(
                select(News.news_id).where(News.news_id.in_(candidate_news_ids))
            ).all()
        )

        valid_records = []
        skipped_count = 0
        for record in candidate_records:
            if record["news_id"] in valid_parent_news_ids:
                valid_records.append(record)
            else:
                skipped_count += 1
                logger.warning(
                    f"Skipping event_market_analysis persistence for news_id='{record['news_id']}': "
                    f"Parent article not found in 'news' table."
                )

        if skipped_count > 0:
            logger.warning(f"Skipped {skipped_count} event-market analysis records due to missing parent news_id.")

        if not valid_records:
            logger.warning("No valid records remaining after foreign-key verification.")
            return 0

        # 3. Idempotent Upsert (Query existing records for matching news_id)
        valid_news_ids = [r["news_id"] for r in valid_records]
        existing_ema_records = session.scalars(
            select(EventMarketAnalysis).where(EventMarketAnalysis.news_id.in_(valid_news_ids))
        ).all()

        ema_by_news_id = {rec.news_id: rec for rec in existing_ema_records}

        inserted_count = 0
        updated_count = 0

        for record in valid_records:
            news_id = record["news_id"]
            if news_id in ema_by_news_id:
                # Update existing record
                ema = ema_by_news_id[news_id]
                ema.event_date = record["event_date"]
                ema.event_day_return = record["event_day_return"]
                ema.next_day_return = record["next_day_return"]
                ema.five_day_forward_return = record["five_day_forward_return"]
                ema.volatility_20 = record["volatility_20"]
                ema.hmm_state = record["hmm_state"]
                updated_count += 1
            else:
                # Insert new record
                new_ema = EventMarketAnalysis(
                    news_id=record["news_id"],
                    event_date=record["event_date"],
                    event_day_return=record["event_day_return"],
                    next_day_return=record["next_day_return"],
                    five_day_forward_return=record["five_day_forward_return"],
                    volatility_20=record["volatility_20"],
                    hmm_state=record["hmm_state"],
                )
                session.add(new_ema)
                inserted_count += 1

        session.commit()

    total_persisted = inserted_count + updated_count
    logger.info(
        f"Event-market analysis persistence completed: {total_persisted} persisted "
        f"({inserted_count} inserted, {updated_count} updated, {skipped_count} skipped/rejected)."
    )
    return total_persisted
