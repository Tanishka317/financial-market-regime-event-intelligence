"""
Raw Financial News Repository & PostgreSQL Persistence Module
Financial Market Regime & Event Intelligence Engine
"""

import logging
import hashlib
from datetime import datetime, timezone
import pandas as pd
from typing import Optional, Dict, Any
from sqlalchemy.dialects.postgresql import insert
from backend.database.connection import get_engine
from backend.database.models import News

logger = logging.getLogger("news_repo")


def generate_news_id(row: Dict[str, Any]) -> str:
    """
    Generates a deterministic news_id if one is not already provided.
    Uses URL if available, otherwise falls back to query_ticker, headline, and published_at.
    """
    if "news_id" in row and row["news_id"] and not pd.isna(row["news_id"]):
        clean_id = str(row["news_id"]).strip()
        if clean_id and clean_id.lower() not in ["none", "nan"]:
            return clean_id

    url = str(row.get("url", "") or "").strip()
    headline = str(row.get("headline", "") or "").strip()
    ticker = str(row.get("query_ticker", "") or "").strip()
    pub_at = str(row.get("published_at", "") or "").strip()

    if url and url.lower() not in ["none", "nan", "n/a"]:
        base_key = f"url:{url}"
    else:
        base_key = f"text:{ticker}:{headline}:{pub_at}"

    hash_digest = hashlib.sha256(base_key.encode("utf-8")).hexdigest()[:24]
    return f"news_{hash_digest}"


def persist_news(news_df: pd.DataFrame) -> int:
    """
    Idempotently persists cleaned raw financial news records into PostgreSQL 'news' table.
    Uses PostgreSQL ON CONFLICT DO UPDATE on 'news_id' to prevent duplicate records.
    
    Accepts:
        news_df: DataFrame containing raw news fields ('published_at', 'headline', 'publisher', 'query_ticker', 'summary', 'url')
        
    Returns:
        Number of processed news records.
    """
    if news_df is None or news_df.empty:
        logger.info("Empty or None DataFrame provided for news persistence.")
        return 0

    records = []
    seen_ids = set()
    seen_urls = set()

    for _, row in news_df.iterrows():
        headline_val = row.get("headline")
        if pd.isna(headline_val) or not headline_val or not str(headline_val).strip():
            continue
        headline_str = str(headline_val).strip()

        pub_val = row.get("published_at")
        if pd.isna(pub_val) or pub_val is None:
            continue

        # Convert timestamp safely
        if isinstance(pub_val, str):
            pub_dt = pd.to_datetime(pub_val, utc=True).to_pydatetime()
        elif hasattr(pub_val, "to_pydatetime"):
            pub_dt = pub_val.to_pydatetime()
        elif isinstance(pub_val, datetime):
            pub_dt = pub_val
        else:
            pub_dt = pd.to_datetime(pub_val, utc=True).to_pydatetime()

        if pub_dt.tzinfo is None:
            pub_dt = pub_dt.replace(tzinfo=timezone.utc)

        publisher_val = row.get("publisher")
        if pd.isna(publisher_val) or not publisher_val or str(publisher_val).strip().lower() in ["none", "nan", "n/a", "unknown"]:
            publisher_str = "Unknown"
        else:
            publisher_str = str(publisher_val).strip()

        ticker_val = row.get("query_ticker")
        if pd.isna(ticker_val) or not ticker_val or str(ticker_val).strip().lower() in ["none", "nan"]:
            ticker_str = "UNKNOWN"
        else:
            ticker_str = str(ticker_val).strip()

        summary_val = row.get("summary")
        if pd.isna(summary_val) or summary_val is None or str(summary_val).strip().lower() in ["", "none", "nan", "n/a"]:
            summary_str = None
        else:
            summary_str = str(summary_val).strip()

        url_val = row.get("url")
        if pd.isna(url_val) or url_val is None or str(url_val).strip().lower() in ["", "none", "nan", "n/a"]:
            url_str = None
        else:
            url_str = str(url_val).strip()

        row_dict = {
            "headline": headline_str,
            "published_at": pub_dt,
            "publisher": publisher_str,
            "query_ticker": ticker_str,
            "summary": summary_str,
            "url": url_str,
        }
        if "news_id" in row and pd.notna(row["news_id"]):
            row_dict["news_id"] = row["news_id"]

        news_id = generate_news_id(row_dict)

        # Batch deduplication in Python
        if news_id in seen_ids:
            continue
        if url_str is not None and url_str in seen_urls:
            continue

        seen_ids.add(news_id)
        if url_str is not None:
            seen_urls.add(url_str)

        ingested_at_val = row.get("ingested_at")
        if pd.isna(ingested_at_val) or ingested_at_val is None:
            ingested_dt = datetime.now(timezone.utc)
        elif isinstance(ingested_at_val, str):
            ingested_dt = pd.to_datetime(ingested_at_val, utc=True).to_pydatetime()
        elif hasattr(ingested_at_val, "to_pydatetime"):
            ingested_dt = ingested_at_val.to_pydatetime()
        elif isinstance(ingested_at_val, datetime):
            ingested_dt = ingested_at_val
        else:
            ingested_dt = datetime.now(timezone.utc)

        if ingested_dt.tzinfo is None:
            ingested_dt = ingested_dt.replace(tzinfo=timezone.utc)

        records.append({
            "news_id": news_id,
            "published_at": pub_dt,
            "headline": headline_str,
            "publisher": publisher_str,
            "query_ticker": ticker_str,
            "summary": summary_str,
            "url": url_str,
            "ingested_at": ingested_dt,
        })

    if not records:
        return 0

    engine = get_engine()

    stmt = insert(News).values(records)
    upsert_stmt = stmt.on_conflict_do_update(
        index_elements=["news_id"],
        set_={
            "published_at": stmt.excluded.published_at,
            "headline": stmt.excluded.headline,
            "publisher": stmt.excluded.publisher,
            "query_ticker": stmt.excluded.query_ticker,
            "summary": stmt.excluded.summary,
            "url": stmt.excluded.url,
            "ingested_at": stmt.excluded.ingested_at,
        }
    )

    with engine.begin() as conn:
        conn.execute(upsert_stmt)

    logger.info(f"Successfully persisted {len(records)} raw news records into PostgreSQL 'news' table.")
    return len(records)
