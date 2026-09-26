"""
Market Data Repository & PostgreSQL Persistence Module
Financial Market Regime & Event Intelligence Engine
"""

import logging
import pandas as pd
import numpy as np
from typing import Optional
from sqlalchemy.dialects.postgresql import insert
from backend.database.connection import get_engine
from backend.database.models import MarketData

logger = logging.getLogger("market_data_repo")


def sanitize_val(val) -> Optional[float]:
    """
    Converts pandas/numpy NaN, NaT, None, or float64 values to float or None for SQL insertion.
    """
    if pd.isna(val) or val is None:
        return None
    return float(val)


def persist_market_data(clean_df: pd.DataFrame, ticker: str = "^GSPC") -> int:
    """
    Idempotently persists the cleaned market DataFrame into PostgreSQL 'market_data' table.
    Uses PostgreSQL ON CONFLICT DO UPDATE on (date, ticker) to update existing rows and prevent duplicates.
    
    Returns the number of records processed.
    """
    if clean_df.empty:
        logger.warning("Empty DataFrame provided for market data persistence.")
        return 0

    records = []
    for idx, row in clean_df.iterrows():
        # Extract Python date object from index
        if hasattr(idx, "date"):
            row_date = idx.date()
        else:
            row_date = pd.to_datetime(idx).date()

        records.append({
            "date": row_date,
            "ticker": ticker,
            "close": float(row["Close"]),
            "daily_return": sanitize_val(row.get("Daily_Return")),
            "volatility_20": sanitize_val(row.get("Rolling_Volatility_20")),
            "momentum_20": sanitize_val(row.get("Momentum_20")),
            "drawdown": sanitize_val(row.get("Drawdown")),
            "sp500_tlt_corr_20": sanitize_val(row.get("SP500_TLT_Corr_20")),
            "sp500_gld_corr_20": sanitize_val(row.get("SP500_GLD_Corr_20")),
        })

    if not records:
        return 0

    engine = get_engine()

    stmt = insert(MarketData).values(records)
    upsert_stmt = stmt.on_conflict_do_update(
        constraint="uq_market_data_date_ticker",
        set_={
            "close": stmt.excluded.close,
            "daily_return": stmt.excluded.daily_return,
            "volatility_20": stmt.excluded.volatility_20,
            "momentum_20": stmt.excluded.momentum_20,
            "drawdown": stmt.excluded.drawdown,
            "sp500_tlt_corr_20": stmt.excluded.sp500_tlt_corr_20,
            "sp500_gld_corr_20": stmt.excluded.sp500_gld_corr_20,
        }
    )

    with engine.begin() as conn:
        conn.execute(upsert_stmt)

    logger.info(f"Successfully persisted {len(records)} market data rows to PostgreSQL 'market_data' table.")
    return len(records)
