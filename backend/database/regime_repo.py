"""
Regime Prediction Repository & PostgreSQL Persistence Module
Financial Market Regime & Event Intelligence Engine
"""

import logging
import pandas as pd
from typing import Dict, Any
from sqlalchemy.dialects.postgresql import insert
from backend.database.connection import get_engine
from backend.database.models import RegimePrediction

logger = logging.getLogger("regime_repo")


def persist_regime_predictions(
    hmm_result: Dict[str, Any],
    ticker: str = "^GSPC",
    model_version: str = "hmm_v1"
) -> int:
    """
    Idempotently persists Gaussian HMM decoded regime predictions into PostgreSQL 'regime_predictions' table.
    Uses PostgreSQL ON CONFLICT DO UPDATE on (date, ticker, model_version) to prevent duplicate rows.
    
    Accepts:
        hmm_result: dict returned by fit_and_decode_hmm() containing 'market_df' and 'state_stats_df'
        ticker: string ticker identifier (default: '^GSPC')
        model_version: version identifier string (default: 'hmm_v1')
        
    Returns:
        Number of processed records.
    """
    market_df = hmm_result.get("market_df")
    state_stats_df = hmm_result.get("state_stats_df")

    if market_df is None or market_df.empty or "HMM_State" not in market_df.columns:
        logger.warning("Invalid or empty market_df provided for regime persistence.")
        return 0

    # Build state integer to qualitative label mapping from state_stats_df
    label_map = {}
    if state_stats_df is not None and not state_stats_df.empty:
        for _, row in state_stats_df.iterrows():
            label_map[int(row["state"])] = str(row["label"])

    records = []
    for idx, row in market_df.iterrows():
        # Extract Python date object from index
        if hasattr(idx, "date"):
            row_date = idx.date()
        else:
            row_date = pd.to_datetime(idx).date()

        state_val = int(row["HMM_State"])
        regime_lbl = label_map.get(state_val, f"State_{state_val}")

        records.append({
            "date": row_date,
            "ticker": ticker,
            "hmm_state": state_val,
            "regime_label": regime_lbl,
            "model_version": model_version,
        })

    if not records:
        return 0

    engine = get_engine()

    stmt = insert(RegimePrediction).values(records)
    upsert_stmt = stmt.on_conflict_do_update(
        constraint="uq_regime_pred_date_ticker_model",
        set_={
            "hmm_state": stmt.excluded.hmm_state,
            "regime_label": stmt.excluded.regime_label,
        }
    )

    with engine.begin() as conn:
        conn.execute(upsert_stmt)

    logger.info(
        f"Successfully persisted {len(records)} regime prediction records to PostgreSQL "
        f"'regime_predictions' table (version: '{model_version}')."
    )
    return len(records)
