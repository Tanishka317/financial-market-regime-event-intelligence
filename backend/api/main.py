"""
FastAPI Main Application Module
Financial Market Regime & Event Intelligence Engine
"""

from datetime import date, datetime
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, Depends, HTTPException, Query, status
from pydantic import BaseModel
from sqlalchemy.orm import Session
from backend.database.connection import get_db
from backend.database.market_data_repo import get_latest_market_data, get_market_history
from backend.database.regime_repo import get_latest_regime_prediction, get_regime_history
from backend.database.news_repo import get_news_records
from backend.database.news_analysis_repo import get_news_analysis_by_id
from backend.database.event_market_repo import get_event_market_analysis_records
from backend.analyst.service import execute_analyst_query


app = FastAPI(
    title="Financial Market Regime & Event Intelligence Engine API",
    description="REST API backend exposing market data, regime predictions, news, and event intelligence.",
    version="1.0.0"
)


class MarketDataResponse(BaseModel):
    id: int
    date: date
    ticker: str
    close: float
    daily_return: Optional[float] = None
    volatility_20: Optional[float] = None
    momentum_20: Optional[float] = None
    drawdown: Optional[float] = None
    sp500_tlt_corr_20: Optional[float] = None
    sp500_gld_corr_20: Optional[float] = None
    created_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class RegimePredictionResponse(BaseModel):
    id: int
    date: date
    ticker: str
    hmm_state: int
    regime_label: str
    model_version: str
    created_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class NewsResponse(BaseModel):
    news_id: str
    published_at: datetime
    headline: str
    publisher: str
    query_ticker: str
    summary: Optional[str] = None
    url: Optional[str] = None
    ingested_at: datetime

    class Config:
        from_attributes = True


class NewsAnalysisResponse(BaseModel):
    news_id: str
    sentiment_label: str
    sentiment_score: float
    event_type: str
    event_trigger: Optional[str] = None
    analyzed_at: datetime
    model_version: str

    class Config:
        from_attributes = True


class EventMarketAnalysisResponse(BaseModel):
    news_id: str
    event_date: date
    event_day_return: Optional[float] = None
    next_day_return: Optional[float] = None
    five_day_forward_return: Optional[float] = None
    volatility_20: Optional[float] = None
    hmm_state: int
    event_type: str
    sentiment_label: str
    sentiment_score: Optional[float] = None
    headline: str

    class Config:
        from_attributes = True


@app.get("/api/health", status_code=status.HTTP_200_OK)
def get_health():
    """
    Health check endpoint returning API service operational status.
    """
    return {
        "status": "ok",
        "service": "financial-intelligence-api"
    }


@app.get("/api/market/latest", response_model=MarketDataResponse, status_code=status.HTTP_200_OK)
def get_latest_market_record(db: Session = Depends(get_db)):
    """
    Returns the most recent available market_data record for ^GSPC, ordered by date descending.
    Raises 404 HTTP exception if no record is found in database.
    """
    record = get_latest_market_data(db, ticker="^GSPC")
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No market data record found for ^GSPC"
        )
    return record


@app.get("/api/market/history", response_model=List[MarketDataResponse], status_code=status.HTTP_200_OK)
def get_market_history_records(
    ticker: str = Query("^GSPC", description="Ticker symbol to query"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum records to return"),
    db: Session = Depends(get_db)
):
    """
    Returns historical market_data records for specified ticker, ordered by date descending.
    Raises 404 HTTP exception if no records match ticker.
    """
    records = get_market_history(db, ticker=ticker, limit=limit)
    if not records:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No market data records found for ticker '{ticker}'"
        )
    return records


@app.get("/api/regimes/latest", response_model=RegimePredictionResponse, status_code=status.HTTP_200_OK)
def get_latest_regime(
    ticker: str = Query("^GSPC", description="Ticker symbol to query"),
    db: Session = Depends(get_db)
):
    """
    Returns the most recent Gaussian HMM regime prediction for specified ticker.
    Raises 404 HTTP exception if no regime prediction exists.
    """
    record = get_latest_regime_prediction(db, ticker=ticker)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No regime prediction found for ticker '{ticker}'"
        )
    return record


@app.get("/api/regimes", response_model=List[RegimePredictionResponse], status_code=status.HTTP_200_OK)
def get_regime_history_records(
    ticker: str = Query("^GSPC", description="Ticker symbol to query"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum records to return"),
    db: Session = Depends(get_db)
):
    """
    Returns historical Gaussian HMM regime predictions for specified ticker, ordered by date descending.
    Raises 404 HTTP exception if no records match ticker.
    """
    records = get_regime_history(db, ticker=ticker, limit=limit)
    if not records:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No regime predictions found for ticker '{ticker}'"
        )
    return records


@app.get("/api/news", response_model=List[NewsResponse], status_code=status.HTTP_200_OK)
def get_news(
    ticker: Optional[str] = Query(None, description="Filter news by query ticker symbol"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum news records to return"),
    db: Session = Depends(get_db)
):
    """
    Returns raw financial news records ordered by published_at descending.
    Optionally filters by query_ticker. Raises HTTP 404 if no matching records exist.
    """
    records = get_news_records(db, ticker=ticker, limit=limit)
    if not records:
        ticker_msg = f" for ticker '{ticker}'" if ticker else ""
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No news records found{ticker_msg}"
        )
    return records


@app.get("/api/news/{news_id}/analysis", response_model=NewsAnalysisResponse, status_code=status.HTTP_200_OK)
def get_news_analysis(
    news_id: str,
    db: Session = Depends(get_db)
):
    """
    Returns persisted sentiment and event classification analysis for supplied news_id.
    Raises HTTP 404 if analysis record does not exist.
    """
    analysis = get_news_analysis_by_id(db, news_id=news_id)
    if analysis is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No news analysis found for news_id '{news_id}'"
        )
    return analysis


@app.get("/api/event-market-analysis", response_model=List[EventMarketAnalysisResponse], status_code=status.HTTP_200_OK)
def get_event_market_analysis(
    event_type: Optional[str] = Query(None, description="Filter by event classification type"),
    hmm_state: Optional[int] = Query(None, ge=0, le=3, description="Filter by Gaussian HMM regime state integer"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum records to return"),
    db: Session = Depends(get_db)
):
    """
    Returns event-market analysis records joined with news and news_analysis context, ordered by event_date descending.
    Raises HTTP 404 if no matching records exist.
    """
    records = get_event_market_analysis_records(db, event_type=event_type, hmm_state=hmm_state, limit=limit)
    if not records:
        filters = []
        if event_type:
            filters.append(f"event_type='{event_type}'")
        if hmm_state is not None:
            filters.append(f"hmm_state={hmm_state}")
        filter_str = f" with filters ({', '.join(filters)})" if filters else ""
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No event-market analysis records found{filter_str}"
        )
    return records


class AnalystQueryRequest(BaseModel):
    question: str
    ticker: Optional[str] = "^GSPC"


class AnalystQueryResponse(BaseModel):
    question: str
    intent: str
    answer: str
    supporting_data: Dict[str, Any]


@app.post("/api/analyst/query", response_model=AnalystQueryResponse, status_code=status.HTTP_200_OK)
def query_financial_analyst(
    request: AnalystQueryRequest,
    db: Session = Depends(get_db)
):
    """
    Data-grounded deterministic Financial Intelligence Analyst endpoint.
    Answers natural language queries about market regimes, latest data, news, event performance, and regimes during events.
    """
    return execute_analyst_query(db=db, question=request.question, ticker=request.ticker)

