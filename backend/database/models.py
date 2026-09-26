"""
SQLAlchemy 2.x Database Models
Financial Market Regime & Event Intelligence Engine
"""

from datetime import date, datetime
from typing import Optional, List
from sqlalchemy import (
    String, Text, Date, DateTime, Numeric, Integer, BigInteger,
    ForeignKey, UniqueConstraint, Index, func
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy 2.x declarative models."""
    pass


class MarketData(Base):
    """
    Stores historical daily market observations and quantitative risk/correlation features.
    """
    __tablename__ = "market_data"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    date: Mapped[date] = mapped_column(Date, nullable=False)
    ticker: Mapped[str] = mapped_column(String(20), nullable=False)
    close: Mapped[float] = mapped_column(Numeric(14, 4), nullable=False)
    daily_return: Mapped[Optional[float]] = mapped_column(Numeric(12, 6), nullable=True)
    volatility_20: Mapped[Optional[float]] = mapped_column(Numeric(12, 6), nullable=True)
    momentum_20: Mapped[Optional[float]] = mapped_column(Numeric(12, 6), nullable=True)
    drawdown: Mapped[Optional[float]] = mapped_column(Numeric(12, 6), nullable=True)
    sp500_tlt_corr_20: Mapped[Optional[float]] = mapped_column(Numeric(10, 6), nullable=True)
    sp500_gld_corr_20: Mapped[Optional[float]] = mapped_column(Numeric(10, 6), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        UniqueConstraint("date", "ticker", name="uq_market_data_date_ticker"),
        Index("ix_market_data_date_ticker", "date", "ticker"),
    )

    def __repr__(self) -> str:
        return f"<MarketData(ticker='{self.ticker}', date='{self.date}', close={self.close})>"


class RegimePrediction(Base):
    """
    Stores decoded Gaussian HMM market regime assignments and state telemetry.
    """
    __tablename__ = "regime_predictions"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    date: Mapped[date] = mapped_column(Date, nullable=False)
    ticker: Mapped[str] = mapped_column(String(20), nullable=False)
    hmm_state: Mapped[int] = mapped_column(Integer, nullable=False)
    regime_label: Mapped[str] = mapped_column(String(50), nullable=False)
    model_version: Mapped[str] = mapped_column(String(50), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        UniqueConstraint("date", "ticker", "model_version", name="uq_regime_pred_date_ticker_model"),
        Index("ix_regime_predictions_date_ticker", "date", "ticker"),
    )

    def __repr__(self) -> str:
        return f"<RegimePrediction(ticker='{self.ticker}', date='{self.date}', state={self.hmm_state}, label='{self.regime_label}')>"


class News(Base):
    """
    Stores raw ingested financial news headlines and metadata.
    """
    __tablename__ = "news"

    news_id: Mapped[str] = mapped_column(String(100), primary_key=True)
    published_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    headline: Mapped[str] = mapped_column(Text, nullable=False)
    publisher: Mapped[str] = mapped_column(String(100), nullable=False)
    query_ticker: Mapped[str] = mapped_column(String(20), nullable=False)
    summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    url: Mapped[Optional[str]] = mapped_column(Text, nullable=True, unique=True)
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    analysis: Mapped[Optional["NewsAnalysis"]] = relationship(
        "NewsAnalysis", back_populates="news", uselist=False, cascade="all, delete-orphan"
    )
    event_market_analyses: Mapped[List["EventMarketAnalysis"]] = relationship(
        "EventMarketAnalysis", back_populates="news", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_news_published_ticker", "published_at", "query_ticker"),
    )

    def __repr__(self) -> str:
        return f"<News(news_id='{self.news_id}', query_ticker='{self.query_ticker}', published_at='{self.published_at}')>"


class NewsAnalysis(Base):
    """
    Stores FinBERT sentiment inference and rule-based event classification results for news headlines.
    """
    __tablename__ = "news_analysis"

    news_id: Mapped[str] = mapped_column(
        String(100), ForeignKey("news.news_id", ondelete="CASCADE"), primary_key=True
    )
    sentiment_label: Mapped[str] = mapped_column(String(20), nullable=False)
    sentiment_score: Mapped[float] = mapped_column(Numeric(8, 6), nullable=False)
    event_type: Mapped[str] = mapped_column(String(50), nullable=False)
    event_trigger: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    analyzed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    model_version: Mapped[str] = mapped_column(String(50), nullable=False)

    # Relationship
    news: Mapped["News"] = relationship("News", back_populates="analysis")

    __table_args__ = (
        Index("ix_news_analysis_event_sentiment", "event_type", "sentiment_label"),
    )

    def __repr__(self) -> str:
        return f"<NewsAnalysis(news_id='{self.news_id}', sentiment='{self.sentiment_label}', event_type='{self.event_type}')>"


class EventMarketAnalysis(Base):
    """
    Stores calendar-aligned event-market return statistics and active HMM state decodes.
    """
    __tablename__ = "event_market_analysis"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    news_id: Mapped[str] = mapped_column(
        String(100), ForeignKey("news.news_id", ondelete="CASCADE"), nullable=False
    )
    event_date: Mapped[date] = mapped_column(Date, nullable=False)
    event_day_return: Mapped[Optional[float]] = mapped_column(Numeric(12, 6), nullable=True)
    next_day_return: Mapped[Optional[float]] = mapped_column(Numeric(12, 6), nullable=True)
    five_day_forward_return: Mapped[Optional[float]] = mapped_column(Numeric(12, 6), nullable=True)
    volatility_20: Mapped[Optional[float]] = mapped_column(Numeric(12, 6), nullable=True)
    hmm_state: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationship
    news: Mapped["News"] = relationship("News", back_populates="event_market_analyses")

    __table_args__ = (
        Index("ix_event_market_analysis_date_hmm_news", "event_date", "hmm_state", "news_id"),
    )

    def __repr__(self) -> str:
        return f"<EventMarketAnalysis(id={self.id}, news_id='{self.news_id}', event_date='{self.event_date}', hmm_state={self.hmm_state})>"
