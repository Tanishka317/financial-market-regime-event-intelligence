"""
Financial Intelligence Analyst Service
Financial Market Regime & Event Intelligence Engine
"""

import logging
from typing import Dict, Any, Optional
from sqlalchemy.orm import Session

from backend.database.market_data_repo import get_latest_market_data, get_market_history
from backend.database.regime_repo import get_latest_regime_prediction, get_regime_history
from backend.database.news_repo import get_news_records, get_news_with_analysis
from backend.database.event_market_repo import get_event_market_analysis_records
from backend.analyst.intents import (
    classify_intent,
    INTENT_LATEST_REGIME,
    INTENT_LATEST_MARKET_DATA,
    INTENT_MARKET_METRICS,
    INTENT_REGIME_DISTRIBUTION,
    INTENT_REGIME_MARKET_BEHAVIOR,
    INTENT_RECENT_NEWS,
    INTENT_NEWS_BY_SENTIMENT,
    INTENT_EVENT_PERFORMANCE,
    INTENT_EVENT_RANKING,
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
    1. Classifies user question into analytical intent and extracts entities.
    2. Queries database using existing repositories.
    3. Synthesizes a structured answer and supporting context.
    """
    intent_info = classify_intent(question)
    intent = intent_info["intent"]
    extracted_event = intent_info.get("event_type")
    extracted_sentiment = intent_info.get("sentiment_label")
    extracted_metric = intent_info.get("metric")
    extracted_ranking = intent_info.get("ranking")
    query_ticker = ticker or intent_info.get("ticker") or "^GSPC"

    # 1. Latest Regime Intent
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
        return {"question": question, "intent": intent, "answer": answer, "supporting_data": supporting_data}

    # 2. Latest Market Data Intent
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
        return {"question": question, "intent": intent, "answer": answer, "supporting_data": supporting_data}

    # 3. Market Metrics Intent (Volatility, Momentum, Drawdown, Correlation)
    elif intent == INTENT_MARKET_METRICS:
        history = get_market_history(db, ticker=query_ticker, limit=100)
        if not history:
            return {
                "question": question,
                "intent": intent,
                "answer": f"No historical market data available for ticker '{query_ticker}'.",
                "supporting_data": {"ticker": query_ticker, "record_count": 0}
            }

        latest = history[0]
        metric_name = extracted_metric or "volatility"

        if metric_name == "volatility":
            vols = [float(r.volatility_20) for r in history if r.volatility_20 is not None]
            avg_vol = (sum(vols) / len(vols) * 100) if vols else 0.0
            latest_vol = (float(latest.volatility_20) * 100) if latest.volatility_20 is not None else 0.0
            answer = (
                f"For {query_ticker} over the last {len(history)} sessions, current 20-day volatility is {latest_vol:.2f}% "
                f"(historical average: {avg_vol:.2f}%, range: {min(vols)*100:.2f}% - {max(vols)*100:.2f}%)."
            )
            data = {"metric": "volatility_20", "latest_pct": round(latest_vol, 4), "avg_pct": round(avg_vol, 4), "sample_size": len(history)}

        elif metric_name == "momentum":
            moms = [float(r.momentum_20) for r in history if r.momentum_20 is not None]
            latest_mom = (float(latest.momentum_20) * 100) if latest.momentum_20 is not None else 0.0
            avg_mom = (sum(moms) / len(moms) * 100) if moms else 0.0
            answer = (
                f"For {query_ticker}, current 20-day momentum is {latest_mom:+.2f}% "
                f"(historical average: {avg_mom:+.2f}% over {len(history)} sessions)."
            )
            data = {"metric": "momentum_20", "latest_pct": round(latest_mom, 4), "avg_pct": round(avg_mom, 4), "sample_size": len(history)}

        elif metric_name == "drawdown":
            dds = [float(r.drawdown) for r in history if r.drawdown is not None]
            latest_dd = (float(latest.drawdown) * 100) if latest.drawdown is not None else 0.0
            max_dd = (min(dds) * 100) if dds else 0.0
            answer = (
                f"For {query_ticker}, current drawdown is {latest_dd:.2f}% "
                f"(maximum drawdown over recent {len(history)} sessions: {max_dd:.2f}%)."
            )
            data = {"metric": "drawdown", "latest_pct": round(latest_dd, 4), "max_drawdown_pct": round(max_dd, 4), "sample_size": len(history)}

        elif metric_name == "correlation":
            tlt_corrs = [float(r.sp500_tlt_corr_20) for r in history if r.sp500_tlt_corr_20 is not None]
            gld_corrs = [float(r.sp500_gld_corr_20) for r in history if r.sp500_gld_corr_20 is not None]
            avg_tlt = (sum(tlt_corrs) / len(tlt_corrs)) if tlt_corrs else 0.0
            avg_gld = (sum(gld_corrs) / len(gld_corrs)) if gld_corrs else 0.0
            latest_tlt = float(latest.sp500_tlt_corr_20) if latest.sp500_tlt_corr_20 is not None else 0.0
            latest_gld = float(latest.sp500_gld_corr_20) if latest.sp500_gld_corr_20 is not None else 0.0
            answer = (
                f"For S&P 500 ({query_ticker}), current 20-day correlation with TLT (Treasuries) is {latest_tlt:+.2f} "
                f"(avg: {avg_tlt:+.2f}) and with GLD (Gold) is {latest_gld:+.2f} (avg: {avg_gld:+.2f})."
            )
            data = {"metric": "cross_asset_correlation", "latest_sp500_tlt": round(latest_tlt, 4), "latest_sp500_gld": round(latest_gld, 4), "avg_sp500_tlt": round(avg_tlt, 4), "avg_sp500_gld": round(avg_gld, 4)}

        else:
            # Default returns summary
            rets = [float(r.daily_return) for r in history if r.daily_return is not None]
            avg_ret = (sum(rets) / len(rets) * 100) if rets else 0.0
            pos_pct = (len([r for r in rets if r > 0]) / len(rets) * 100) if rets else 0.0
            answer = (
                f"Over the past {len(history)} sessions for {query_ticker}, average daily return was {avg_ret:+.2f}% "
                f"with a win rate of {pos_pct:.1f}% positive sessions."
            )
            data = {"metric": "returns", "avg_daily_return_pct": round(avg_ret, 4), "win_rate_pct": round(pos_pct, 2), "sample_size": len(history)}

        return {"question": question, "intent": intent, "answer": answer, "supporting_data": data}

    # 4. Regime Distribution Intent
    elif intent == INTENT_REGIME_DISTRIBUTION:
        regimes = get_regime_history(db, ticker=query_ticker, limit=1000)
        if not regimes:
            return {
                "question": question,
                "intent": intent,
                "answer": f"No historical regime records found for ticker '{query_ticker}'.",
                "supporting_data": {"ticker": query_ticker, "record_count": 0}
            }

        counts = {}
        for r in regimes:
            lbl = r.regime_label
            counts[lbl] = counts.get(lbl, 0) + 1

        total = len(regimes)
        breakdown = [f"'{lbl}': {cnt} days ({cnt/total*100:.1f}%)" for lbl, cnt in counts.items()]
        answer = (
            f"Over {total} historical sessions for {query_ticker}, the market regime distribution was: "
            f"{', '.join(breakdown)}."
        )
        supporting_data = {
            "ticker": query_ticker,
            "total_sessions": total,
            "distribution": {lbl: {"days": cnt, "pct": round(cnt / total * 100, 2)} for lbl, cnt in counts.items()}
        }
        return {"question": question, "intent": intent, "answer": answer, "supporting_data": supporting_data}

    # 5. Regime Market Behavior Intent
    elif intent == INTENT_REGIME_MARKET_BEHAVIOR:
        regimes = get_regime_history(db, ticker=query_ticker, limit=500)
        market_map = {r.date: r for r in get_market_history(db, ticker=query_ticker, limit=500)}

        if not regimes or not market_map:
            return {
                "question": question,
                "intent": intent,
                "answer": f"Insufficient regime and market data for ticker '{query_ticker}'.",
                "supporting_data": {"ticker": query_ticker, "record_count": 0}
            }

        stats_by_regime = {}
        for reg in regimes:
            lbl = reg.regime_label
            mkt = market_map.get(reg.date)
            if not mkt:
                continue
            if lbl not in stats_by_regime:
                stats_by_regime[lbl] = {"returns": [], "vols": []}
            if mkt.daily_return is not None:
                stats_by_regime[lbl]["returns"].append(float(mkt.daily_return))
            if mkt.volatility_20 is not None:
                stats_by_regime[lbl]["vols"].append(float(mkt.volatility_20))

        summaries = []
        parsed_data = {}
        for lbl, d in stats_by_regime.items():
            avg_ret = (sum(d["returns"]) / len(d["returns"]) * 100) if d["returns"] else 0.0
            avg_vol = (sum(d["vols"]) / len(d["vols"]) * 100) if d["vols"] else 0.0
            summaries.append(f"'{lbl}': Avg Daily Return {avg_ret:+.2f}%, Avg Volatility {avg_vol:.2f}%")
            parsed_data[lbl] = {"avg_daily_return_pct": round(avg_ret, 4), "avg_volatility_pct": round(avg_vol, 4), "sample_days": len(d["returns"])}

        answer = (
            f"Market behavior by regime for {query_ticker}: " + "; ".join(summaries) + "."
        )
        return {"question": question, "intent": intent, "answer": answer, "supporting_data": {"ticker": query_ticker, "regime_behavior": parsed_data}}

    # 6. News by Sentiment Intent
    elif intent == INTENT_NEWS_BY_SENTIMENT:
        lbl = extracted_sentiment or "negative"
        news_list = get_news_with_analysis(db, ticker=None if query_ticker == "^GSPC" else query_ticker, sentiment_label=lbl, limit=5)
        if not news_list:
            return {
                "question": question,
                "intent": intent,
                "answer": f"No recent '{lbl}' sentiment news articles were found in the database.",
                "supporting_data": {"sentiment_label": lbl, "count": 0, "articles": []}
            }

        top_h = news_list[0]["headline"]
        top_pub = news_list[0]["publisher"]
        answer = (
            f"Retrieved {len(news_list)} recent '{lbl}' sentiment news items. "
            f"Top headline: '{top_h}' published by {top_pub}."
        )
        return {
            "question": question,
            "intent": intent,
            "answer": answer,
            "supporting_data": {"sentiment_label": lbl, "count": len(news_list), "articles": news_list}
        }

    # 7. Recent News Intent
    elif intent == INTENT_RECENT_NEWS:
        news_list = get_news_with_analysis(db, ticker=None if query_ticker == "^GSPC" else query_ticker, event_type=extracted_event, limit=5)
        if not news_list:
            return {
                "question": question,
                "intent": intent,
                "answer": "No financial news records were found in the database.",
                "supporting_data": {"count": 0, "articles": []}
            }

        top_h = news_list[0]["headline"]
        top_pub = news_list[0]["publisher"]
        answer = (
            f"Retrieved {len(news_list)} recent financial news items. "
            f"Latest headline: '{top_h}' published by {top_pub}."
        )
        return {
            "question": question,
            "intent": intent,
            "answer": answer,
            "supporting_data": {"count": len(news_list), "articles": news_list}
        }

    # 8. Event Performance Intent
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
        return {"question": question, "intent": intent, "answer": answer, "supporting_data": supporting_data}

    # 9. Event Ranking / Comparison Intent
    elif intent == INTENT_EVENT_RANKING:
        records = get_event_market_analysis_records(db, limit=200)
        if not records:
            return {
                "question": question,
                "intent": intent,
                "answer": "No event-market analysis records available for ranking.",
                "supporting_data": {"record_count": 0}
            }

        event_stats = {}
        for r in records:
            et = r.get("event_type", "Other")
            if et not in event_stats:
                event_stats[et] = {"next_day": [], "five_day": [], "count": 0}
            event_stats[et]["count"] += 1
            if r.get("next_day_return") is not None:
                event_stats[et]["next_day"].append(r["next_day_return"])
            if r.get("five_day_forward_return") is not None:
                event_stats[et]["five_day"].append(r["five_day_forward_return"])

        ranked_list = []
        for et, d in event_stats.items():
            avg_nxt = (sum(d["next_day"]) / len(d["next_day"]) * 100) if d["next_day"] else 0.0
            avg_5d = (sum(d["five_day"]) / len(d["five_day"]) * 100) if d["five_day"] else 0.0
            ranked_list.append({"event_type": et, "count": d["count"], "avg_next_day_return_pct": round(avg_nxt, 4), "avg_5day_forward_return_pct": round(avg_5d, 4)})

        direction = extracted_ranking or "best"
        reverse_sort = (direction in ["best", "highest", "ranking", "top"])
        ranked_list.sort(key=lambda x: x["avg_next_day_return_pct"], reverse=reverse_sort)

        top_event = ranked_list[0] if ranked_list else None
        if top_event:
            answer = (
                f"Ranked {len(ranked_list)} event categories by next-day return ({direction}). "
                f"Top category: '{top_event['event_type']}' with {top_event['avg_next_day_return_pct']:+.2f}% average next-day return ({top_event['count']} events)."
            )
        else:
            answer = "Completed event ranking."

        return {"question": question, "intent": intent, "answer": answer, "supporting_data": {"ranking_direction": direction, "ranked_events": ranked_list}}

    # 10. Event Regime Intent
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
        return {"question": question, "intent": intent, "answer": answer, "supporting_data": supporting_data}

    # 11. Unsupported / Out of Scope Intent
    else:
        answer = (
            "I could not match your question to a supported financial intelligence intent based on our dataset. "
            "Supported questions include: 1) Latest regime & historical distribution, 2) Market price, volatility, momentum, drawdown & correlation metrics, "
            "3) Recent news & news by sentiment/event, 4) Event performance & event ranking, and 5) Market regimes during events."
        )
        return {
            "question": question,
            "intent": INTENT_UNSUPPORTED,
            "answer": answer,
            "supporting_data": {
                "supported_intents": [
                    INTENT_LATEST_REGIME,
                    INTENT_LATEST_MARKET_DATA,
                    INTENT_MARKET_METRICS,
                    INTENT_REGIME_DISTRIBUTION,
                    INTENT_REGIME_MARKET_BEHAVIOR,
                    INTENT_RECENT_NEWS,
                    INTENT_NEWS_BY_SENTIMENT,
                    INTENT_EVENT_PERFORMANCE,
                    INTENT_EVENT_RANKING,
                    INTENT_EVENT_REGIME
                ]
            }
        }
