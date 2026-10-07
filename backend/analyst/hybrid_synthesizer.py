"""
Local Hybrid Synthesis Layer
Financial Market Regime & Event Intelligence Engine

Combines structured PostgreSQL statistical analytics with local FAISS semantic RAG retrieval
into grounded, transparent financial answers without requiring external APIs or credits.
"""

import logging
from typing import Dict, Any, List

logger = logging.getLogger("analyst.hybrid_synthesizer")


def synthesize_local_hybrid_answer(
    question: str,
    det_res: Dict[str, Any],
    rag_sources: List[Dict[str, Any]],
    formatted_sources: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Constructs a grounded, multi-source financial answer combining:
    1. Structured PostgreSQL SQL calculations (historical prices, HMM states, risk metrics).
    2. Local FAISS semantic context (news headlines, FinBERT sentiment, event classification).

    Accepts:
        question: User query prompt
        det_res: Output dictionary from execute_analyst_query()
        rag_sources: Raw retrieved RAG document list
        formatted_sources: Formatted source metadata list

    Returns:
        Structured response dictionary matching AnalystQueryResponse schema.
    """
    intent = det_res.get("intent", "unsupported")
    sql_answer = det_res.get("answer", "")
    data = det_res.get("supporting_data", {})

    # Case 1: Insufficient Data / Unsupported Question with no RAG context
    if intent == "unsupported" and not rag_sources:
        answer = (
            "Based on the available project data in PostgreSQL, I could not find sufficient "
            "structured metrics or news context to answer this query. "
            "Supported queries cover S&P 500 market data, decoded HMM regimes, risk/correlation metrics, "
            "financial news, and event return statistics."
        )
        return {
            "question": question,
            "intent": intent,
            "answer": answer,
            "mode": "insufficient_context",
            "sources": [],
            "supporting_data": data
        }

    # Case 2: Structured SQL Analytics + RAG Context Combined
    if intent != "unsupported" and rag_sources:
        rag_context_lines = []
        for src in formatted_sources:
            h = src.get("headline", "")
            pub = src.get("publisher", "")
            et = src.get("event_type", "Other")
            sent = src.get("sentiment_label", "neutral")
            score = src.get("score")
            score_str = f"relevance: {score:.2f}" if score is not None else ""

            line = f"• '{h}' ({pub}) — Event: {et}, Sentiment: {sent}"
            if score_str:
                line += f" ({score_str})"
            rag_context_lines.append(line)

        combined_answer = (
            f"**Structured Analytics (PostgreSQL)**:\n{sql_answer}\n\n"
            f"**Retrieved Local News Context (FAISS RAG)**:\n" + "\n".join(rag_context_lines)
        )

        return {
            "question": question,
            "intent": intent,
            "answer": combined_answer,
            "mode": "local_hybrid",
            "sources": formatted_sources,
            "supporting_data": data
        }

    # Case 3: RAG News Context Only (for news/event topic questions not in 10 SQL intents)
    if intent == "unsupported" and rag_sources:
        top_h = formatted_sources[0].get("headline", "") if formatted_sources else ""
        top_pub = formatted_sources[0].get("publisher", "") if formatted_sources else ""
        top_et = formatted_sources[0].get("event_type", "Other") if formatted_sources else "Other"

        rag_context_lines = [f"• '{src.get('headline')}' ({src.get('publisher')}) — Event: {src.get('event_type')}, Sentiment: {src.get('sentiment_label')}" for src in formatted_sources]

        answer = (
            f"Retrieved {len(formatted_sources)} relevant financial news records from local FAISS storage. "
            f"Top headline: '{top_h}' ({top_pub}) classified under '{top_et}' events.\n\n"
            f"**Retrieved News Context**:\n" + "\n".join(rag_context_lines)
        )

        return {
            "question": question,
            "intent": intent,
            "answer": answer,
            "mode": "local_rag",
            "sources": formatted_sources,
            "supporting_data": {"count": len(formatted_sources), "articles": formatted_sources}
        }

    # Case 4: Structured SQL Analytics Only (no RAG sources retrieved)
    return {
        "question": question,
        "intent": intent,
        "answer": sql_answer,
        "mode": "structured_analyst",
        "sources": [],
        "supporting_data": data
    }
