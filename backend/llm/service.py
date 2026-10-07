"""
LLM & RAG Hybrid Financial Intelligence Service
Financial Market Regime & Event Intelligence Engine

Orchestrates deterministic SQL analysis, semantic FAISS RAG context retrieval,
and OpenAI Chat Completion synthesis into a unified, data-grounded response.
"""

import logging
from typing import Dict, Any, Optional, List
from sqlalchemy.orm import Session
from backend.analyst.service import execute_analyst_query
from backend.rag.retriever import retrieve_rag_context
from backend.llm.client import OpenAILLMClient
from backend.llm.prompts import build_synthesis_prompt

logger = logging.getLogger("llm.service")


def execute_hybrid_analyst_query(
    db: Session,
    question: str,
    ticker: Optional[str] = "^GSPC",
    use_llm: bool = True,
    use_rag: bool = True,
    use_mock_rag: bool = False,
    rag_storage_dir: str = "backend/rag/storage",
    llm_client: Optional[OpenAILLMClient] = None
) -> Dict[str, Any]:
    """
    Orchestrates deterministic PostgreSQL analysis, FAISS RAG context retrieval,
    and optional OpenAI LLM synthesis.

    Accepts:
        db: SQLAlchemy Session
        question: User query prompt
        ticker: Ticker symbol (default: '^GSPC')
        use_llm: Whether to attempt LLM synthesis (default: True)
        use_rag: Whether to retrieve RAG financial news context (default: True)
        use_mock_rag: Whether to use mock embeddings for RAG retrieval testing
        llm_client: Optional pre-configured OpenAILLMClient instance

    Returns:
        Structured dictionary matching AnalystQueryResponse contract.
    """
    # 1. Deterministic SQL Analysis
    det_res = execute_analyst_query(db=db, question=question, ticker=ticker)

    # 2. Semantic RAG Context Retrieval (if enabled)
    rag_sources = []
    formatted_sources = []
    if use_rag:
        try:
            rag_sources = retrieve_rag_context(query=question, top_k=3, storage_dir=rag_storage_dir, use_mock=use_mock_rag)
            for r in rag_sources:
                meta = r.get("metadata", {})
                formatted_sources.append({
                    "news_id": meta.get("news_id"),
                    "headline": meta.get("headline"),
                    "event_type": meta.get("event_type"),
                    "sentiment_label": meta.get("sentiment_label"),
                    "score": r.get("score"),
                    "url": meta.get("url"),
                })
        except Exception as e:
            logger.warning(f"RAG context retrieval failed: {e}. Continuing without RAG sources.")

    # 3. OpenAI LLM Synthesis (if enabled & available)
    if use_llm:
        client = llm_client or OpenAILLMClient()
        if client.is_available():
            try:
                prompt = build_synthesis_prompt(
                    question=question,
                    deterministic_answer=det_res["answer"],
                    supporting_data=det_res.get("supporting_data", {}),
                    rag_sources=rag_sources
                )
                llm_answer = client.generate_synthesis(prompt)

                mode = "llm_hybrid" if (rag_sources and det_res["intent"] != "unsupported") else (
                    "llm_rag" if rag_sources else "llm_analytical"
                )

                return {
                    "question": question,
                    "intent": det_res["intent"],
                    "answer": llm_answer,
                    "mode": mode,
                    "sources": formatted_sources,
                    "supporting_data": det_res.get("supporting_data", {})
                }
            except Exception as e:
                logger.warning(f"LLM synthesis failed: {e}. Falling back to deterministic analysis.")

    # 4. Fallback: Deterministic SQL Analyst Output
    return {
        "question": question,
        "intent": det_res["intent"],
        "answer": det_res["answer"],
        "mode": "deterministic",
        "sources": formatted_sources,
        "supporting_data": det_res.get("supporting_data", {})
    }
