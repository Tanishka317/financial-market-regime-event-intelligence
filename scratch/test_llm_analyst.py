"""
Phase 10F-6: Local Hybrid Analyst & RAG Synthesis Test Suite
Financial Market Regime & Event Intelligence Engine

Verifies:
1. Structured-only queries (PostgreSQL data only).
2. Local RAG-enhanced queries (SQL stats + local FAISS news retrieval).
3. Insufficient context query handling (out-of-scope prompts).
4. OpenAI unavailable fallback behavior.
5. Deterministic analyst & API backward compatibility.
6. PostgreSQL row count immutability across all 5 tables.
"""

import os
import shutil
import unittest
from unittest.mock import patch, MagicMock
from sqlalchemy import select, func
from fastapi.testclient import TestClient

from backend.database.connection import get_session
from backend.database.models import MarketData, RegimePrediction, News, NewsAnalysis, EventMarketAnalysis
from backend.analyst.service import execute_analyst_query
from backend.rag.retriever import retrieve_rag_context, build_and_save_rag_index
from backend.llm.client import OpenAILLMClient
from backend.llm.prompts import build_synthesis_prompt, SYSTEM_PROMPT
from backend.llm.service import execute_hybrid_analyst_query
from backend.api.main import app

client = TestClient(app)


class TestLLMAnalystIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test_storage_dir = "scratch/test_llm_storage"
        os.makedirs(cls.test_storage_dir, exist_ok=True)

        # Pre-build test FAISS index using mock embeddings
        build_and_save_rag_index(storage_dir=cls.test_storage_dir, use_mock=True)

        with get_session() as session:
            cls.initial_counts = {
                "market_data": session.scalar(select(func.count(MarketData.id))),
                "regime_predictions": session.scalar(select(func.count(RegimePrediction.id))),
                "news": session.scalar(select(func.count(News.news_id))),
                "news_analysis": session.scalar(select(func.count(NewsAnalysis.news_id))),
                "event_market_analysis": session.scalar(select(func.count(EventMarketAnalysis.id))),
            }

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.test_storage_dir):
            shutil.rmtree(cls.test_storage_dir, ignore_errors=True)

    def test_01_structured_only_query(self):
        """Test query with use_rag=False and use_llm=False returns structured SQL analytics only."""
        with get_session() as session:
            res = execute_hybrid_analyst_query(
                db=session,
                question="What is the latest market regime?",
                ticker="^GSPC",
                use_llm=False,
                use_rag=False
            )
            self.assertEqual(res["intent"], "latest_regime")
            self.assertEqual(res["mode"], "structured_analyst")
            self.assertEqual(len(res["sources"]), 0)
            self.assertIn("Low-Vol Bull", res["answer"])

    def test_02_rag_enhanced_local_hybrid_query(self):
        """Test query with use_rag=True and use_llm=False returns combined SQL analytics and local RAG context."""
        with get_session() as session:
            res = execute_hybrid_analyst_query(
                db=session,
                question="How did Inflation perform?",
                ticker="^GSPC",
                use_llm=False,
                use_rag=True,
                use_mock_rag=True,
                rag_storage_dir=self.test_storage_dir
            )
            self.assertEqual(res["intent"], "event_performance")
            self.assertEqual(res["mode"], "local_hybrid")
            self.assertGreater(len(res["sources"]), 0)
            self.assertIn("Structured Analytics", res["answer"])
            self.assertIn("Retrieved Local News Context", res["answer"])

    def test_03_insufficient_context_query(self):
        """Test query with out-of-scope question and no RAG sources explicitly states context is insufficient."""
        with get_session() as session:
            with patch("backend.llm.service.retrieve_rag_context") as mock_retrieve:
                mock_retrieve.return_value = []
                res = execute_hybrid_analyst_query(
                    db=session,
                    question="What is the population of Mars?",
                    ticker="^GSPC",
                    use_llm=False,
                    use_rag=True
                )
                self.assertEqual(res["intent"], "unsupported")
                self.assertEqual(res["mode"], "insufficient_context")
                self.assertIn("available project data in PostgreSQL", res["answer"])

    def test_04_openai_unavailable_fallback(self):
        """Test fallback to local hybrid answer when OpenAI LLM is enabled but unavailable."""
        mock_client = OpenAILLMClient(api_key="") # Missing API key
        with get_session() as session:
            res = execute_hybrid_analyst_query(
                db=session,
                question="How did Inflation perform?",
                ticker="^GSPC",
                use_llm=True,
                use_rag=True,
                use_mock_rag=True,
                rag_storage_dir=self.test_storage_dir,
                llm_client=mock_client
            )
            self.assertEqual(res["mode"], "local_hybrid")
            self.assertIn("Structured Analytics", res["answer"])

    @patch("openai.OpenAI")
    def test_05_optional_openai_llm_synthesis(self, mock_openai_cls):
        """Test optional OpenAI synthesis when valid API key is present."""
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content="Mocked LLM Synthesis Answer."))
        ]
        mock_openai_instance = MagicMock()
        mock_openai_instance.chat.completions.create.return_value = mock_response
        mock_openai_cls.return_value = mock_openai_instance

        mock_client = OpenAILLMClient(api_key="sk-mock-key")

        with get_session() as session:
            res = execute_hybrid_analyst_query(
                db=session,
                question="How did Inflation perform?",
                ticker="^GSPC",
                use_llm=True,
                use_rag=True,
                use_mock_rag=True,
                rag_storage_dir=self.test_storage_dir,
                llm_client=mock_client
            )
            self.assertEqual(res["mode"], "llm_hybrid")
            self.assertIn("Mocked LLM Synthesis Answer", res["answer"])

    def test_06_fastapi_endpoint_local_default(self):
        """Test POST /api/analyst/query returns 200 with local hybrid response by default."""
        resp = client.post("/api/analyst/query", json={"question": "What is the latest market regime?"})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("answer", data)
        self.assertIn("intent", data)
        self.assertIn("mode", data)

    def test_07_database_immutability(self):
        """Verify PostgreSQL row counts remain 100% unchanged across all tables."""
        with get_session() as session:
            final_counts = {
                "market_data": session.scalar(select(func.count(MarketData.id))),
                "regime_predictions": session.scalar(select(func.count(RegimePrediction.id))),
                "news": session.scalar(select(func.count(News.news_id))),
                "news_analysis": session.scalar(select(func.count(NewsAnalysis.news_id))),
                "event_market_analysis": session.scalar(select(func.count(EventMarketAnalysis.id))),
            }
        for table, count in self.initial_counts.items():
            self.assertEqual(
                final_counts[table], count,
                f"Row count for table '{table}' changed from {count} to {final_counts[table]}!"
            )


if __name__ == "__main__":
    unittest.main()
