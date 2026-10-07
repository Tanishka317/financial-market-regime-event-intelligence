"""
Phase 10F-6: LLM Integration & Hybrid Analyst Test Suite
Financial Market Regime & Event Intelligence Engine

Verifies:
1. Deterministic analyst functionality remains fully intact.
2. RAG context retrieval operates as expected.
3. OpenAILLMClient missing-key and error handling behavior.
4. LLM response synthesis, prompt building, and source grounding preservation using OpenAI mocks.
5. API behavior when LLM is disabled or unavailable.
6. End-to-end FastAPI endpoint POST /api/analyst/query with mocks (0 paid API calls).
7. Database row count immutability across all 5 tables.
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

    def test_01_deterministic_analyst_remains_intact(self):
        """Verify original deterministic analyst returns valid SQL answers."""
        with get_session() as session:
            res = execute_analyst_query(db=session, question="What is the latest market regime?", ticker="^GSPC")
            self.assertEqual(res["intent"], "latest_regime")
            self.assertIn("Low-Vol Bull", res["answer"])
            self.assertIn("hmm_state", res["supporting_data"])

    def test_02_rag_retrieval_remains_functional(self):
        """Verify RAG retrieval returns top context using mock embeddings."""
        results = retrieve_rag_context(query="inflation risks", top_k=3, storage_dir=self.test_storage_dir, use_mock=True)
        self.assertGreater(len(results), 0)
        self.assertIn("doc_id", results[0])
        self.assertIn("headline", results[0]["metadata"])

    def test_03_llm_client_missing_key_behavior(self):
        """Verify LLM client handles missing API key gracefully without crashing."""
        llm_client = OpenAILLMClient(api_key="")
        self.assertFalse(llm_client.is_available())

        with self.assertRaises(ValueError) as ctx:
            llm_client.generate_synthesis("Test prompt")
        self.assertIn("OPENAI_API_KEY", str(ctx.exception))

    def test_04_prompt_building_and_grounding_preservation(self):
        """Verify prompt construction combines question, SQL stats, and RAG metadata."""
        prompt = build_synthesis_prompt(
            question="How did Inflation affect returns?",
            deterministic_answer="Analyzed 2 Inflation events.",
            supporting_data={"event_type": "Inflation", "avg_next_day_return_pct": 0.51},
            rag_sources=[{
                "text": "Headline: Inflation rises",
                "score": 0.89,
                "metadata": {"headline": "Inflation rises", "published_at": "2026-09-27", "event_type": "Inflation"}
            }]
        )
        self.assertIn("USER QUESTION: How did Inflation affect returns?", prompt)
        self.assertIn("Analyzed 2 Inflation events.", prompt)
        self.assertIn("Headline: Inflation rises", prompt)
        self.assertIn("Relevance Score: 0.89", prompt)

    @patch("openai.OpenAI")
    def test_05_hybrid_service_with_mocked_openai_synthesis(self, mock_openai_cls):
        """Verify hybrid analyst query integrates LLM response when OpenAI API returns valid completion."""
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content="Mocked LLM Synthesis: Inflation events resulted in an average next-day return of +0.51%."))
        ]
        mock_openai_instance = MagicMock()
        mock_openai_instance.chat.completions.create.return_value = mock_response
        mock_openai_cls.return_value = mock_openai_instance

        mock_client = OpenAILLMClient(api_key="sk-mock-valid-key-for-testing")

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

        self.assertIn("mode", res)
        self.assertIn("llm", res["mode"])
        self.assertIn("Mocked LLM Synthesis", res["answer"])
        self.assertGreater(len(res["sources"]), 0)
        self.assertIn("headline", res["sources"][0])

    def test_06_hybrid_service_fallback_when_llm_disabled(self):
        """Verify graceful fallback to deterministic analysis when LLM is disabled or key is missing."""
        mock_client = OpenAILLMClient(api_key="")

        with get_session() as session:
            res = execute_hybrid_analyst_query(
                db=session,
                question="What is the latest market data?",
                ticker="^GSPC",
                use_llm=True,
                use_rag=False,
                use_mock_rag=True,
                rag_storage_dir=self.test_storage_dir,
                llm_client=mock_client
            )

        self.assertEqual(res["mode"], "deterministic")
        self.assertIn("closing price", res["answer"])

    @patch("openai.OpenAI")
    def test_07_api_endpoint_post_analyst_query(self, mock_openai_cls):
        """Verify POST /api/analyst/query endpoint with mocked OpenAI client."""
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content="API Synthesis Answer: S&P 500 regime is currently Low-Vol Bull."))
        ]
        mock_openai_instance = MagicMock()
        mock_openai_instance.chat.completions.create.return_value = mock_response
        mock_openai_cls.return_value = mock_openai_instance

        # Test request with use_llm=False
        resp_det = client.post("/api/analyst/query", json={"question": "What is the latest regime?", "use_llm": False})
        self.assertEqual(resp_det.status_code, 200)
        data_det = resp_det.json()
        self.assertEqual(data_det["mode"], "deterministic")

        # Test request with mocked LLM enabled
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-mock-key"}):
            with patch("backend.llm.service.retrieve_rag_context") as mock_retrieve:
                mock_retrieve.return_value = []
                resp_llm = client.post("/api/analyst/query", json={"question": "What is the latest regime?", "use_llm": True})
                self.assertEqual(resp_llm.status_code, 200)
                data_llm = resp_llm.json()
                self.assertIn("answer", data_llm)
                self.assertIn("intent", data_llm)

    def test_08_database_immutability(self):
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
