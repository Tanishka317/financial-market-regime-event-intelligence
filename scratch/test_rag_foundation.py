"""
Phase 10F-5: RAG Foundation Test Suite
Financial Market Regime & Event Intelligence Engine

Verifies:
1. Document construction & metadata preservation from PostgreSQL news/analysis records.
2. OpenAI API key error handling & missing key protection.
3. Mock vector embedding generation.
4. FAISS index creation, local disk persistence, and loading.
5. Semantic vector retrieval & scoring.
6. Database row count immutability across all 5 tables.
7. Local SentenceTransformers (all-MiniLM-L6-v2) offline index building & semantic retrieval.
"""

import os
import shutil
import unittest
import numpy as np
from sqlalchemy import select, func
from backend.database.connection import get_session
from backend.database.models import MarketData, RegimePrediction, News, NewsAnalysis, EventMarketAnalysis
from backend.rag.document_builder import build_rag_documents
from backend.rag.embedder import LocalEmbedder, OpenAIEmbedder, MockEmbedder
from backend.rag.vector_store import FAISSVectorStore
from backend.rag.retriever import RAGRetriever, build_and_save_rag_index, retrieve_rag_context


class TestRAGFoundation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test_storage_dir = "scratch/test_rag_storage"
        os.makedirs(cls.test_storage_dir, exist_ok=True)

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

    def test_01_document_construction_and_metadata_preservation(self):
        """Test document construction converts PostgreSQL records to text and preserves metadata."""
        documents = build_rag_documents()
        self.assertGreater(len(documents), 0, "RAG document list should not be empty.")
        self.assertEqual(len(documents), self.initial_counts["news"], "Document count should match raw news count.")

        sample_doc = documents[0]
        self.assertIn("doc_id", sample_doc)
        self.assertIn("text", sample_doc)
        self.assertIn("metadata", sample_doc)

        meta = sample_doc["metadata"]
        required_keys = [
            "news_id", "published_at", "event_date", "ticker",
            "publisher", "event_type", "sentiment_label",
            "sentiment_score", "hmm_state", "headline", "url"
        ]
        for key in required_keys:
            self.assertIn(key, meta, f"Metadata must preserve '{key}'.")

        self.assertIn("Headline:", sample_doc["text"])
        self.assertIn("Event Analysis:", sample_doc["text"])

    def test_02_missing_openai_api_key_handling(self):
        """Test OpenAI embedder fails gracefully with clear error when API key is missing."""
        embedder = OpenAIEmbedder(api_key="")
        with self.assertRaises(ValueError) as ctx:
            embedder.embed_text("Test market prompt")

        self.assertIn("OPENAI_API_KEY", str(ctx.exception))
        self.assertIn("missing or empty", str(ctx.exception))

    def test_03_mock_embedder(self):
        """Test mock embedder generates valid 1D and 2D normalized vector arrays."""
        mock = MockEmbedder(dim=384)
        vec = mock.embed_text("Inflation report update")
        self.assertEqual(vec.shape, (384,))
        self.assertAlmostEqual(float(np.linalg.norm(vec)), 1.0, places=4)

        batch = mock.embed_batch(["Headline 1", "Headline 2"])
        self.assertEqual(batch.shape, (2, 384))

    def test_04_faiss_vector_store_index_persistence_and_loading(self):
        """Test FAISS index creation, disk persistence, and loading."""
        documents = build_rag_documents()
        mock = MockEmbedder(dim=384)
        texts = [doc["text"] for doc in documents]
        embeddings = mock.embed_batch(texts)

        store = FAISSVectorStore(embedding_dim=384)
        store.add_documents(embeddings, documents)
        self.assertEqual(store.total_documents, len(documents))

        store.save(self.test_storage_dir)
        self.assertTrue(os.path.exists(os.path.join(self.test_storage_dir, "faiss_index.bin")))
        self.assertTrue(os.path.exists(os.path.join(self.test_storage_dir, "documents_metadata.json")))

        loaded_store = FAISSVectorStore()
        loaded = loaded_store.load(self.test_storage_dir)
        self.assertTrue(loaded)
        self.assertEqual(loaded_store.total_documents, len(documents))

    def test_05_semantic_retrieval(self):
        """Test semantic context retrieval and scoring."""
        retriever = RAGRetriever(storage_dir=self.test_storage_dir, use_mock=True)
        build_res = retriever.build_and_save_index()
        self.assertEqual(build_res["status"], "success")
        self.assertEqual(build_res["document_count"], self.initial_counts["news"])

        query = "What impact did inflation news have on S&P 500 returns?"
        results = retriever.retrieve(query=query, top_k=5)

        self.assertEqual(len(results), 5)
        for res in results:
            self.assertIn("doc_id", res)
            self.assertIn("text", res)
            self.assertIn("metadata", res)
            self.assertIn("score", res)
            self.assertIsInstance(res["score"], float)

    def test_06_local_sentence_transformer_embedding_and_retrieval(self):
        """Test local sentence-transformer (all-MiniLM-L6-v2) index building and semantic retrieval offline."""
        local_embedder = LocalEmbedder()
        retriever = RAGRetriever(storage_dir=self.test_storage_dir, embedder=local_embedder)
        build_res = retriever.build_and_save_index()

        self.assertEqual(build_res["status"], "success")
        self.assertEqual(build_res["embedding_dim"], 384)
        self.assertEqual(build_res["provider"], "LocalEmbedder")

        results = retriever.retrieve(query="What news articles discuss inflation risks?", top_k=3)
        self.assertGreater(len(results), 0)
        self.assertIn("headline", results[0]["metadata"])
        self.assertIn("score", results[0])

    def test_07_database_immutability(self):
        """Verify PostgreSQL row counts remained 100% unchanged."""
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
