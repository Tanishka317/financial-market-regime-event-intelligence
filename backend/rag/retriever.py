"""
RAG Semantic Retriever & Index Management Module
Financial Market Regime & Event Intelligence Engine

Orchestrates document construction, vector embedding (Local SentenceTransformers by default,
or optional OpenAI / Mock providers), FAISS indexing, persistence, and semantic context retrieval.
"""

import os
import logging
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from backend.rag.document_builder import build_rag_documents
from backend.rag.embedder import get_embedder, MockEmbedder, LocalEmbedder
from backend.rag.vector_store import FAISSVectorStore

logger = logging.getLogger("rag.retriever")


class RAGRetriever:
    """
    RAG service wrapper managing vector index loading, embedding generation,
    and semantic similarity search.
    """

    def __init__(
        self,
        storage_dir: str = "backend/rag/storage",
        embedder: Optional[Any] = None,
        provider: Optional[str] = None,
        use_mock: bool = False
    ):
        self.storage_dir = storage_dir
        if embedder is not None:
            self.embedder = embedder
        elif use_mock:
            self.embedder = MockEmbedder()
        else:
            self.embedder = get_embedder(provider)

        self.vector_store = FAISSVectorStore()
        self._is_loaded = False

    def build_and_save_index(self, db: Optional[Session] = None) -> Dict[str, Any]:
        """
        Fetches PostgreSQL records, constructs RAG documents, generates embeddings,
        builds FAISS vector index, and saves to disk.

        Returns:
            Dictionary with indexing telemetry (document count, storage location).
        """
        documents = build_rag_documents(db=db)
        if not documents:
            logger.warning("No PostgreSQL documents found to index.")
            return {"document_count": 0, "status": "empty", "storage_dir": self.storage_dir}

        texts = [doc["text"] for doc in documents]
        logger.info(f"Generating embeddings for {len(texts)} RAG documents using {self.embedder.__class__.__name__}...")
        embeddings = self.embedder.embed_batch(texts)

        dim = embeddings.shape[1] if len(embeddings) > 0 else 384
        self.vector_store = FAISSVectorStore(embedding_dim=dim)
        self.vector_store.add_documents(embeddings, documents)
        self.vector_store.save(self.storage_dir)
        self._is_loaded = True

        return {
            "document_count": len(documents),
            "embedding_dim": dim,
            "storage_dir": self.storage_dir,
            "provider": self.embedder.__class__.__name__,
            "status": "success",
        }

    def ensure_loaded(self) -> bool:
        """
        Ensures vector store is loaded from storage directory.
        """
        if self._is_loaded and self.vector_store.total_documents > 0:
            return True
        success = self.vector_store.load(self.storage_dir)
        self._is_loaded = success
        return success

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Executes semantic retrieval for a natural-language query prompt.

        Accepts:
            query: Query string
            top_k: Top k documents to return

        Returns:
            List of dicts: [{"doc_id": ..., "text": ..., "metadata": ..., "score": ...}]
        """
        if not query or not query.strip():
            return []

        if not self.ensure_loaded():
            logger.warning(
                f"FAISS vector store could not be loaded from '{self.storage_dir}'. "
                "Please run build_and_save_index() first."
            )
            return []

        query_embedding = self.embedder.embed_text(query.strip())
        results = self.vector_store.search(query_embedding, top_k=top_k)
        return results


def build_and_save_rag_index(
    storage_dir: str = "backend/rag/storage",
    db: Optional[Session] = None,
    provider: Optional[str] = None,
    use_mock: bool = False
) -> Dict[str, Any]:
    """
    Standalone function to explicitly trigger RAG index build and local persistence.
    Defaults to local sentence-transformer provider.
    """
    retriever = RAGRetriever(storage_dir=storage_dir, provider=provider, use_mock=use_mock)
    return retriever.build_and_save_index(db=db)


def retrieve_rag_context(
    query: str,
    top_k: int = 5,
    storage_dir: str = "backend/rag/storage",
    provider: Optional[str] = None,
    use_mock: bool = False
) -> List[Dict[str, Any]]:
    """
    Standalone function to retrieve semantic RAG context for a query prompt.
    Defaults to local sentence-transformer provider.
    """
    retriever = RAGRetriever(storage_dir=storage_dir, provider=provider, use_mock=use_mock)
    return retriever.retrieve(query=query, top_k=top_k)
