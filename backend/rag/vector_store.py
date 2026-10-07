"""
FAISS Vector Store Module
Financial Market Regime & Event Intelligence Engine

Manages FAISS vector indexing, similarity search, and local persistence of embeddings & metadata.
"""

import os
import json
import logging
import faiss
import numpy as np
from typing import List, Dict, Any, Optional

logger = logging.getLogger("rag.vector_store")


class FAISSVectorStore:
    """
    Local FAISS vector index wrapper supporting cosine similarity retrieval
    and serialization to disk.
    """

    def __init__(self, embedding_dim: int = 1536):
        self.embedding_dim = embedding_dim
        # IndexFlatIP measures Inner Product (Cosine Similarity when vectors are L2-normalized)
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.documents: List[Dict[str, Any]] = []

    @property
    def total_documents(self) -> int:
        return self.index.ntotal

    def add_documents(self, embeddings: np.ndarray, documents: List[Dict[str, Any]]):
        """
        Adds vector embeddings and corresponding document metadata to the FAISS index.

        Accepts:
            embeddings: 2D numpy array of shape (N, dim)
            documents: List of dicts [{"doc_id": ..., "text": ..., "metadata": ...}]
        """
        if embeddings is None or len(embeddings) == 0:
            logger.warning("Empty embeddings array provided to add_documents.")
            return

        if len(embeddings) != len(documents):
            raise ValueError(
                f"Mismatch between number of embeddings ({len(embeddings)}) "
                f"and documents ({len(documents)})."
            )

        embeddings_arr = np.ascontiguousarray(embeddings.astype(np.float32))
        faiss.normalize_L2(embeddings_arr)

        self.index.add(embeddings_arr)
        self.documents.extend(documents)
        logger.info(f"Added {len(documents)} documents to FAISS vector index (total: {self.index.ntotal}).")

    def save(self, storage_dir: str = "backend/rag/storage"):
        """
        Persists FAISS index binary and documents metadata JSON to local disk.
        """
        os.makedirs(storage_dir, exist_ok=True)
        index_path = os.path.join(storage_dir, "faiss_index.bin")
        meta_path = os.path.join(storage_dir, "documents_metadata.json")

        faiss.write_index(self.index, index_path)

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self.documents, f, indent=2, ensure_ascii=False)

        logger.info(f"Successfully saved FAISS index ({self.index.ntotal} items) to '{storage_dir}'.")

    def load(self, storage_dir: str = "backend/rag/storage") -> bool:
        """
        Loads FAISS index binary and documents metadata from disk.
        Returns True if successful, False if directory or files are missing.
        """
        index_path = os.path.join(storage_dir, "faiss_index.bin")
        meta_path = os.path.join(storage_dir, "documents_metadata.json")

        if not os.path.exists(index_path) or not os.path.exists(meta_path):
            logger.info(f"FAISS storage directory or files missing at '{storage_dir}'.")
            return False

        try:
            self.index = faiss.read_index(index_path)
            with open(meta_path, "r", encoding="utf-8") as f:
                self.documents = json.load(f)

            self.embedding_dim = self.index.d
            logger.info(f"Loaded FAISS vector index ({self.index.ntotal} docs, dim={self.embedding_dim}) from '{storage_dir}'.")
            return True
        except Exception as e:
            logger.error(f"Failed to load FAISS index from '{storage_dir}': {e}")
            return False

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Performs vector similarity search against the indexed documents.

        Accepts:
            query_embedding: 1D or 2D numpy array of dimension embedding_dim.
            top_k: Number of top relevant documents to retrieve.

        Returns:
            List of dicts containing document text, metadata, and similarity score.
        """
        if self.index.ntotal == 0:
            logger.warning("FAISS vector index is empty. Returning 0 results.")
            return []

        query_arr = np.ascontiguousarray(query_embedding.astype(np.float32))
        if query_arr.ndim == 1:
            query_arr = query_arr.reshape(1, -1)

        faiss.normalize_L2(query_arr)

        actual_k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_arr, actual_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.documents):
                continue
            doc = self.documents[idx]
            results.append({
                "doc_id": doc["doc_id"],
                "text": doc["text"],
                "metadata": doc["metadata"],
                "score": round(float(score), 4),
            })

        return results
