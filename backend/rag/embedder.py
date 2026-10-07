"""
RAG Embedding Layer Module
Financial Market Regime & Event Intelligence Engine

Generates vector embeddings using OpenAI API (text-embedding-3-small)
and provides a deterministic MockEmbedder for testing/offline operations.
"""

import os
import logging
import numpy as np
from typing import List, Optional, Union

logger = logging.getLogger("rag.embedder")


class OpenAIEmbedder:
    """
    Embedding layer wrapper around OpenAI's embedding API.
    Does NOT automatically invoke API calls on module import or instantiation.
    Enforces strict API key validation without exposing keys.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None
    ):
        self.model = model
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")

    def _get_client(self):
        if not self._api_key or self._api_key.strip() in ("", "your_openai_api_key_here"):
            raise ValueError(
                "OPENAI_API_KEY environment variable is missing or empty. "
                "Please configure OPENAI_API_KEY in your environment or .env file."
            )
        try:
            from openai import OpenAI
            return OpenAI(api_key=self._api_key.strip())
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise RuntimeError(f"OpenAI client initialization failed: {e}")

    def embed_text(self, text: str) -> np.ndarray:
        """
        Generates 1D float32 normalized embedding for a single string prompt.
        """
        if not text or not text.strip():
            raise ValueError("Input text for embedding cannot be empty.")

        client = self._get_client()
        response = client.embeddings.create(
            input=text,
            model=self.model
        )
        embedding_vec = np.array(response.data[0].embedding, dtype=np.float32)
        norm = np.linalg.norm(embedding_vec)
        if norm > 0:
            embedding_vec = embedding_vec / norm
        return embedding_vec

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Generates 2D float32 normalized embeddings for a batch of strings.
        """
        if not texts:
            return np.empty((0, 1536), dtype=np.float32)

        client = self._get_client()
        response = client.embeddings.create(
            input=texts,
            model=self.model
        )
        embeddings_list = [item.embedding for item in response.data]
        arr = np.array(embeddings_list, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms


class MockEmbedder:
    """
    Deterministic mock embedder for offline unit testing and development
    without requiring OpenAI API credentials or network requests.
    """

    def __init__(self, dim: int = 1536):
        self.dim = dim

    def embed_text(self, text: str) -> np.ndarray:
        import hashlib
        seed = int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16) % (2**32)
        rng = np.random.RandomState(seed)
        vec = rng.randn(self.dim).astype(np.float32)
        norm = np.linalg.norm(vec)
        return vec / (norm if norm > 0 else 1.0)

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        return np.vstack([self.embed_text(t) for t in texts])
