"""
RAG Embedding Layer Module
Financial Market Regime & Event Intelligence Engine

Supports configurable embedding providers:
- LocalEmbedder (DEFAULT): Local SentenceTransformers (all-MiniLM-L6-v2, 384 dim, 0 API cost)
- OpenAIEmbedder (OPTIONAL): OpenAI embeddings API (text-embedding-3-small, 1536 dim)
- MockEmbedder (TESTING): Deterministic offline mock embeddings for fast tests
"""

import os
import logging
import numpy as np
from typing import List, Optional, Union
from dotenv import load_dotenv

# Ensure environment variables are loaded from .env
load_dotenv()

logger = logging.getLogger("rag.embedder")


class LocalEmbedder:
    """
    Local embedding layer using SentenceTransformers (default: sentence-transformers/all-MiniLM-L6-v2).
    Requires zero external API calls or API keys. Lazy-loads model weights on first embedding request.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info(f"Loading local SentenceTransformer model: '{self.model_name}'...")
                self._model = SentenceTransformer(self.model_name)
            except Exception as e:
                logger.error(f"Failed to load local SentenceTransformer model '{self.model_name}': {e}")
                raise RuntimeError(f"Local embedding model initialization failed: {e}")
        return self._model

    def embed_text(self, text: str) -> np.ndarray:
        """
        Generates 1D float32 L2-normalized embedding for a single string prompt.
        """
        if not text or not text.strip():
            raise ValueError("Input text for embedding cannot be empty.")
        model = self._get_model()
        vec = model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        return vec.astype(np.float32)

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Generates 2D float32 L2-normalized embeddings for a batch of strings.
        """
        if not texts:
            return np.empty((0, 384), dtype=np.float32)
        model = self._get_model()
        vecs = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return vecs.astype(np.float32)


class OpenAIEmbedder:
    """
    Optional embedding layer wrapper around OpenAI's embedding API.
    Does NOT automatically invoke API calls on module import or instantiation.
    Enforces strict API key validation without exposing keys.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None
    ):
        self.model = model
        self._api_key = api_key if api_key is not None else os.getenv("OPENAI_API_KEY")

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
    without requiring API credentials or network requests.
    """

    def __init__(self, dim: int = 384):
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


def get_embedder(provider: Optional[str] = None):
    """
    Factory function returning the configured embedding provider instance.
    Provider options:
      - 'local' / None (default): Local SentenceTransformers (sentence-transformers/all-MiniLM-L6-v2)
      - 'openai': Optional OpenAI Embeddings API (text-embedding-3-small)
      - 'mock': Fast deterministic offline mock embedder
    """
    selected_provider = (provider or os.getenv("EMBEDDING_PROVIDER", "local")).strip().lower()

    if selected_provider in ("local", "sentence_transformers", "sentence-transformers"):
        return LocalEmbedder()
    elif selected_provider == "openai":
        return OpenAIEmbedder()
    elif selected_provider == "mock":
        return MockEmbedder()
    else:
        logger.warning(f"Unknown EMBEDDING_PROVIDER '{selected_provider}'. Defaulting to 'local'.")
        return LocalEmbedder()
