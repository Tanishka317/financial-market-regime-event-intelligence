"""
RAG Foundation Module
Financial Market Regime & Event Intelligence Engine
"""

from backend.rag.document_builder import build_rag_documents
from backend.rag.embedder import OpenAIEmbedder, MockEmbedder
from backend.rag.vector_store import FAISSVectorStore
from backend.rag.retriever import RAGRetriever, build_and_save_rag_index, retrieve_rag_context

__all__ = [
    "build_rag_documents",
    "OpenAIEmbedder",
    "MockEmbedder",
    "FAISSVectorStore",
    "RAGRetriever",
    "build_and_save_rag_index",
    "retrieve_rag_context",
]
