from backend.rag.retriever import build_and_save_rag_index

if __name__ == "__main__":
    res = build_and_save_rag_index(storage_dir="backend/rag/storage", use_mock=True)
    print("Default FAISS index created successfully:", res)
