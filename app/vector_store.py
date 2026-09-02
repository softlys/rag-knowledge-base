import uuid

import chromadb

from app.config import settings

_client = None
_collection = None


def get_collection():
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=settings.chroma_dir)
        # Без указания embedding_function Chroma использует свою встроенную
        # локальную модель эмбеддингов (ONNX MiniLM) — работает без API-ключей.
        _collection = _client.get_or_create_collection(name="documents")
    return _collection


def add_chunks(chunks: list[str], source: str) -> int:
    if not chunks:
        return 0
    collection = get_collection()
    ids = [f"{source}-{uuid.uuid4().hex[:8]}-{i}" for i in range(len(chunks))]
    metadatas = [{"source": source, "chunk_index": i} for i in range(len(chunks))]
    collection.add(documents=chunks, ids=ids, metadatas=metadatas)
    return len(chunks)


def query(question: str, top_k: int) -> list[dict]:
    collection = get_collection()
    if collection.count() == 0:
        return []
    results = collection.query(query_texts=[question], n_results=min(top_k, collection.count()))

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    return [
        {"text": doc, "source": meta.get("source", "unknown"), "distance": dist}
        for doc, meta, dist in zip(docs, metas, distances)
    ]


def list_sources() -> list[str]:
    collection = get_collection()
    if collection.count() == 0:
        return []
    all_metas = collection.get(include=["metadatas"])["metadatas"]
    return sorted({m["source"] for m in all_metas})
