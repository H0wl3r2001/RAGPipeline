"""Query embedding + Qdrant similarity search."""

import logging

import httpx
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from app.config import settings

logger = logging.getLogger(__name__)


def embed_query(question: str) -> list[float]:
    """Embed a question via Ollama /api/embed."""
    resp = httpx.post(
        f"{settings.ollama_url}/api/embed",
        json={"model": settings.embed_model, "input": question},
        timeout=120.0,
    )
    resp.raise_for_status()
    return resp.json()["embeddings"][0]


def retrieve(question: str) -> list[dict]:
    """Embed the question and search Qdrant for top-k chunks.

    Returns a list of dicts: {"file", "chunk_index", "score", "text"}.
    Raises ValueError if the collection does not exist.
    """
    query_vector = embed_query(question)
    client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)

    collections = [c.name for c in client.get_collections().collections]
    if settings.qdrant_collection not in collections:
        raise ValueError(
            f"Collection '{settings.qdrant_collection}' not found — "
            f"run /ingest first to populate the vector store."
        )

    results = client.query_points(
        collection_name=settings.qdrant_collection,
        query=query_vector,
        limit=settings.top_k,
        with_payload=True,
    ).points

    chunks = []
    for r in results:
        chunks.append(
            {
                "file": r.payload["source_file"],
                "chunk_index": r.payload["chunk_index"],
                "score": r.score,
                "text": r.payload["text"],
            }
        )
    logger.info("Retrieved %d chunk(s) for query", len(chunks))
    return chunks
