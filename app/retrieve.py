"""Query embedding + hybrid retrieval (Qdrant dense + BM25 sparse, fused via RRF)."""

import logging

import httpx
from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi

from app.config import settings
from app.generate import ModelNotPulledError

logger = logging.getLogger(__name__)


def embed_query(question: str) -> list[float]:
    """Embed a question via Ollama /api/embed."""
    resp = httpx.post(
        f"{settings.ollama_url}/api/embed",
        json={"model": settings.embed_model, "input": question},
        timeout=120.0,
    )
    if resp.status_code == 404:
        raise ModelNotPulledError(settings.embed_model)
    resp.raise_for_status()
    return resp.json()["embeddings"][0]


def _tokenize(text: str) -> list[str]:
    """Lightweight tokenization for BM25: lowercase + whitespace split.

    Per the brief: "no torch/transformers." This is intentionally simple —
    aggressive stemming/stopwords would help marginally and are easy to add
    later if measured lift justifies it.
    """
    return text.lower().split()


def retrieve(question: str) -> list[dict]:
    """Embed the question; score the corpus with Qdrant cosine + BM25; fuse via RRF.

    Returns a list of dicts: {"file", "chunk_index", "score", "text"}.
    Raises ValueError if the collection does not exist or is empty.
    """
    query_vector = embed_query(question)
    query_tokens = _tokenize(question)
    client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)

    collections = [c.name for c in client.get_collections().collections]
    if settings.qdrant_collection not in collections:
        raise ValueError(
            f"Collection '{settings.qdrant_collection}' not found — "
            f"run /ingest first to populate the vector store."
        )

    # Scroll the entire corpus once per query. For this project's scope (≤ a
    # few thousand chunks) the cost is negligible vs. ~30-60s of CPU LLM
    # generation. If the corpus grows past ~10k chunks, this should switch to
    # dense-KNN top-N + BM25 rerank over that narrow candidate set.
    records, _ = client.scroll(
        collection_name=settings.qdrant_collection,
        limit=10000,
        with_payload=True,
        with_vectors=False,
    )
    if not records:
        raise ValueError(
            f"Collection '{settings.qdrant_collection}' is empty — "
            f"re-run /ingest."
        )

    corpus_ids = [r.id for r in records]
    corpus_meta = [
        {
            "file": r.payload["source_file"],
            "chunk_index": r.payload["chunk_index"],
            "text": r.payload["text"],
        }
        for r in records
    ]
    corpus_tokens = [_tokenize(meta["text"]) for meta in corpus_meta]

    # RRF candidate pool: take top N from each ranking, where N is generous
    # enough that top_k results survive fusion even if one ranking is poor.
    rrf_pool = max(settings.top_k * 8, 32)

    # BM25 sparse scoring over the full corpus (cheap at this size).
    bm25 = BM25Okapi(corpus_tokens)
    bm25_scores = bm25.get_scores(query_tokens).tolist()
    bm25_ranked = sorted(
        [(rid, score) for rid, score in zip(corpus_ids, bm25_scores) if score > 0],
        key=lambda p: -p[1],
    )[:rrf_pool]

    # Qdrant dense cosine scoring (HNSW index).
    dense_hits = client.query_points(
        collection_name=settings.qdrant_collection,
        query=query_vector,
        limit=rrf_pool,
        with_payload=False,
    ).points
    dense_ranked = [(h.id, h.score) for h in dense_hits]

    # Reciprocal Rank Fusion: final score is the sum of 1/(k + rank) across
    # both rankings. Rank-based — no need to normalize scores from different
    # distributions (cosine 0-1 vs BM25 unbounded).
    rrf_scores: dict[int, float] = {}
    for rank, (rid, _score) in enumerate(dense_ranked, start=1):
        rrf_scores[rid] = rrf_scores.get(rid, 0.0) + 1.0 / (settings.rrf_k + rank)
    for rank, (rid, _score) in enumerate(bm25_ranked, start=1):
        rrf_scores[rid] = rrf_scores.get(rid, 0.0) + 1.0 / (settings.rrf_k + rank)

    fused = sorted(rrf_scores.items(), key=lambda p: -p[1])[: settings.top_k]

    id_to_meta = dict(zip(corpus_ids, corpus_meta))
    chunks = []
    for cid, score in fused:
        meta = id_to_meta[cid]
        chunks.append(
            {
                "file": meta["file"],
                "chunk_index": meta["chunk_index"],
                "score": float(score),
                "text": meta["text"],
            }
        )
    logger.info("Retrieved %d chunk(s) for query (hybrid RRF k=%d)", len(chunks), settings.rrf_k)
    return chunks
