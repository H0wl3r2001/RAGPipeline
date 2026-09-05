"""Document ingestion: load, chunk, embed, upsert to Qdrant."""

import logging
from pathlib import Path

import httpx
from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from app.config import settings

logger = logging.getLogger(__name__)

# Embedding dimension for nomic-embed-text (768-dim). If EMBED_MODEL is swapped
# for a different embedding model, update this to match and recreate the collection.
EMBED_DIMENSION = 768

# Baseline chunking strategy: fixed-size word windows with overlap.
# Defaults ~500 words/chunk with 50-word overlap approximate the "500 tokens /
# 50 overlap" target without pulling in a tokenizer dependency (see README
# design-decision section). Overridable via .env: CHUNK_WORDS / OVERLAP_WORDS.


def _load_pdf(path: Path) -> str:
    """Extract text from a PDF using pypdf."""
    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages)


def _load_text(path: Path) -> str:
    """Read a .md or .txt file as UTF-8 text."""
    return path.read_text(encoding="utf-8")


_LOADERS = {".pdf": _load_pdf, ".md": _load_text, ".txt": _load_text}


def load_documents(data_dir: Path) -> list[tuple[str, str]]:
    """Load all .pdf, .md, .txt files from data_dir.

    Returns a list of (source_file_name, raw_text) tuples.
    """
    docs: list[tuple[str, str]] = []
    for path in sorted(data_dir.iterdir()):
        if not path.is_file():
            continue
        loader = _LOADERS.get(path.suffix.lower())
        if loader is None:
            continue
        logger.info("Loading %s", path.name)
        text = loader(path)
        if text.strip():
            docs.append((path.name, text))
    return docs


def chunk_text(text: str, source_file: str) -> list[dict]:
    """Split text into overlapping word-based chunks.

    Returns a list of dicts: {"source_file", "chunk_index", "text"}.
    """
    words = text.split()
    if not words:
        return []

    chunks: list[dict] = []
    step = settings.chunk_words - settings.overlap_words
    chunk_index = 0
    start = 0
    while start < len(words):
        end = start + settings.chunk_words
        chunk = {
            "source_file": source_file,
            "chunk_index": chunk_index,
            "text": " ".join(words[start:end]),
        }
        chunks.append(chunk)
        chunk_index += 1
        if end >= len(words):
            break
        start += step
    return chunks


def _embed_one(text: str) -> list[float]:
    """Embed a single text string via Ollama /api/embed."""
    resp = httpx.post(
        f"{settings.ollama_url}/api/embed",
        json={"model": settings.embed_model, "input": text},
        timeout=120.0,
    )
    resp.raise_for_status()
    return resp.json()["embeddings"][0]


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Embed each chunk via Ollama. Adds 'vector' key to each chunk dict."""
    for i, chunk in enumerate(chunks):
        chunk["vector"] = _embed_one(chunk["text"])
        if (i + 1) % 10 == 0:
            logger.info("Embedded %d/%d chunks", i + 1, len(chunks))
    return chunks


def upsert_to_qdrant(chunks: list[dict]) -> int:
    """Recreate the Qdrant collection and upsert all chunks.

    Returns the number of points upserted.
    """
    client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)

    existing = [c.name for c in client.get_collections().collections]
    if settings.qdrant_collection in existing:
        client.delete_collection(settings.qdrant_collection)
        logger.info("Deleted existing collection '%s'", settings.qdrant_collection)

    client.create_collection(
        collection_name=settings.qdrant_collection,
        vectors_config=qmodels.VectorParams(
            size=EMBED_DIMENSION,
            distance=qmodels.Distance.COSINE,
        ),
    )
    logger.info(
        "Created collection '%s' (dim=%d, cosine)", settings.qdrant_collection, EMBED_DIMENSION
    )

    points = [
        qmodels.PointStruct(
            id=i,
            vector=chunk["vector"],
            payload={
                "source_file": chunk["source_file"],
                "chunk_index": chunk["chunk_index"],
                "text": chunk["text"],
            },
        )
        for i, chunk in enumerate(chunks)
    ]
    client.upsert(collection_name=settings.qdrant_collection, points=points)
    return len(points)


def run_ingest() -> dict:
    """Run the full ingestion pipeline.

    Returns a summary dict with document/chunk/upsert counts.
    """
    data_dir = Path(settings.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    docs = load_documents(data_dir)
    if not docs:
        raise ValueError(f"No .pdf, .md, or .txt files found in {data_dir}")

    logger.info("Loaded %d document(s)", len(docs))

    all_chunks: list[dict] = []
    for source_file, text in docs:
        chunks = chunk_text(text, source_file)
        all_chunks.extend(chunks)
        logger.info("  %s -> %d chunk(s)", source_file, len(chunks))

    if not all_chunks:
        raise ValueError("No text content extracted from documents")

    embed_chunks(all_chunks)
    count = upsert_to_qdrant(all_chunks)

    return {"documents": len(docs), "chunks": len(all_chunks), "upserted": count}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    result = run_ingest()
    print(f"\nIngestion complete: {result}")
