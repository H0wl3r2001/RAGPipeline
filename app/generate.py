"""Prompt construction + Ollama generation."""

import logging

import httpx

from app.config import settings

logger = logging.getLogger(__name__)


class ModelNotPulledError(Exception):
    """Raised when the Ollama model has not been pulled into the container."""

    def __init__(self, model: str):
        self.model = model
        super().__init__(
            f"Model '{model}' not found in Ollama. "
            f"Pull it first: docker compose exec ollama ollama pull {model}"
        )

PROMPT_TEMPLATE = """Answer the question using only the context below.
If the context does not contain enough information to answer, say "I don't know."
Do not use any outside knowledge.

Context:
{context}

Question: {question}

Answer:"""


def build_prompt(question: str, chunks: list[dict]) -> str:
    """Construct the RAG prompt from retrieved chunks and the question."""
    context_parts = [
        f"[{i + 1}] (source: {c['file']}, chunk {c['chunk_index']})\n{c['text']}"
        for i, c in enumerate(chunks)
    ]
    context = "\n\n".join(context_parts)
    return PROMPT_TEMPLATE.format(context=context, question=question)


def generate_answer(question: str, chunks: list[dict]) -> str:
    """Build the prompt and call Ollama /api/generate for an answer.

    Raises httpx.HTTPError if Ollama is unreachable.
    Raises ModelNotPulledError if the model has not been pulled into Ollama.
    """
    prompt = build_prompt(question, chunks)
    logger.info("Calling Ollama model '%s' for generation", settings.ollama_model)

    resp = httpx.post(
        f"{settings.ollama_url}/api/generate",
        json={
            "model": settings.ollama_model,
            "prompt": prompt,
            "stream": False,
        },
        timeout=600.0,
    )
    if resp.status_code == 404:
        raise ModelNotPulledError(settings.ollama_model)
    resp.raise_for_status()
    return resp.json()["response"].strip()
