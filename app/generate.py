"""Prompt construction + Ollama generation.

Prompt templates live in app/prompts/*.txt (Phase 6). The currently selected
variant is loaded from disk and cached for the lifetime of the process.
"""

import functools
import logging
from pathlib import Path

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "app" / "prompts"


class ModelNotPulledError(Exception):
    """Raised when the Ollama model has not been pulled into the container."""

    def __init__(self, model: str):
        self.model = model
        super().__init__(
            f"Model '{model}' not found in Ollama. "
            f"Pull it first: docker compose exec ollama ollama pull {model}"
        )


class UnknownPromptVariantError(Exception):
    """Raised when a requested prompt variant is not in app/prompts/."""

    def __init__(self, name: str, available: list[str]):
        self.name = name
        self.available = available
        super().__init__(
            f"Prompt variant '{name}' not found in app/prompts/. "
            f"Available: {available}"
        )


def list_variants() -> list[str]:
    """Return the names of all prompt variants on disk (basenames without .txt)."""
    if not PROMPTS_DIR.is_dir():
        return []
    return sorted(p.stem for p in PROMPTS_DIR.glob("*.txt"))


def resolve_variant(variant: str | None) -> str:
    """Pick the variant to use; if `variant` is None, fall back to the configured default.

    Raises UnknownPromptVariantError if either the requested or the default
    name does not exist on disk.
    """
    available = list_variants()
    chosen = variant if variant is not None else settings.prompt_variant
    if chosen not in available:
        raise UnknownPromptVariantError(chosen, available)
    return chosen


@functools.lru_cache(maxsize=8)
def _load_template(variant: str) -> str:
    """Read a prompt template from disk and cache it."""
    path = PROMPTS_DIR / f"{variant}.txt"
    template = path.read_text(encoding="utf-8")
    logger.info("Loaded prompt variant '%s' from %s", variant, path)
    return template


def build_prompt(question: str, chunks: list[dict], variant: str = "v1_baseline") -> str:
    """Construct the RAG prompt from retrieved chunks and the question."""
    context_parts = [
        f"[{i + 1}] (source: {c['file']}, chunk {c['chunk_index']})\n{c['text']}"
        for i, c in enumerate(chunks)
    ]
    context = "\n\n".join(context_parts)
    return _load_template(variant).format(context=context, question=question)


def generate_answer(
    question: str, chunks: list[dict], variant: str | None = None
) -> str:
    """Build the prompt and call Ollama /api/generate for an answer.

    If `variant` is None, the configured default (`settings.prompt_variant`) is
    used. If `variant` is unrecognized, raises UnknownPromptVariantError.

    Raises httpx.HTTPError if Ollama is unreachable.
    Raises ModelNotPulledError if the model has not been pulled into Ollama.
    """
    chosen = resolve_variant(variant)
    prompt = build_prompt(question, chunks, variant=chosen)
    logger.info(
        "Calling Ollama model '%s' for generation (variant=%s)",
        settings.ollama_model,
        chosen,
    )

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
