"""turbovec.llm — client-side TurboQuant for hosted LLM APIs.

Headline pieces:

* :class:`SemanticCache` — cosine-match prior prompts on a 2-4 bit quantized index.
  Sync + async + streaming, tenant scoping, temperature / tool-call gating,
  PII redaction, negative caching, $-saved tracking, Prometheus metrics.
* :class:`RAGCompressor` — retrieve top-k chunks and pack into a token budget.
* :class:`ConversationMemory` — per-session chat history with optional semantic recall.

Adapters for OpenAI / OpenRouter / Anthropic / Gemini, plus embedders for
OpenAI / Cohere / Voyage / sentence-transformers. A FastAPI OpenAI-compatible
proxy is available in :mod:`turbovec.llm.proxy`.
"""
from __future__ import annotations

from .cache import SemanticCache
from .memory import ConversationMemory
from .observability import AuditLogger, Metrics
from .rag import Chunk, RAGCompressor

__all__ = [
    "SemanticCache",
    "RAGCompressor",
    "Chunk",
    "ConversationMemory",
    "Metrics",
    "AuditLogger",
]
