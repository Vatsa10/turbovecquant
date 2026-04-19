"""turbovec.llm — client-side TurboQuant for hosted LLM APIs.

Two headline pieces:

* :class:`SemanticCache` — cosine-match prior prompts on a 2-4 bit quantized index
  and reuse the cached completion instead of calling the paid API.
* :class:`RAGCompressor` — retrieve top-k chunks and pack them into a token
  budget before sending, to reduce billed input tokens.

Plus :class:`ConversationMemory` for per-session chat history with optional
semantic recall across turns.
"""
from __future__ import annotations

from .cache import SemanticCache
from .memory import ConversationMemory
from .rag import Chunk, RAGCompressor

__all__ = [
    "SemanticCache",
    "RAGCompressor",
    "Chunk",
    "ConversationMemory",
]
