"""RAG context compressor: retrieve + token-budget-pack chunks before sending to an LLM."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np

from .._turbovec import TurboQuantIndex
from .embedders import Embedder


@dataclass
class Chunk:
    text: str
    score: float
    metadata: dict[str, Any]


def _fixed_splitter(size: int, overlap: int) -> Callable[[str], list[str]]:
    step = max(1, size - overlap)
    def split(text: str) -> list[str]:
        if len(text) <= size:
            return [text]
        return [text[i : i + size] for i in range(0, len(text), step)]
    return split


def _default_token_counter() -> Callable[[str], int]:
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return lambda s: len(enc.encode(s))
    except ImportError:
        return lambda s: max(1, len(s) // 4)


class RAGCompressor:
    """Retrieve top-k chunks and pack them into a token budget."""

    def __init__(
        self,
        *,
        embedder: Embedder,
        bit_width: int = 4,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        token_counter: Optional[Callable[[str], int]] = None,
        splitter: Optional[Callable[[str], list[str]]] = None,
    ) -> None:
        self._embedder = embedder
        self._index = TurboQuantIndex(embedder.dim, bit_width)
        self._splitter = splitter or _fixed_splitter(chunk_size, chunk_overlap)
        self._count_tokens = token_counter or _default_token_counter()
        self._chunks: list[str] = []
        self._meta: list[dict[str, Any]] = []

    @property
    def index(self) -> TurboQuantIndex:
        return self._index

    def add_documents(
        self,
        texts: list[str],
        metadatas: Optional[list[dict[str, Any]]] = None,
    ) -> int:
        if not texts:
            return 0
        if metadatas is None:
            metadatas = [{} for _ in texts]
        if len(metadatas) != len(texts):
            raise ValueError("texts and metadatas must have the same length")

        all_chunks: list[str] = []
        all_meta: list[dict[str, Any]] = []
        for doc_i, (text, meta) in enumerate(zip(texts, metadatas)):
            for chunk in self._splitter(text):
                all_chunks.append(chunk)
                all_meta.append({**meta, "_doc_index": doc_i})

        if not all_chunks:
            return 0
        vecs = self._embedder.embed(all_chunks)
        self._index.add(vecs)
        self._chunks.extend(all_chunks)
        self._meta.extend(all_meta)
        return len(all_chunks)

    def retrieve(self, query: str, k: int = 10) -> list[Chunk]:
        if len(self._index) == 0:
            return []
        qvec = self._embedder.embed([query])
        k = min(k, len(self._index))
        scores, indices = self._index.search(qvec, k)
        out: list[Chunk] = []
        for score, idx in zip(scores[0], indices[0]):
            i = int(idx)
            out.append(Chunk(text=self._chunks[i], score=float(score), metadata=dict(self._meta[i])))
        return out

    def compress(
        self,
        query: str,
        token_budget: int,
        *,
        over_retrieve: int = 20,
        separator: str = "\n\n---\n\n",
        return_chunks: bool = False,
    ) -> str | list[Chunk]:
        """Greedy-pack highest-scoring chunks until ``token_budget`` is exhausted."""
        if token_budget <= 0 or len(self._index) == 0:
            return [] if return_chunks else ""
        candidates = self.retrieve(query, k=over_retrieve)
        sep_tokens = self._count_tokens(separator)
        kept: list[Chunk] = []
        used = 0
        for c in candidates:
            cost = self._count_tokens(c.text) + (sep_tokens if kept else 0)
            if used + cost > token_budget:
                continue
            kept.append(c)
            used += cost
        if return_chunks:
            return kept
        return separator.join(c.text for c in kept)


__all__ = ["RAGCompressor", "Chunk"]
