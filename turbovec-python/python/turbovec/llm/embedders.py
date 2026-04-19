"""Embedder adapters for turbovec.llm.

An ``Embedder`` implements ``embed(texts) -> np.ndarray[n, dim]`` returning
L2-normalized float32 rows. Cosine similarity on the unit hypersphere is what
the TurboQuant scan computes.
"""
from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

import numpy as np


def _normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x[None, :]
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return np.ascontiguousarray(x / norms, dtype=np.float32)


@runtime_checkable
class Embedder(Protocol):
    dim: int
    def embed(self, texts: list[str]) -> np.ndarray: ...


class OpenAIEmbedder:
    """OpenAI (and OpenAI-compatible) embeddings.

    Works with OpenRouter, Together, etc., via ``base_url``.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        dim: Optional[int] = None,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "openai is required for OpenAIEmbedder. Install with: pip install turbovec[llm-openai]"
            ) from exc
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self._dim = dim

    @property
    def dim(self) -> int:
        if self._dim is None:
            probe = self.embed(["probe"])
            self._dim = int(probe.shape[1])
        return self._dim

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dim or 1), dtype=np.float32)
        resp = self._client.embeddings.create(model=self.model, input=texts)
        vecs = np.asarray([d.embedding for d in resp.data], dtype=np.float32)
        if self._dim is None:
            self._dim = int(vecs.shape[1])
        return _normalize(vecs)


class CohereEmbedder:
    def __init__(
        self,
        model: str = "embed-english-v3.0",
        *,
        api_key: Optional[str] = None,
        input_type: str = "search_document",
    ) -> None:
        try:
            import cohere
        except ImportError as exc:
            raise ImportError(
                "cohere is required for CohereEmbedder. Install with: pip install turbovec[llm-cohere]"
            ) from exc
        self._client = cohere.Client(api_key=api_key) if api_key else cohere.Client()
        self.model = model
        self._input_type = input_type
        self._dim: Optional[int] = None

    @property
    def dim(self) -> int:
        if self._dim is None:
            self.embed(["probe"])
        assert self._dim is not None
        return self._dim

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dim or 1), dtype=np.float32)
        resp = self._client.embed(texts=texts, model=self.model, input_type=self._input_type)
        vecs = np.asarray(resp.embeddings, dtype=np.float32)
        self._dim = int(vecs.shape[1])
        return _normalize(vecs)


class VoyageEmbedder:
    def __init__(
        self,
        model: str = "voyage-3",
        *,
        api_key: Optional[str] = None,
        input_type: str = "document",
    ) -> None:
        try:
            import voyageai
        except ImportError as exc:
            raise ImportError(
                "voyageai is required for VoyageEmbedder. Install with: pip install turbovec[llm-voyage]"
            ) from exc
        self._client = voyageai.Client(api_key=api_key) if api_key else voyageai.Client()
        self.model = model
        self._input_type = input_type
        self._dim: Optional[int] = None

    @property
    def dim(self) -> int:
        if self._dim is None:
            self.embed(["probe"])
        assert self._dim is not None
        return self._dim

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dim or 1), dtype=np.float32)
        resp = self._client.embed(texts, model=self.model, input_type=self._input_type)
        vecs = np.asarray(resp.embeddings, dtype=np.float32)
        self._dim = int(vecs.shape[1])
        return _normalize(vecs)


class SentenceTransformerEmbedder:
    """Offline embeddings via sentence-transformers."""

    def __init__(self, model: str = "BAAI/bge-small-en-v1.5", *, device: Optional[str] = None) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required. Install with: pip install turbovec[llm-local]"
            ) from exc
        self._model = SentenceTransformer(model, device=device)
        self._dim = int(self._model.get_sentence_embedding_dimension())

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        vecs = self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=False)
        return _normalize(np.asarray(vecs, dtype=np.float32))


__all__ = [
    "Embedder",
    "OpenAIEmbedder",
    "CohereEmbedder",
    "VoyageEmbedder",
    "SentenceTransformerEmbedder",
]
