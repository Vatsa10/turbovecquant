"""Semantic response cache for LLM APIs, backed by TurboQuantIndex."""
from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from .._turbovec import TurboQuantIndex
from .embedders import Embedder
from .llm import LLM, CompletionResult
from .stores import InMemoryStore, PayloadStore, PickleStore


def _last_user(messages: list[dict]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            return m.get("content", "")
    return ""


def _system_fingerprint(messages: list[dict]) -> str:
    sys_text = "\n\n".join(m.get("content", "") for m in messages if m.get("role") == "system")
    return hashlib.sha256(sys_text.encode("utf-8")).hexdigest()[:16]


class SemanticCache:
    """Cache LLM completions keyed by the embedding of the user's request.

    A near-duplicate query (cosine similarity ≥ ``similarity_threshold``) reuses
    the prior response instead of calling the upstream LLM. System-prompt
    fingerprints must match so two personas sharing the cache don't cross-talk.
    """

    def __init__(
        self,
        *,
        embedder: Embedder,
        llm: LLM,
        bit_width: int = 4,
        similarity_threshold: float = 0.93,
        store: Optional[PayloadStore] = None,
        path: Optional[str | Path] = None,
        key_fn: Optional[Callable[[list[dict]], str]] = None,
    ) -> None:
        self._embedder = embedder
        self._llm = llm
        self._threshold = similarity_threshold
        self._key_fn = key_fn or _last_user
        self._path = Path(path) if path else None

        dim = embedder.dim
        index_path = self._path.with_suffix(".tv") if self._path else None
        if index_path and index_path.exists():
            self._index = TurboQuantIndex.load(str(index_path))
        else:
            self._index = TurboQuantIndex(dim, bit_width)

        if store is not None:
            self._store = store
        elif self._path is not None:
            self._store = PickleStore(self._path.with_suffix(".pkl"))
        else:
            self._store = InMemoryStore()

        self._hits = 0
        self._misses = 0
        self._tokens_saved = 0
        self._dead_ids: set[int] = set()

    @property
    def index(self) -> TurboQuantIndex:
        return self._index

    @property
    def store(self) -> PayloadStore:
        return self._store

    def complete(self, messages: list[dict], **llm_kwargs: Any) -> CompletionResult:
        key_text = self._key_fn(messages)
        sys_fp = _system_fingerprint(messages)
        qvec = self._embedder.embed([key_text])

        hit = self._lookup(qvec, sys_fp)
        if hit is not None:
            self._hits += 1
            self._tokens_saved += int(hit.get("tokens_in", 0)) + int(hit.get("tokens_out", 0))
            return CompletionResult(
                text=hit["response"],
                tokens_in=0,
                tokens_out=0,
                model=hit.get("model", ""),
                raw={"cached": True, "entry": hit},
            )

        self._misses += 1
        result = self._llm.complete(messages, **llm_kwargs)
        self._insert(qvec, sys_fp, messages, result, key_text)
        return result

    def _lookup(self, qvec: np.ndarray, sys_fp: str) -> Optional[dict]:
        if len(self._index) == 0:
            return None
        k = min(5, len(self._index))
        scores, indices = self._index.search(qvec, k)
        for score, idx in zip(scores[0], indices[0]):
            idx_i = int(idx)
            if idx_i in self._dead_ids:
                continue
            if float(score) < self._threshold:
                return None
            entry = self._store.get(idx_i)
            if entry is None:
                self._dead_ids.add(idx_i)
                continue
            if entry.get("system_fp") != sys_fp:
                continue
            self._store.incr_hits(idx_i)
            return entry
        return None

    def _insert(
        self,
        qvec: np.ndarray,
        sys_fp: str,
        messages: list[dict],
        result: CompletionResult,
        key_text: str,
    ) -> None:
        new_id = len(self._index)
        self._index.add(qvec)
        prompt_hash = hashlib.sha256(key_text.encode("utf-8")).hexdigest()[:16]
        self._store.set(
            new_id,
            {
                "prompt_hash": prompt_hash,
                "system_fp": sys_fp,
                "messages": messages,
                "response": result.text,
                "tokens_in": result.tokens_in,
                "tokens_out": result.tokens_out,
                "model": result.model,
                "created_at": time.time(),
                "hits": 0,
            },
        )

    def stats(self) -> dict[str, Any]:
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": (self._hits / total) if total else 0.0,
            "tokens_saved": self._tokens_saved,
            "entries": len(self._store),
            "index_size": len(self._index),
            "dead_slots": len(self._dead_ids),
        }

    def save(self) -> None:
        if self._path is None:
            raise ValueError("SemanticCache was constructed without a path; nothing to save")
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._index.write(str(self._path.with_suffix(".tv")))
        if isinstance(self._store, PickleStore):
            self._store.save()

    def compact(self) -> None:
        """Rebuild the index from live store entries, dropping dead slots.

        Required after Redis TTL evictions if you want to reclaim vector-ID space.
        Re-embeds from stored ``key_text`` — expensive; run offline.
        """
        live_ids = [i for i in self._store.keys() if i not in self._dead_ids]
        live_entries = [(i, self._store.get(i)) for i in live_ids]
        live_entries = [(i, e) for i, e in live_entries if e is not None]
        dim = self._index.dim
        bit_width = self._index.bit_width
        new_index = TurboQuantIndex(dim, bit_width)
        new_store: PayloadStore
        if isinstance(self._store, InMemoryStore):
            new_store = InMemoryStore()
        else:
            new_store = self._store  # keep persistent backends in place; we rewrite keys below
            for i in list(self._store.keys()):
                self._store.delete(i)

        for new_id, (_old_id, entry) in enumerate(live_entries):
            key_text = self._key_fn(entry["messages"])
            vec = self._embedder.embed([key_text])
            new_index.add(vec)
            new_store.set(new_id, entry)

        self._index = new_index
        self._store = new_store
        self._dead_ids.clear()


__all__ = ["SemanticCache"]
