"""Per-session conversation memory with optional semantic recall.

Short-term history lives in a backend (in-memory dict or Redis list with TTL).
If an ``embedder`` is supplied, each turn is also indexed into a
:class:`TurboQuantIndex` so ``recall(session_id, query)`` can surface relevant
past turns without resending the full transcript.
"""
from __future__ import annotations

import json
from typing import Any, Optional, Protocol, runtime_checkable

import numpy as np

from .._turbovec import TurboQuantIndex
from .embedders import Embedder


@runtime_checkable
class HistoryBackend(Protocol):
    def append(self, session_id: str, message: dict) -> None: ...
    def history(self, session_id: str, last_n: Optional[int]) -> list[dict]: ...
    def clear(self, session_id: str) -> None: ...


class InMemoryHistoryBackend:
    def __init__(self) -> None:
        self._d: dict[str, list[dict]] = {}

    def append(self, session_id: str, message: dict) -> None:
        self._d.setdefault(session_id, []).append(dict(message))

    def history(self, session_id: str, last_n: Optional[int]) -> list[dict]:
        msgs = self._d.get(session_id, [])
        if last_n is None:
            return list(msgs)
        return list(msgs[-last_n:])

    def clear(self, session_id: str) -> None:
        self._d.pop(session_id, None)


class RedisHistoryBackend:
    def __init__(
        self,
        url: str = "redis://localhost:6379/0",
        *,
        namespace: str = "tv:mem:",
        ttl_seconds: Optional[int] = 3600,
        client: Optional[Any] = None,
    ) -> None:
        try:
            import redis
        except ImportError as exc:
            raise ImportError(
                "redis is required for RedisHistoryBackend. Install with: pip install turbovec[llm-redis]"
            ) from exc
        self._r = client if client is not None else redis.Redis.from_url(url, decode_responses=True)
        self._ns = namespace
        self._ttl = ttl_seconds

    def _k(self, session_id: str) -> str:
        return f"{self._ns}{session_id}"

    def append(self, session_id: str, message: dict) -> None:
        pipe = self._r.pipeline()
        pipe.rpush(self._k(session_id), json.dumps(message))
        if self._ttl is not None:
            pipe.expire(self._k(session_id), self._ttl)
        pipe.execute()

    def history(self, session_id: str, last_n: Optional[int]) -> list[dict]:
        start = -last_n if last_n is not None else 0
        raw = self._r.lrange(self._k(session_id), start, -1)
        return [json.loads(r) for r in raw]

    def clear(self, session_id: str) -> None:
        self._r.delete(self._k(session_id))


class ConversationMemory:
    """Per-session chat history.

    If ``embedder`` is provided, each appended message is indexed so that
    ``recall(session_id, query, k)`` returns semantically nearest prior messages
    across this memory's lifetime.
    """

    def __init__(
        self,
        *,
        backend: Optional[HistoryBackend] = None,
        embedder: Optional[Embedder] = None,
        bit_width: int = 4,
    ) -> None:
        self._backend = backend if backend is not None else InMemoryHistoryBackend()
        self._embedder = embedder
        self._index: Optional[TurboQuantIndex] = None
        if embedder is not None:
            self._index = TurboQuantIndex(embedder.dim, bit_width)
        self._slot_session: list[str] = []
        self._slot_message: list[dict] = []

    def append(self, session_id: str, message: dict) -> None:
        self._backend.append(session_id, message)
        if self._index is not None and self._embedder is not None:
            content = message.get("content", "")
            if content:
                vec = self._embedder.embed([content])
                self._index.add(vec)
                self._slot_session.append(session_id)
                self._slot_message.append(dict(message))

    def history(self, session_id: str, last_n: Optional[int] = None) -> list[dict]:
        return self._backend.history(session_id, last_n)

    def clear(self, session_id: str) -> None:
        self._backend.clear(session_id)

    def recall(self, session_id: str, query: str, k: int = 5) -> list[dict]:
        if self._index is None or self._embedder is None:
            raise RuntimeError("recall() requires an embedder at construction time")
        if len(self._index) == 0:
            return []
        qvec = self._embedder.embed([query])
        over_k = min(len(self._index), max(k * 4, k))
        scores, indices = self._index.search(qvec, over_k)
        out: list[dict] = []
        for score, idx in zip(scores[0], indices[0]):
            i = int(idx)
            if self._slot_session[i] != session_id:
                continue
            msg = dict(self._slot_message[i])
            msg["_score"] = float(score)
            out.append(msg)
            if len(out) >= k:
                break
        return out


__all__ = [
    "ConversationMemory",
    "HistoryBackend",
    "InMemoryHistoryBackend",
    "RedisHistoryBackend",
]
