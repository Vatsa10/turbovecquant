"""Pluggable payload stores for turbovec.llm.

A PayloadStore is a keyed key-value map from integer vector IDs to JSON-serializable
payloads. Backends: InMemoryStore (default, dev), PickleStore (single-process disk),
RedisStore (multi-process + TTL + eviction).
"""
from __future__ import annotations

from typing import Any, Iterable, Optional, Protocol, runtime_checkable


@runtime_checkable
class PayloadStore(Protocol):
    def get(self, key: int) -> Optional[dict[str, Any]]: ...
    def set(self, key: int, value: dict[str, Any]) -> None: ...
    def delete(self, key: int) -> None: ...
    def incr_hits(self, key: int) -> None: ...
    def keys(self) -> Iterable[int]: ...
    def __len__(self) -> int: ...


from .memory_store import InMemoryStore
from .pickle_store import PickleStore

__all__ = ["PayloadStore", "InMemoryStore", "PickleStore"]

try:
    from .redis_store import RedisStore
    __all__.append("RedisStore")
except ImportError:
    pass
