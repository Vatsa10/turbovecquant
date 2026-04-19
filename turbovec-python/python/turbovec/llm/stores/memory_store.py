from __future__ import annotations

from typing import Any, Iterable, Optional


class InMemoryStore:
    def __init__(self) -> None:
        self._data: dict[int, dict[str, Any]] = {}

    def get(self, key: int) -> Optional[dict[str, Any]]:
        return self._data.get(key)

    def set(self, key: int, value: dict[str, Any]) -> None:
        self._data[key] = dict(value)

    def delete(self, key: int) -> None:
        self._data.pop(key, None)

    def incr_hits(self, key: int) -> None:
        entry = self._data.get(key)
        if entry is not None:
            entry["hits"] = int(entry.get("hits", 0)) + 1

    def keys(self) -> Iterable[int]:
        return list(self._data.keys())

    def __len__(self) -> int:
        return len(self._data)
