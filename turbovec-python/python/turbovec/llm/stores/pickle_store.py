from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Iterable, Optional


class PickleStore:
    """On-disk payload store backed by a single pickle file.

    Writes are buffered in memory; call ``save()`` to flush. Mirrors the sidecar
    pattern used by :class:`turbovec.langchain.TurboQuantVectorStore`.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._data: dict[int, dict[str, Any]] = {}
        if self._path.exists():
            with open(self._path, "rb") as f:
                self._data = pickle.load(f)

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

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "wb") as f:
            pickle.dump(self._data, f)
