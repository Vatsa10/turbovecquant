from __future__ import annotations

import json
from typing import Any, Iterable, Optional

try:
    import redis
except ImportError as exc:
    raise ImportError(
        "redis is required to use turbovec.llm.stores.RedisStore. "
        "Install with: pip install turbovec[llm-redis]"
    ) from exc


class RedisStore:
    """Redis-backed payload store.

    Each payload is stored at key ``{namespace}{key}`` as a JSON blob. TTL (if set)
    is applied on every write so frequently-accessed entries naturally live longer
    when combined with ``touch_on_read=True``.

    Hit counters live in a separate hash at ``{namespace}hits`` and are updated
    atomically via ``HINCRBY``.
    """

    def __init__(
        self,
        url: str = "redis://localhost:6379/0",
        *,
        namespace: str = "tv:",
        ttl_seconds: Optional[int] = None,
        touch_on_read: bool = True,
        client: Optional["redis.Redis"] = None,
    ) -> None:
        self._r = client if client is not None else redis.Redis.from_url(url, decode_responses=True)
        self._ns = namespace
        self._ttl = ttl_seconds
        self._touch = touch_on_read and ttl_seconds is not None
        self._hits_key = f"{namespace}hits"
        self._keys_set = f"{namespace}keys"

    def _k(self, key: int) -> str:
        return f"{self._ns}{key}"

    def get(self, key: int) -> Optional[dict[str, Any]]:
        raw = self._r.get(self._k(key))
        if raw is None:
            return None
        if self._touch:
            self._r.expire(self._k(key), self._ttl)
        payload = json.loads(raw)
        hits = self._r.hget(self._hits_key, str(key))
        if hits is not None:
            payload["hits"] = int(hits)
        return payload

    def set(self, key: int, value: dict[str, Any]) -> None:
        data = json.dumps(value)
        pipe = self._r.pipeline()
        if self._ttl is not None:
            pipe.set(self._k(key), data, ex=self._ttl)
        else:
            pipe.set(self._k(key), data)
        pipe.sadd(self._keys_set, str(key))
        pipe.execute()

    def delete(self, key: int) -> None:
        pipe = self._r.pipeline()
        pipe.delete(self._k(key))
        pipe.srem(self._keys_set, str(key))
        pipe.hdel(self._hits_key, str(key))
        pipe.execute()

    def incr_hits(self, key: int) -> None:
        self._r.hincrby(self._hits_key, str(key), 1)

    def keys(self) -> Iterable[int]:
        members = self._r.smembers(self._keys_set)
        out: list[int] = []
        for m in members:
            if self._r.exists(self._k(int(m))):
                out.append(int(m))
            else:
                self._r.srem(self._keys_set, m)
                self._r.hdel(self._hits_key, m)
        return out

    def __len__(self) -> int:
        return len(list(self.keys()))
