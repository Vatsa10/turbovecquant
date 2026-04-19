"""Eviction policies for PayloadStore (LFU / TTL / size).

Run periodically — not in the hot path. The underlying store may already have
native TTL (Redis ``EXPIRE``); this layer covers size caps and LFU across
backends that don't.
"""
from __future__ import annotations

import time
from typing import Optional

from .stores import PayloadStore


def lfu_evict(store: PayloadStore, max_entries: int, *, keep_ratio: float = 0.9) -> list[int]:
    """Drop lowest-hit entries until ``len(store) <= max_entries * keep_ratio``.

    Returns the list of evicted IDs so the caller can track them in the
    cache's dead-ID set (the TurboQuant index can't delete slots).
    """
    if max_entries <= 0:
        return []
    target = int(max_entries * keep_ratio)
    keys = list(store.keys())
    if len(keys) <= max_entries:
        return []
    entries = []
    for k in keys:
        e = store.get(k)
        if e is None:
            continue
        entries.append((int(e.get("hits", 0)), float(e.get("created_at", 0.0)), k))
    entries.sort()  # lowest hits, oldest first
    victims = [k for _h, _t, k in entries[: max(0, len(entries) - target)]]
    for k in victims:
        store.delete(k)
    return victims


def ttl_evict(store: PayloadStore, max_age_seconds: float) -> list[int]:
    now = time.time()
    victims: list[int] = []
    for k in list(store.keys()):
        e = store.get(k)
        if e is None:
            continue
        if now - float(e.get("created_at", now)) > max_age_seconds:
            store.delete(k)
            victims.append(k)
    return victims


__all__ = ["lfu_evict", "ttl_evict"]
