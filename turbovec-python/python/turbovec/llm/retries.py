"""Lightweight retry helpers with exponential backoff + jitter.

No dependency on tenacity — keeps the base install small.
"""
from __future__ import annotations

import asyncio
import random
import time
from functools import wraps
from typing import Any, Awaitable, Callable, Iterable, TypeVar

T = TypeVar("T")

RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})


def _is_retryable(exc: BaseException, retryable_status: Iterable[int]) -> bool:
    code = getattr(exc, "status_code", None)
    if code is None:
        resp = getattr(exc, "response", None)
        code = getattr(resp, "status_code", None) if resp is not None else None
    if code is not None:
        try:
            return int(code) in retryable_status
        except (TypeError, ValueError):
            return False
    # Transport-layer errors without a status code → retry.
    name = type(exc).__name__.lower()
    return any(k in name for k in ("timeout", "connection", "apierror"))


def with_retries(
    fn: Callable[..., T],
    *,
    max_attempts: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 10.0,
    retryable_status: Iterable[int] = RETRYABLE_STATUS,
    on_retry: Callable[[int, BaseException], None] | None = None,
) -> Callable[..., T]:
    status = frozenset(retryable_status)

    @wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        last: BaseException | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                return fn(*args, **kwargs)
            except BaseException as e:
                last = e
                if attempt >= max_attempts or not _is_retryable(e, status):
                    raise
                if on_retry:
                    on_retry(attempt, e)
                delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
                time.sleep(delay + random.uniform(0, delay * 0.25))
        assert last is not None
        raise last
    return wrapper


def with_retries_async(
    fn: Callable[..., Awaitable[T]],
    *,
    max_attempts: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 10.0,
    retryable_status: Iterable[int] = RETRYABLE_STATUS,
) -> Callable[..., Awaitable[T]]:
    status = frozenset(retryable_status)

    @wraps(fn)
    async def wrapper(*args: Any, **kwargs: Any) -> T:
        last: BaseException | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                return await fn(*args, **kwargs)
            except BaseException as e:
                last = e
                if attempt >= max_attempts or not _is_retryable(e, status):
                    raise
                delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
                await asyncio.sleep(delay + random.uniform(0, delay * 0.25))
        assert last is not None
        raise last
    return wrapper


__all__ = ["with_retries", "with_retries_async", "RETRYABLE_STATUS"]
