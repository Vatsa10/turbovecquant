from __future__ import annotations

import pytest

from turbovec.llm import SemanticCache
from turbovec.llm.eviction import lfu_evict, ttl_evict
from turbovec.llm.retries import with_retries

from _llm_fakes import FakeEmbedder, FakeLLM


def _msg(t):
    return [{"role": "user", "content": t}]


def test_lfu_evict_drops_lowest_hits():
    llm = FakeLLM()
    cache = SemanticCache(embedder=FakeEmbedder(), llm=llm, similarity_threshold=0.9)
    for q in ["alpha unique", "beta unique", "gamma unique", "delta unique"]:
        cache.complete(_msg(q))
    # boost hits on alpha
    for _ in range(3):
        cache.complete(_msg("alpha unique"))
    victims = lfu_evict(cache.store, max_entries=2, keep_ratio=1.0)
    assert len(victims) == 2
    remaining = []
    for k in cache.store.keys():
        e = cache.store.get(k)
        if e:
            remaining.append(e["messages"][-1]["content"])
    assert any("alpha" in r for r in remaining)


def test_ttl_evict():
    import time as _t
    llm = FakeLLM()
    cache = SemanticCache(embedder=FakeEmbedder(), llm=llm, similarity_threshold=0.9)
    cache.complete(_msg("a"))
    _t.sleep(0.05)
    cache.complete(_msg("b"))
    victims = ttl_evict(cache.store, max_age_seconds=0.04)
    assert len(victims) >= 1


def test_retries_gives_up_after_max():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        e = RuntimeError("boom")
        setattr(e, "status_code", 503)
        raise e

    wrapped = with_retries(flaky, max_attempts=3, base_delay=0.001, max_delay=0.002)
    with pytest.raises(RuntimeError):
        wrapped()
    assert calls["n"] == 3


def test_retries_does_not_retry_non_retryable():
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        e = RuntimeError("bad request")
        setattr(e, "status_code", 400)
        raise e

    wrapped = with_retries(fn, max_attempts=5, base_delay=0.001)
    with pytest.raises(RuntimeError):
        wrapped()
    assert calls["n"] == 1


def test_retries_succeeds_on_second_attempt():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 2:
            e = RuntimeError("boom")
            setattr(e, "status_code", 429)
            raise e
        return "ok"

    wrapped = with_retries(flaky, max_attempts=5, base_delay=0.001)
    assert wrapped() == "ok"
    assert calls["n"] == 2
