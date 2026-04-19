from __future__ import annotations

import time

import pytest

from turbovec.llm import SemanticCache
from turbovec.llm.pricing import estimate_cost_usd
from turbovec.llm.redaction import default_redactor

from _llm_fakes import FakeEmbedder, FakeLLM


def _msg(user):
    return [{"role": "user", "content": user}]


def test_tenant_isolation():
    llm = FakeLLM()
    cache = SemanticCache(embedder=FakeEmbedder(), llm=llm, similarity_threshold=0.9)
    cache.complete(_msg("hello"), tenant="alice")
    cache.complete(_msg("hello"), tenant="bob")   # different tenant → miss
    cache.complete(_msg("hello"), tenant="alice") # same tenant → hit
    assert llm.calls == 2
    stats = cache.stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 2


def test_invalidate_by_tenant():
    llm = FakeLLM()
    cache = SemanticCache(embedder=FakeEmbedder(), llm=llm, similarity_threshold=0.9)
    cache.complete(_msg("one"), tenant="a")
    cache.complete(_msg("two"), tenant="b")
    removed = cache.invalidate(tenant="a")
    assert removed == 1
    assert cache.stats()["entries"] == 1


def test_invalidate_older_than():
    llm = FakeLLM()
    cache = SemanticCache(embedder=FakeEmbedder(), llm=llm, similarity_threshold=0.9)
    cache.complete(_msg("old"))
    time.sleep(0.05)
    cache.complete(_msg("new"))
    removed = cache.invalidate(older_than_seconds=0.04)
    assert removed >= 1


def test_dollars_saved_tracked():
    llm = FakeLLM(model="gpt-4o-mini", tokens_in=1000, tokens_out=500)
    cache = SemanticCache(embedder=FakeEmbedder(), llm=llm, similarity_threshold=0.9)
    cache.complete(_msg("q"))
    cache.complete(_msg("q"))
    stats = cache.stats()
    assert stats["tokens_saved"] == 1500
    expected = estimate_cost_usd("gpt-4o-mini", 1000, 500)
    assert abs(stats["dollars_saved"] - expected) < 1e-9
    assert stats["dollars_saved"] > 0


def test_negative_cache():
    llm = FakeLLM(fail_times=10)
    cache = SemanticCache(
        embedder=FakeEmbedder(), llm=llm, similarity_threshold=0.9,
        negative_cache_seconds=10.0,
    )
    with pytest.raises(RuntimeError, match="kaboom"):
        cache.complete(_msg("query"))
    assert llm.calls == 1
    # second call hits the negative entry — upstream NOT called again
    with pytest.raises(RuntimeError, match="negative-cached"):
        cache.complete(_msg("query"))
    assert llm.calls == 1


def test_negative_cache_expires():
    llm = FakeLLM(fail_times=1)
    cache = SemanticCache(
        embedder=FakeEmbedder(), llm=llm, similarity_threshold=0.9,
        negative_cache_seconds=0.05,
    )
    with pytest.raises(RuntimeError):
        cache.complete(_msg("query"))
    time.sleep(0.1)
    r = cache.complete(_msg("query"))
    assert r.text.startswith("fake-reply")


def test_redactor_applied_to_response():
    llm = FakeLLM(reply="contact alice@example.com for details")
    cache = SemanticCache(embedder=FakeEmbedder(), llm=llm, similarity_threshold=0.9)
    cache.complete(_msg("who do I email"))
    # second call returns the cached (redacted) response
    r = cache.complete(_msg("who do I email"))
    assert "alice@example.com" not in r.text
    assert "[EMAIL]" in r.text


def test_disable_redactor():
    llm = FakeLLM(reply="email=alice@example.com")
    cache = SemanticCache(embedder=FakeEmbedder(), llm=llm, similarity_threshold=0.9, redactor=None)
    cache.complete(_msg("q"))
    r = cache.complete(_msg("q"))
    assert "alice@example.com" in r.text


def test_default_redactor_patterns():
    s = "ssn 123-45-6789, card 4111 1111 1111 1111, key sk-abcdefghij1234567890"
    out = default_redactor(s)
    assert "[SSN]" in out
    assert "[CARD]" in out
    assert "[API_KEY]" in out
