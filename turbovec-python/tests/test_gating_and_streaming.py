from __future__ import annotations

import asyncio

from turbovec.llm import SemanticCache

from _llm_fakes import FakeEmbedder, FakeLLM


def _msg(user):
    return [{"role": "user", "content": user}]


def test_temperature_bypasses_cache():
    llm = FakeLLM()
    cache = SemanticCache(embedder=FakeEmbedder(), llm=llm, similarity_threshold=0.9)
    cache.complete(_msg("hello"), temperature=0.7)
    cache.complete(_msg("hello"), temperature=0.7)
    assert llm.calls == 2
    assert cache.stats()["bypassed"] == 2
    assert cache.stats()["hits"] == 0


def test_temperature_zero_caches():
    llm = FakeLLM()
    cache = SemanticCache(embedder=FakeEmbedder(), llm=llm, similarity_threshold=0.9)
    cache.complete(_msg("hello"), temperature=0)
    cache.complete(_msg("hello"), temperature=0)
    assert llm.calls == 1


def test_tools_kwarg_bypasses():
    llm = FakeLLM()
    cache = SemanticCache(embedder=FakeEmbedder(), llm=llm, similarity_threshold=0.9)
    tools = [{"type": "function", "function": {"name": "x"}}]
    cache.complete(_msg("hello"), tools=tools)
    cache.complete(_msg("hello"), tools=tools)
    assert llm.calls == 2


def test_force_cache_override():
    llm = FakeLLM()
    cache = SemanticCache(embedder=FakeEmbedder(), llm=llm, similarity_threshold=0.9)
    cache.complete(_msg("hello"), temperature=0.7, force_cache=True)
    cache.complete(_msg("hello"), temperature=0.7, force_cache=True)
    assert llm.calls == 1


def test_stream_cache_through():
    llm = FakeLLM(reply="hi")
    cache = SemanticCache(embedder=FakeEmbedder(), llm=llm, similarity_threshold=0.9)
    out1 = "".join(cache.stream_complete(_msg("ping")))
    assert out1 == "hi:ping"
    assert llm.calls == 1
    # second call — should stream from cache, no upstream
    out2 = "".join(cache.stream_complete(_msg("ping")))
    assert out2 == "hi:ping"
    assert llm.calls == 1
    assert cache.stats()["hits"] == 1


def test_astream_cache_through():
    llm = FakeLLM(reply="hi")
    cache = SemanticCache(embedder=FakeEmbedder(), llm=llm, similarity_threshold=0.9)

    async def run():
        chunks = []
        async for c in cache.astream_complete(_msg("ping")):
            chunks.append(c)
        return "".join(chunks)

    first = asyncio.run(run())
    second = asyncio.run(run())
    assert first == second == "hi:ping"
    assert llm.calls == 1


def test_acomplete():
    llm = FakeLLM()
    cache = SemanticCache(embedder=FakeEmbedder(), llm=llm, similarity_threshold=0.9)
    r1 = asyncio.run(cache.acomplete(_msg("foo")))
    r2 = asyncio.run(cache.acomplete(_msg("foo")))
    assert r1.text == r2.text
    assert llm.calls == 1
