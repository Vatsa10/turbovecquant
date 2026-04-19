from __future__ import annotations

from turbovec.llm import SemanticCache

from _llm_fakes import FakeEmbedder, FakeLLM


def _msg(user, system=None):
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": user})
    return msgs


def test_identical_prompt_hits():
    llm = FakeLLM()
    cache = SemanticCache(embedder=FakeEmbedder(), llm=llm, similarity_threshold=0.9)
    cache.complete(_msg("what is the capital of france"))
    cache.complete(_msg("what is the capital of france"))
    assert llm.calls == 1
    stats = cache.stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["tokens_saved"] == 30


def test_different_prompt_misses():
    llm = FakeLLM()
    cache = SemanticCache(embedder=FakeEmbedder(), llm=llm, similarity_threshold=0.95)
    cache.complete(_msg("what is the capital of france"))
    cache.complete(_msg("explain the general theory of relativity in depth"))
    assert llm.calls == 2


def test_different_system_prompt_does_not_collide():
    llm = FakeLLM(reply="A")
    cache = SemanticCache(embedder=FakeEmbedder(), llm=llm, similarity_threshold=0.9)
    cache.complete(_msg("hello there", system="you are a pirate"))
    llm.reply = "B"
    r = cache.complete(_msg("hello there", system="you are a formal assistant"))
    assert llm.calls == 2
    assert r.text.startswith("B:")


def test_persist_and_reload(tmp_path):
    path = tmp_path / "cache"
    llm = FakeLLM()
    cache = SemanticCache(embedder=FakeEmbedder(), llm=llm, path=path, similarity_threshold=0.9)
    cache.complete(_msg("what is the capital of france"))
    cache.save()

    llm2 = FakeLLM()
    cache2 = SemanticCache(embedder=FakeEmbedder(), llm=llm2, path=path, similarity_threshold=0.9)
    cache2.complete(_msg("what is the capital of france"))
    assert llm2.calls == 0
    assert cache2.stats()["hits"] == 1


def test_compact_drops_dead_slots():
    llm = FakeLLM()
    cache = SemanticCache(embedder=FakeEmbedder(), llm=llm, similarity_threshold=0.9)
    cache.complete(_msg("alpha beta gamma"))
    cache.complete(_msg("one two three"))
    cache._store.delete(0)
    cache._dead_ids.add(0)
    assert len(cache.index) == 2
    cache.compact()
    assert len(cache.index) == 1
    assert cache.stats()["dead_slots"] == 0
