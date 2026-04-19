from __future__ import annotations

import pytest

from turbovec.llm import ConversationMemory
from turbovec.llm.memory import InMemoryHistoryBackend

from _llm_fakes import FakeEmbedder


def test_append_and_history():
    mem = ConversationMemory()
    mem.append("s1", {"role": "user", "content": "hello"})
    mem.append("s1", {"role": "assistant", "content": "hi"})
    mem.append("s2", {"role": "user", "content": "other"})
    h1 = mem.history("s1")
    assert [m["content"] for m in h1] == ["hello", "hi"]
    assert mem.history("s2") == [{"role": "user", "content": "other"}]
    assert mem.history("s1", last_n=1) == [{"role": "assistant", "content": "hi"}]


def test_clear():
    mem = ConversationMemory()
    mem.append("s1", {"role": "user", "content": "hello"})
    mem.clear("s1")
    assert mem.history("s1") == []


def test_recall_scoped_to_session():
    mem = ConversationMemory(embedder=FakeEmbedder())
    mem.append("a", {"role": "user", "content": "paris france capital"})
    mem.append("a", {"role": "user", "content": "bananas tropical fruit"})
    mem.append("b", {"role": "user", "content": "paris france capital"})
    hits = mem.recall("a", "paris", k=1)
    assert len(hits) == 1
    assert "paris" in hits[0]["content"]


def test_recall_requires_embedder():
    mem = ConversationMemory()
    with pytest.raises(RuntimeError):
        mem.recall("s1", "query")


def test_redis_history(tmp_path):
    redis = pytest.importorskip("redis")
    try:
        client = redis.Redis.from_url("redis://localhost:6379/0", decode_responses=True)
        client.ping()
    except Exception:
        pytest.skip("Redis/Memurai not available on localhost:6379")

    from turbovec.llm.memory import RedisHistoryBackend

    ns = f"tv:test:mem:{tmp_path.name}:"
    backend = RedisHistoryBackend(client=client, namespace=ns, ttl_seconds=None)
    mem = ConversationMemory(backend=backend)
    try:
        mem.append("s1", {"role": "user", "content": "hello"})
        mem.append("s1", {"role": "assistant", "content": "hi"})
        assert [m["content"] for m in mem.history("s1")] == ["hello", "hi"]
        assert [m["content"] for m in mem.history("s1", last_n=1)] == ["hi"]
        mem.clear("s1")
        assert mem.history("s1") == []
    finally:
        for k in client.scan_iter(match=f"{ns}*"):
            client.delete(k)
