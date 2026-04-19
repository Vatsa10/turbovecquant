from __future__ import annotations

import pytest

from turbovec.llm.stores import InMemoryStore, PickleStore


@pytest.mark.parametrize("factory", [lambda tmp: InMemoryStore(), lambda tmp: PickleStore(tmp / "s.pkl")])
def test_store_basic(factory, tmp_path):
    s = factory(tmp_path)
    assert s.get(0) is None
    s.set(0, {"a": 1, "hits": 0})
    assert s.get(0)["a"] == 1
    s.incr_hits(0)
    assert s.get(0)["hits"] == 1
    assert 0 in list(s.keys())
    assert len(s) == 1
    s.delete(0)
    assert s.get(0) is None
    assert len(s) == 0


def test_pickle_store_roundtrip(tmp_path):
    path = tmp_path / "s.pkl"
    s = PickleStore(path)
    s.set(0, {"x": "hello"})
    s.set(1, {"x": "world"})
    s.save()

    s2 = PickleStore(path)
    assert s2.get(0) == {"x": "hello"}
    assert s2.get(1) == {"x": "world"}
    assert len(s2) == 2


def _redis_client():
    redis = pytest.importorskip("redis")
    try:
        c = redis.Redis.from_url("redis://localhost:6379/0", decode_responses=True)
        c.ping()
        return c
    except Exception:
        pytest.skip("Redis/Memurai not available on localhost:6379")


def test_redis_store(tmp_path):
    from turbovec.llm.stores import RedisStore

    client = _redis_client()
    ns = f"test:{tmp_path.name}:"
    store = RedisStore(client=client, namespace=ns)
    try:
        store.set(0, {"a": 1})
        store.set(1, {"a": 2})
        assert store.get(0)["a"] == 1
        store.incr_hits(0)
        store.incr_hits(0)
        assert store.get(0)["hits"] == 2
        assert sorted(store.keys()) == [0, 1]
        store.delete(0)
        assert store.get(0) is None
        assert list(store.keys()) == [1]
    finally:
        for k in client.scan_iter(match=f"{ns}*"):
            client.delete(k)


def test_redis_store_ttl(tmp_path):
    import time as _t
    from turbovec.llm.stores import RedisStore

    client = _redis_client()
    ns = f"test:ttl:{tmp_path.name}:"
    store = RedisStore(client=client, namespace=ns, ttl_seconds=1, touch_on_read=False)
    try:
        store.set(0, {"a": 1})
        assert store.get(0)["a"] == 1
        _t.sleep(1.2)
        assert store.get(0) is None
    finally:
        for k in client.scan_iter(match=f"{ns}*"):
            client.delete(k)
