"""End-to-end tests for native TurboQuantIndex.delete() and its interaction
with SemanticCache (tombstone reuse, stale-payload autoclean)."""
from __future__ import annotations

import numpy as np

from turbovec import TurboQuantIndex
from turbovec.llm import SemanticCache
from turbovec.llm.stores import InMemoryStore

from _llm_fakes import FakeEmbedder, FakeLLM


def _msg(t):
    return [{"role": "user", "content": t}]


def test_index_add_returns_slot_ids():
    idx = TurboQuantIndex(64, 4)
    vecs = np.random.RandomState(0).randn(5, 64).astype(np.float32)
    ids = idx.add(vecs)
    assert ids == [0, 1, 2, 3, 4]
    idx.delete([1, 3])
    more = np.random.RandomState(1).randn(2, 64).astype(np.float32)
    reused = idx.add(more)
    # LIFO reuse — 3 was pushed last, so it's popped first
    assert sorted(reused) == [1, 3]
    assert idx.capacity() == 5
    assert idx.num_deleted() == 0


def test_delete_excludes_from_search():
    idx = TurboQuantIndex(64, 4)
    rs = np.random.RandomState(0)
    vecs = rs.randn(32, 64).astype(np.float32)
    idx.add(vecs)
    q = vecs[0:1].copy()
    scores, inds = idx.search(q, k=3)
    assert 0 in inds[0].tolist()
    idx.delete([0])
    scores2, inds2 = idx.search(q, k=3)
    assert 0 not in inds2[0].tolist()
    assert len(inds2[0]) == 3


def test_stale_payload_autocleans_index():
    """Store evicted an entry (Redis TTL) but the index still holds the vector.
    The cache must tombstone that slot on first post-eviction lookup."""
    llm = FakeLLM()
    store = InMemoryStore()
    cache = SemanticCache(embedder=FakeEmbedder(), llm=llm, similarity_threshold=0.9, store=store)
    cache.complete(_msg("ping"))
    # Simulate TTL eviction in the payload store.
    for k in list(store.keys()):
        store.delete(k)
    # Now the index has a live slot with no payload. A near-duplicate
    # lookup finds it, store.get returns None, cache tombstones it,
    # falls through to upstream, and the miss-insert reuses the freed
    # slot. End result: upstream called, but capacity stays at 1.
    r = cache.complete(_msg("ping"))
    assert r.text.startswith("fake-reply")
    assert llm.calls == 2
    assert cache.index.capacity() == 1
    assert cache.stats()["tombstones"] == 0


def test_deletion_survives_save_load(tmp_path):
    path = tmp_path / "cache"
    llm = FakeLLM()
    cache = SemanticCache(embedder=FakeEmbedder(), llm=llm, similarity_threshold=0.9, path=path)
    cache.complete(_msg("alpha"))
    cache.complete(_msg("beta"))
    cache.invalidate(predicate=lambda e: "alpha" in e["messages"][-1]["content"])
    cache.save()

    llm2 = FakeLLM()
    cache2 = SemanticCache(embedder=FakeEmbedder(), llm=llm2, similarity_threshold=0.9, path=path)
    # Tombstone survived — alpha is gone, beta hits.
    assert cache2.index.num_deleted() == 1
    cache2.complete(_msg("beta"))
    assert llm2.calls == 0   # cache hit
    cache2.complete(_msg("alpha"))
    assert llm2.calls == 1   # miss (alpha was deleted)
