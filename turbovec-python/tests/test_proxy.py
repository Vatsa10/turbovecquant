from __future__ import annotations

import json

import pytest

fastapi = pytest.importorskip("fastapi")
starlette_testclient = pytest.importorskip("starlette.testclient")

from starlette.testclient import TestClient

from turbovec.llm import SemanticCache
from turbovec.llm.proxy import build_app

from _llm_fakes import FakeEmbedder, FakeLLM


def _make_client(**cache_kw):
    llm = FakeLLM()
    cache = SemanticCache(embedder=FakeEmbedder(), llm=llm, similarity_threshold=0.9, **cache_kw)
    app = build_app(cache)
    return TestClient(app), cache, llm


def test_chat_completions_hit_miss():
    client, cache, llm = _make_client()
    body = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hello"}]}
    r1 = client.post("/v1/chat/completions", json=body)
    assert r1.status_code == 200
    d1 = r1.json()
    assert d1["choices"][0]["message"]["content"].startswith("fake-reply:hello")
    assert d1["turbovec"]["cached"] is False

    r2 = client.post("/v1/chat/completions", json=body)
    assert r2.json()["turbovec"]["cached"] is True
    assert llm.calls == 1


def test_chat_completions_streaming():
    client, cache, llm = _make_client()
    body = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "stream me"}], "stream": True}
    with client.stream("POST", "/v1/chat/completions", json=body) as r:
        assert r.status_code == 200
        events = []
        for line in r.iter_lines():
            if line.startswith("data: "):
                payload = line[6:]
                if payload == "[DONE]":
                    break
                events.append(json.loads(payload))
    content = "".join(e["choices"][0]["delta"].get("content", "") for e in events)
    assert content.startswith("fake-reply:stream me")


def test_tenants_isolated_by_bearer():
    client, cache, llm = _make_client()
    body = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "same"}]}
    client.post("/v1/chat/completions", json=body, headers={"Authorization": "Bearer key-alice"})
    client.post("/v1/chat/completions", json=body, headers={"Authorization": "Bearer key-bob"})
    assert llm.calls == 2  # different tenants
    client.post("/v1/chat/completions", json=body, headers={"Authorization": "Bearer key-alice"})
    assert llm.calls == 2  # cache hit


def test_stats_and_invalidate():
    client, cache, llm = _make_client()
    body = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]}
    client.post("/v1/chat/completions", json=body, headers={"Authorization": "Bearer k1"})
    s = client.get("/stats").json()
    assert s["misses"] == 1
    r = client.post("/admin/invalidate", json={})
    assert r.json()["removed"] >= 1
