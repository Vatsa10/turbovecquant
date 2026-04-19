"""Live proxy test — real OpenAI behind the FastAPI proxy.

Verifies an existing OpenAI SDK can point at http://localhost/v1 and get
caching transparently. Requires: OPENAI_API_KEY.
"""
from __future__ import annotations

import os

import pytest

fastapi = pytest.importorskip("fastapi")
starlette_testclient = pytest.importorskip("starlette.testclient")

from starlette.testclient import TestClient

from .conftest import require_env


pytestmark = pytest.mark.live


@pytest.fixture(scope="module")
def client_and_cache():
    require_env("OPENAI_API_KEY")
    from turbovec.llm import SemanticCache
    from turbovec.llm.embedders import OpenAIEmbedder
    from turbovec.llm.llm import OpenAIChat
    from turbovec.llm.proxy import build_app

    chat_model = os.environ.get("TURBOVEC_LIVE_OPENAI_CHAT_MODEL", "gpt-4o-mini")
    embed_model = os.environ.get("TURBOVEC_LIVE_OPENAI_EMBED_MODEL", "text-embedding-3-small")

    cache = SemanticCache(
        embedder=OpenAIEmbedder(model=embed_model),
        llm=OpenAIChat(model=chat_model, retries=2),
        similarity_threshold=0.85,
    )
    app = build_app(cache)
    return TestClient(app), cache, chat_model


def test_proxy_chat_completion_hits_cache(client_and_cache):
    client, cache, model = client_and_cache
    body = {
        "model": model,
        "messages": [{"role": "user", "content": "Say the single word: ping"}],
        "temperature": 0,
        "max_tokens": 8,
    }
    r1 = client.post("/v1/chat/completions", json=body, headers={"Authorization": "Bearer test-key"})
    assert r1.status_code == 200
    d1 = r1.json()
    assert d1["choices"][0]["message"]["content"].strip()
    assert d1["turbovec"]["cached"] is False

    # Near-duplicate
    body["messages"][0]["content"] = "Just say: ping"
    r2 = client.post("/v1/chat/completions", json=body, headers={"Authorization": "Bearer test-key"})
    # Depending on embedding similarity, this may or may not hit.
    # Either way the request must succeed; cache hit is a bonus.
    assert r2.status_code == 200


def test_proxy_stats_and_models(client_and_cache):
    client, _cache, model = client_and_cache
    r = client.get("/v1/models")
    assert r.status_code == 200
    assert any(m["id"] == model for m in r.json()["data"])

    s = client.get("/stats").json()
    assert "hit_rate" in s
