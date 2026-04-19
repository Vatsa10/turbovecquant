"""Live OpenAI smoke tests.

Run with: pytest -m live tests/live/test_openai_live.py
Requires: OPENAI_API_KEY in the environment (or .env at repo root).
Cost: ~1-3 embedding calls + 2 short chat completions per full run (< $0.001).
"""
from __future__ import annotations

import os

import pytest

from .conftest import require_env


pytestmark = pytest.mark.live


@pytest.fixture(scope="module")
def api_key():
    return require_env("OPENAI_API_KEY")


@pytest.fixture(scope="module")
def chat_model():
    return os.environ.get("TURBOVEC_LIVE_OPENAI_CHAT_MODEL", "gpt-4o-mini")


@pytest.fixture(scope="module")
def embed_model():
    return os.environ.get("TURBOVEC_LIVE_OPENAI_EMBED_MODEL", "text-embedding-3-small")


def test_openai_embedder_real(api_key, embed_model):
    from turbovec.llm.embedders import OpenAIEmbedder

    emb = OpenAIEmbedder(model=embed_model)
    vecs = emb.embed(["hello world", "goodbye world"])
    assert vecs.shape == (2, emb.dim)
    # Output is L2-normalized per embedder contract.
    norms = (vecs ** 2).sum(axis=1) ** 0.5
    assert all(abs(n - 1.0) < 1e-3 for n in norms.tolist())


def test_openai_chat_real(api_key, chat_model):
    from turbovec.llm.llm import OpenAIChat

    llm = OpenAIChat(model=chat_model, retries=2)
    result = llm.complete(
        [{"role": "user", "content": "Reply with exactly the word: pong"}],
        temperature=0,
        max_tokens=8,
    )
    assert result.text.strip()
    assert result.tokens_in > 0
    assert result.tokens_out > 0


def test_semantic_cache_hit_real(api_key, chat_model, embed_model):
    """Full round-trip: miss, then near-duplicate hits the cache (no upstream call)."""
    from turbovec.llm import SemanticCache
    from turbovec.llm.embedders import OpenAIEmbedder
    from turbovec.llm.llm import OpenAIChat

    cache = SemanticCache(
        embedder=OpenAIEmbedder(model=embed_model),
        llm=OpenAIChat(model=chat_model, retries=2),
        similarity_threshold=0.85,
    )

    # Miss — hits the real API.
    r1 = cache.complete(
        [{"role": "user", "content": "What is the capital of France? One word."}],
        temperature=0, max_tokens=8,
    )
    assert r1.tokens_in > 0

    # Near-duplicate — must cache-hit, zero upstream tokens billed.
    r2 = cache.complete(
        [{"role": "user", "content": "What's the capital of France? Just one word."}],
        temperature=0, max_tokens=8,
    )
    assert r2.tokens_in == 0
    assert r2.tokens_out == 0
    stats = cache.stats()
    assert stats["hits"] == 1
    assert stats["dollars_saved"] > 0


def test_openai_stream_real(api_key, chat_model):
    from turbovec.llm.llm import OpenAIChat

    llm = OpenAIChat(model=chat_model, retries=2)
    chunks = list(
        llm.stream(
            [{"role": "user", "content": "Count: 1 2 3. Stop."}],
            temperature=0, max_tokens=16,
        )
    )
    assert chunks
    assert "".join(chunks).strip()
