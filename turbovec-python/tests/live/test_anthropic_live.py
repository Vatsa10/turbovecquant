"""Live Anthropic smoke tests.

Run with: pytest -m live tests/live/test_anthropic_live.py
Requires: ANTHROPIC_API_KEY.
"""
from __future__ import annotations

import os

import pytest

from .conftest import require_env


pytestmark = pytest.mark.live


@pytest.fixture(scope="module")
def api_key():
    return require_env("ANTHROPIC_API_KEY")


@pytest.fixture(scope="module")
def model():
    return os.environ.get("TURBOVEC_LIVE_ANTHROPIC_MODEL", "claude-haiku-4-5")


def test_anthropic_chat_real(api_key, model):
    from turbovec.llm.llm import AnthropicChat

    llm = AnthropicChat(model=model, retries=2, max_tokens=16)
    result = llm.complete(
        [{"role": "user", "content": "Reply with exactly the word: pong"}],
    )
    assert result.text.strip()
    assert result.tokens_in > 0
    assert result.tokens_out > 0


def test_anthropic_stream_real(api_key, model):
    from turbovec.llm.llm import AnthropicChat

    llm = AnthropicChat(model=model, retries=2, max_tokens=16)
    chunks = list(
        llm.stream([{"role": "user", "content": "Count: 1 2 3. Stop."}])
    )
    assert chunks
    assert "".join(chunks).strip()


def test_anthropic_with_system_prompt_real(api_key, model):
    from turbovec.llm.llm import AnthropicChat

    llm = AnthropicChat(model=model, retries=2, max_tokens=16)
    result = llm.complete([
        {"role": "system", "content": "Always reply with exactly one word."},
        {"role": "user", "content": "Hello?"},
    ])
    assert result.text.strip()
