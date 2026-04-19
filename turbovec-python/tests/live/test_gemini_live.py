"""Live Gemini smoke tests.

Run with: pytest -m live tests/live/test_gemini_live.py
Requires: GOOGLE_API_KEY (or GEMINI_API_KEY).
"""
from __future__ import annotations

import os

import pytest

from .conftest import require_any_env


pytestmark = pytest.mark.live


@pytest.fixture(scope="module")
def api_key():
    return require_any_env("GOOGLE_API_KEY", "GEMINI_API_KEY")


@pytest.fixture(scope="module")
def model():
    return os.environ.get("TURBOVEC_LIVE_GEMINI_MODEL", "gemini-2.0-flash")


def test_gemini_chat_real(api_key, model):
    from turbovec.llm.llm import GeminiChat

    llm = GeminiChat(model=model, api_key=api_key, retries=2)
    result = llm.complete(
        [{"role": "user", "content": "Reply with exactly the word: pong"}],
    )
    assert result.text.strip()


def test_gemini_stream_real(api_key, model):
    from turbovec.llm.llm import GeminiChat

    llm = GeminiChat(model=model, api_key=api_key, retries=2)
    chunks = list(
        llm.stream([{"role": "user", "content": "Count: 1 2 3. Stop."}])
    )
    assert chunks
    assert "".join(chunks).strip()
