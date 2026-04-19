"""Shared fakes for turbovec.llm tests — no network, no heavy deps."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from turbovec.llm.llm import CompletionResult


class FakeEmbedder:
    """Deterministic text → dense vector via hashing, then L2-normalize.

    Similar strings produce similar vectors because we inject shared character-
    n-gram signal into the hash bucket. Good enough to exercise the threshold.
    """

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            tokens = t.lower().split()
            for tok in tokens:
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                out[i, h % self.dim] += 1.0
                if len(tok) > 2:
                    h2 = int(hashlib.md5(tok[:3].encode()).hexdigest(), 16)
                    out[i, h2 % self.dim] += 0.5
            if not tokens:
                out[i, 0] = 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms


@dataclass
class FakeLLM:
    reply: str = "fake-reply"
    calls: int = 0
    tokens_in: int = 10
    tokens_out: int = 20
    model: str = "gpt-4o-mini"
    fail_times: int = 0

    def complete(self, messages, **_):
        self.calls += 1
        if self.fail_times > 0:
            self.fail_times -= 1
            raise RuntimeError("upstream kaboom")
        last = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        return CompletionResult(
            text=f"{self.reply}:{last}",
            tokens_in=self.tokens_in,
            tokens_out=self.tokens_out,
            model=self.model,
        )

    async def acomplete(self, messages, **kw):
        return self.complete(messages, **kw)

    def stream(self, messages, **_):
        self.calls += 1
        last = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        text = f"{self.reply}:{last}"
        for i in range(0, len(text), 3):
            yield text[i:i + 3]

    async def astream(self, messages, **kw):
        self.calls += 1
        last = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        text = f"{self.reply}:{last}"
        for i in range(0, len(text), 3):
            yield text[i:i + 3]
