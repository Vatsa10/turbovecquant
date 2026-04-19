"""PII redaction hook for the semantic cache.

A redactor is ``Callable[[str], str]``. Applied to the cache key text *and* the
cached response text before it hits the payload store. Default implementation
catches email, US SSN, credit-card-like 16-digit blocks, and AWS access keys.
Users can provide their own redactor (e.g., a Presidio-based one).
"""
from __future__ import annotations

import re
from typing import Callable


_PATTERNS = [
    (re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"), "[EMAIL]"),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[SSN]"),
    (re.compile(r"\b(?:\d[ -]*?){13,16}\b"), "[CARD]"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "[AWS_KEY]"),
    (re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"), "[API_KEY]"),
    (re.compile(r"\bBearer\s+[A-Za-z0-9\-._~+/]+=*", re.IGNORECASE), "Bearer [TOKEN]"),
]


def default_redactor(text: str) -> str:
    for pat, repl in _PATTERNS:
        text = pat.sub(repl, text)
    return text


Redactor = Callable[[str], str]


__all__ = ["default_redactor", "Redactor"]
