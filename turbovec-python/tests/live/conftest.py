"""Shared helpers for live (real-API) tests."""
from __future__ import annotations

import os

import pytest


def require_env(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        pytest.skip(f"{name} not set — live test skipped")
    return val


def require_any_env(*names: str) -> str:
    for n in names:
        v = os.environ.get(n)
        if v:
            return v
    pytest.skip(f"none of {names} set — live test skipped")
