"""Shared pytest fixtures and config.

* Loads ``.env`` from the repo root so live tests pick up API keys
  without the user having to `export` them manually.
* Registers the ``live`` marker. Live tests are excluded from the default
  run via the ``-m "not live"`` default in ``pyproject.toml``; invoke
  ``pytest -m live`` explicitly to run them.
"""
from __future__ import annotations

import os
from pathlib import Path


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        env = parent / ".env"
        if env.exists():
            load_dotenv(env, override=False)
            break


_load_dotenv()


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "live: requires a live API key / provider. Skipped by default; run with `-m live`.",
    )


def env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)
