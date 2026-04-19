"""Optional observability: Prometheus metrics + structured JSON audit log.

All functions are no-ops if the optional deps (``prometheus_client``) are
missing, so importing this module is free. Wire up via
``SemanticCache(metrics=Metrics())``.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Optional


class _NoopMetric:
    def labels(self, *a: Any, **k: Any) -> "_NoopMetric": return self
    def inc(self, *a: Any, **k: Any) -> None: pass
    def observe(self, *a: Any, **k: Any) -> None: pass
    def set(self, *a: Any, **k: Any) -> None: pass


class Metrics:
    """Prometheus metrics facade. No-op when prometheus_client is unavailable."""

    def __init__(self, registry: Optional[Any] = None, prefix: str = "turbovec_llm") -> None:
        try:
            from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry
        except ImportError:
            Counter = Histogram = Gauge = None  # type: ignore
            CollectorRegistry = None
        self._enabled = Counter is not None
        if not self._enabled:
            self.requests = self.hits = self.misses = self.errors = _NoopMetric()
            self.latency = self.tokens_saved = self.dollars_saved = _NoopMetric()
            self.entries = _NoopMetric()
            return
        reg = registry if registry is not None else CollectorRegistry()
        self.registry = reg
        self.requests = Counter(f"{prefix}_requests_total", "Requests", ["tenant", "model"], registry=reg)
        self.hits = Counter(f"{prefix}_hits_total", "Cache hits", ["tenant", "model"], registry=reg)
        self.misses = Counter(f"{prefix}_misses_total", "Cache misses", ["tenant", "model"], registry=reg)
        self.errors = Counter(f"{prefix}_errors_total", "Upstream errors", ["tenant", "model"], registry=reg)
        self.latency = Histogram(f"{prefix}_latency_seconds", "End-to-end latency",
                                 ["tenant", "model", "outcome"], registry=reg)
        self.tokens_saved = Counter(f"{prefix}_tokens_saved_total", "Tokens saved", ["tenant", "model"], registry=reg)
        self.dollars_saved = Counter(f"{prefix}_dollars_saved_total", "USD saved", ["tenant", "model"], registry=reg)
        self.entries = Gauge(f"{prefix}_entries", "Live cache entries", registry=reg)


class AuditLogger:
    """One structured JSON line per cache request."""

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self._log = logger or logging.getLogger("turbovec.llm.audit")

    def emit(self, event: dict[str, Any]) -> None:
        event.setdefault("ts", time.time())
        try:
            self._log.info(json.dumps(event, default=str))
        except Exception:  # pragma: no cover
            pass


__all__ = ["Metrics", "AuditLogger"]
