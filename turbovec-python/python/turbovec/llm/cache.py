"""Production-grade semantic response cache for hosted LLM APIs.

Features:
  * Sync + async completion, with optional cache-through streaming.
  * Temperature / tool-call gating (non-deterministic calls bypass the cache).
  * Multi-tenant scoping via a ``tenant`` argument per request.
  * PII redaction hook applied to both cache-key text and cached response text.
  * Negative caching — upstream errors cached briefly to absorb retry storms.
  * Cost tracking — ``stats()`` reports tokens and USD saved per model.
  * LFU-aware invalidation (``invalidate``, ``compact``).
  * Prometheus metrics + structured audit log (opt-in, no-op without deps).

Storage split:
  * In-process :class:`TurboQuantIndex` — the quantized ANN.
  * Pluggable :class:`PayloadStore` — in-memory / pickle / Redis.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Iterator, Optional

import numpy as np

from .._turbovec import TurboQuantIndex
from .embedders import Embedder
from .gating import GateDecision, should_cache
from .llm import LLM, CompletionResult
from .observability import AuditLogger, Metrics
from .pricing import DEFAULT_PRICING, estimate_cost_usd
from .redaction import Redactor, default_redactor
from .stores import InMemoryStore, PayloadStore, PickleStore


logger = logging.getLogger(__name__)


def _last_user(messages: list[dict]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            return m.get("content", "")
    return ""


def _system_fingerprint(messages: list[dict]) -> str:
    sys_text = "\n\n".join(m.get("content", "") for m in messages if m.get("role") == "system")
    return hashlib.sha256(sys_text.encode("utf-8")).hexdigest()[:16]


class SemanticCache:
    """Cosine-match prior prompts and reuse cached completions.

    :param embedder: produces L2-normalized vectors for cache keys.
    :param llm: upstream provider adapter (any object with ``complete``).
    :param bit_width: quantization bit-width (2 / 3 / 4).
    :param similarity_threshold: min cosine sim to accept a hit.
    :param store: pluggable payload store; defaults to :class:`PickleStore` if
        ``path`` is given, else :class:`InMemoryStore`.
    :param path: disk location (``.tv`` + ``.pkl`` split); auto-loaded if exists.
    :param key_fn: extract the cache-key text from messages. Defaults to the
        most recent user message.
    :param redactor: applied to cache-key text and cached response text.
        Pass ``None`` to disable. Defaults to :func:`default_redactor`.
    :param cache_nondeterministic: allow caching of ``temperature > 0`` calls.
    :param allow_tools: allow caching of calls that include tool definitions or
        tool-call messages.
    :param negative_cache_seconds: if > 0, upstream exceptions get a short-lived
        entry so retry bursts don't hammer the provider.
    :param pricing: override the built-in ``model → (in_per_1k, out_per_1k)`` table.
    :param metrics: ``Metrics()`` facade for Prometheus instrumentation.
    :param audit: ``AuditLogger()`` for one-JSON-line-per-request logs.
    """

    def __init__(
        self,
        *,
        embedder: Embedder,
        llm: LLM,
        bit_width: int = 4,
        similarity_threshold: float = 0.93,
        store: Optional[PayloadStore] = None,
        path: Optional[str | Path] = None,
        key_fn: Optional[Callable[[list[dict]], str]] = None,
        redactor: Optional[Redactor] = default_redactor,
        cache_nondeterministic: bool = False,
        allow_tools: bool = False,
        negative_cache_seconds: float = 0.0,
        pricing: Optional[dict[str, tuple[float, float]]] = None,
        metrics: Optional[Metrics] = None,
        audit: Optional[AuditLogger] = None,
    ) -> None:
        self._embedder = embedder
        self._llm = llm
        self._threshold = similarity_threshold
        self._key_fn = key_fn or _last_user
        self._redact = redactor or (lambda s: s)
        self._cache_nondet = cache_nondeterministic
        self._allow_tools = allow_tools
        self._neg_ttl = float(negative_cache_seconds)
        self._pricing = pricing if pricing is not None else DEFAULT_PRICING
        self._metrics = metrics or Metrics()
        self._audit = audit
        self._path = Path(path) if path else None

        dim = embedder.dim
        index_path = self._path.with_suffix(".tv") if self._path else None
        if index_path and index_path.exists():
            self._index = TurboQuantIndex.load(str(index_path))
        else:
            self._index = TurboQuantIndex(dim, bit_width)

        if store is not None:
            self._store = store
        elif self._path is not None:
            self._store = PickleStore(self._path.with_suffix(".pkl"))
        else:
            self._store = InMemoryStore()

        self._hits = 0
        self._misses = 0
        self._bypassed = 0
        self._errors = 0
        self._tokens_saved_in = 0
        self._tokens_saved_out = 0
        self._dollars_saved = 0.0
        # Native tombstones in the Rust index handle deletion; Python-
        # side bookkeeping is only needed for stale payload detection
        # after a store-level TTL eviction (store miss on an index hit
        # that isn't yet reflected in the index).
        self._pending_dead: set[int] = set()

    # ---------------------------------------------------------- public props
    @property
    def index(self) -> TurboQuantIndex:
        return self._index

    @property
    def store(self) -> PayloadStore:
        return self._store

    # ---------------------------------------------------------- key helpers
    def _embed_key(self, messages: list[dict]) -> tuple[np.ndarray, str, str]:
        key_text = self._redact(self._key_fn(messages))
        sys_fp = _system_fingerprint(messages)
        qvec = self._embedder.embed([key_text])
        return qvec, sys_fp, key_text

    def _gate(self, messages: list[dict], kwargs: dict, force_cache: bool) -> GateDecision:
        if force_cache:
            return GateDecision(True, "forced")
        return should_cache(
            messages, kwargs,
            cache_nondeterministic=self._cache_nondet,
            allow_tools=self._allow_tools,
        )

    # ---------------------------------------------------------- sync complete
    def complete(
        self,
        messages: list[dict],
        *,
        tenant: Optional[str] = None,
        force_cache: bool = False,
        **llm_kwargs: Any,
    ) -> CompletionResult:
        model = getattr(self._llm, "model", "unknown")
        self._metrics.requests.labels(tenant or "_", model).inc()
        t0 = time.perf_counter()

        gate = self._gate(messages, llm_kwargs, force_cache)
        if not gate.cache:
            self._bypassed += 1
            self._audit_emit("bypass", tenant, model, gate.reason)
            result = self._llm.complete(messages, **llm_kwargs)
            self._metrics.latency.labels(tenant or "_", model, "bypass").observe(time.perf_counter() - t0)
            return result

        qvec, sys_fp, key_text = self._embed_key(messages)

        hit = self._lookup(qvec, sys_fp, tenant)
        if hit is not None:
            self._record_hit(hit, tenant, model, t0)
            return CompletionResult(
                text=hit["response"], tokens_in=0, tokens_out=0,
                model=hit.get("model", model), raw={"cached": True, "entry": hit},
            )

        try:
            result = self._llm.complete(messages, **llm_kwargs)
        except Exception as exc:
            return self._record_error(exc, qvec, sys_fp, key_text, messages, tenant, model, t0)

        self._record_miss(qvec, sys_fp, key_text, messages, result, tenant, t0)
        return result

    # ---------------------------------------------------------- async complete
    async def acomplete(
        self,
        messages: list[dict],
        *,
        tenant: Optional[str] = None,
        force_cache: bool = False,
        **llm_kwargs: Any,
    ) -> CompletionResult:
        model = getattr(self._llm, "model", "unknown")
        self._metrics.requests.labels(tenant or "_", model).inc()
        t0 = time.perf_counter()

        gate = self._gate(messages, llm_kwargs, force_cache)
        if not gate.cache:
            self._bypassed += 1
            result = await self._llm.acomplete(messages, **llm_kwargs)  # type: ignore[attr-defined]
            self._metrics.latency.labels(tenant or "_", model, "bypass").observe(time.perf_counter() - t0)
            return result

        qvec, sys_fp, key_text = self._embed_key(messages)
        hit = self._lookup(qvec, sys_fp, tenant)
        if hit is not None:
            self._record_hit(hit, tenant, model, t0)
            return CompletionResult(
                text=hit["response"], tokens_in=0, tokens_out=0,
                model=hit.get("model", model), raw={"cached": True, "entry": hit},
            )

        try:
            result = await self._llm.acomplete(messages, **llm_kwargs)  # type: ignore[attr-defined]
        except Exception as exc:
            return self._record_error(exc, qvec, sys_fp, key_text, messages, tenant, model, t0)

        self._record_miss(qvec, sys_fp, key_text, messages, result, tenant, t0)
        return result

    # ---------------------------------------------------------- streaming
    def stream_complete(
        self,
        messages: list[dict],
        *,
        tenant: Optional[str] = None,
        force_cache: bool = False,
        chunk_size: int = 64,
        **llm_kwargs: Any,
    ) -> Iterator[str]:
        model = getattr(self._llm, "model", "unknown")
        self._metrics.requests.labels(tenant or "_", model).inc()
        t0 = time.perf_counter()

        gate = self._gate(messages, llm_kwargs, force_cache)
        if gate.cache:
            qvec, sys_fp, key_text = self._embed_key(messages)
            hit = self._lookup(qvec, sys_fp, tenant)
            if hit is not None:
                self._record_hit(hit, tenant, model, t0)
                text = hit["response"]
                for i in range(0, len(text), chunk_size):
                    yield text[i:i + chunk_size]
                return
        else:
            self._bypassed += 1

        buf: list[str] = []
        try:
            for chunk in self._llm.stream(messages, **llm_kwargs):  # type: ignore[attr-defined]
                buf.append(chunk)
                yield chunk
        except Exception as exc:
            self._errors += 1
            self._audit_emit("error", tenant, model, repr(exc))
            raise

        if gate.cache:
            full = "".join(buf)
            result = CompletionResult(text=full, model=model)
            self._record_miss(qvec, sys_fp, key_text, messages, result, tenant, t0)

    async def astream_complete(
        self,
        messages: list[dict],
        *,
        tenant: Optional[str] = None,
        force_cache: bool = False,
        chunk_size: int = 64,
        **llm_kwargs: Any,
    ) -> AsyncIterator[str]:
        model = getattr(self._llm, "model", "unknown")
        self._metrics.requests.labels(tenant or "_", model).inc()
        t0 = time.perf_counter()

        gate = self._gate(messages, llm_kwargs, force_cache)
        if gate.cache:
            qvec, sys_fp, key_text = self._embed_key(messages)
            hit = self._lookup(qvec, sys_fp, tenant)
            if hit is not None:
                self._record_hit(hit, tenant, model, t0)
                text = hit["response"]
                for i in range(0, len(text), chunk_size):
                    yield text[i:i + chunk_size]
                    await asyncio.sleep(0)
                return
        else:
            self._bypassed += 1

        buf: list[str] = []
        try:
            async for chunk in self._llm.astream(messages, **llm_kwargs):  # type: ignore[attr-defined]
                buf.append(chunk)
                yield chunk
        except Exception as exc:
            self._errors += 1
            self._audit_emit("error", tenant, model, repr(exc))
            raise

        if gate.cache:
            full = "".join(buf)
            result = CompletionResult(text=full, model=model)
            self._record_miss(qvec, sys_fp, key_text, messages, result, tenant, t0)

    # ---------------------------------------------------------- internals
    def _lookup(self, qvec: np.ndarray, sys_fp: str, tenant: Optional[str]) -> Optional[dict]:
        if len(self._index) == 0:
            return None
        k = min(10, len(self._index))
        scores, indices = self._index.search(qvec, k)
        for score, idx in zip(scores[0], indices[0]):
            idx_i = int(idx)
            if idx_i < 0:
                continue
            if float(score) < self._threshold:
                return None
            entry = self._store.get(idx_i)
            if entry is None:
                # Payload was evicted (e.g., Redis TTL) but the vector is
                # still live in the index — tombstone it so the slot can
                # be reused on the next miss.
                self._index.delete([idx_i])
                continue
            if entry.get("system_fp") != sys_fp:
                continue
            if entry.get("tenant") != tenant:
                continue
            if entry.get("kind") == "error":
                expires = float(entry.get("expires_at", 0))
                if expires and time.time() > expires:
                    self._store.delete(idx_i)
                    self._index.delete([idx_i])
                    continue
                raise RuntimeError(f"negative-cached upstream error: {entry.get('error', 'unknown')}")
            self._store.incr_hits(idx_i)
            return entry
        return None

    def _insert(
        self,
        qvec: np.ndarray,
        sys_fp: str,
        key_text: str,
        messages: list[dict],
        *,
        tenant: Optional[str],
        response: str,
        tokens_in: int,
        tokens_out: int,
        model: str,
        kind: str = "ok",
        expires_at: float = 0.0,
        error: str = "",
    ) -> int:
        # `index.add` returns the slot ids it wrote to. With native
        # tombstones this might be a reused slot rather than a new one
        # at the tail, so we cannot use `len(index)` as the key.
        new_ids = self._index.add(qvec)
        new_id = int(new_ids[0])
        prompt_hash = hashlib.sha256(key_text.encode("utf-8")).hexdigest()[:16]
        # If this slot previously held an entry (payload evicted,
        # tombstone reused), clear any stale payload under the same id.
        self._store.delete(new_id)
        self._store.set(
            new_id,
            {
                "kind": kind,
                "tenant": tenant,
                "prompt_hash": prompt_hash,
                "system_fp": sys_fp,
                "messages": messages,
                "response": self._redact(response),
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "model": model,
                "created_at": time.time(),
                "expires_at": expires_at,
                "error": error,
                "hits": 0,
            },
        )
        return new_id

    def _record_hit(self, hit: dict, tenant: Optional[str], model: str, t0: float) -> None:
        self._hits += 1
        tin, tout = int(hit.get("tokens_in", 0)), int(hit.get("tokens_out", 0))
        hit_model = hit.get("model", model)
        self._tokens_saved_in += tin
        self._tokens_saved_out += tout
        dollars = estimate_cost_usd(hit_model, tin, tout, self._pricing)
        self._dollars_saved += dollars
        self._metrics.hits.labels(tenant or "_", hit_model).inc()
        self._metrics.tokens_saved.labels(tenant or "_", hit_model).inc(tin + tout)
        self._metrics.dollars_saved.labels(tenant or "_", hit_model).inc(dollars)
        self._metrics.latency.labels(tenant or "_", hit_model, "hit").observe(time.perf_counter() - t0)
        self._audit_emit("hit", tenant, hit_model, "", tokens_saved=tin + tout, dollars=dollars)

    def _record_miss(
        self,
        qvec: np.ndarray, sys_fp: str, key_text: str, messages: list[dict],
        result: CompletionResult, tenant: Optional[str], t0: float,
    ) -> None:
        self._misses += 1
        model = result.model or getattr(self._llm, "model", "unknown")
        self._insert(
            qvec, sys_fp, key_text, messages,
            tenant=tenant, response=result.text,
            tokens_in=result.tokens_in, tokens_out=result.tokens_out, model=model,
        )
        self._metrics.misses.labels(tenant or "_", model).inc()
        self._metrics.latency.labels(tenant or "_", model, "miss").observe(time.perf_counter() - t0)
        self._metrics.entries.set(len(self._store))
        self._audit_emit("miss", tenant, model, "", tokens=(result.tokens_in + result.tokens_out))

    def _record_error(
        self,
        exc: Exception,
        qvec: np.ndarray, sys_fp: str, key_text: str, messages: list[dict],
        tenant: Optional[str], model: str, t0: float,
    ) -> CompletionResult:
        self._errors += 1
        self._metrics.errors.labels(tenant or "_", model).inc()
        self._metrics.latency.labels(tenant or "_", model, "error").observe(time.perf_counter() - t0)
        self._audit_emit("error", tenant, model, repr(exc))
        if self._neg_ttl > 0:
            self._insert(
                qvec, sys_fp, key_text, messages,
                tenant=tenant, response="", tokens_in=0, tokens_out=0, model=model,
                kind="error", expires_at=time.time() + self._neg_ttl, error=repr(exc),
            )
        raise exc

    def _audit_emit(self, event: str, tenant: Optional[str], model: str, reason: str, **extra: Any) -> None:
        if self._audit is None:
            return
        self._audit.emit({"event": event, "tenant": tenant, "model": model, "reason": reason, **extra})

    # ---------------------------------------------------------- admin API
    def stats(self) -> dict[str, Any]:
        total = self._hits + self._misses
        num_deleted = getattr(self._index, "num_deleted", lambda: 0)()
        capacity = getattr(self._index, "capacity", lambda: len(self._index))()
        return {
            "hits": self._hits,
            "misses": self._misses,
            "bypassed": self._bypassed,
            "errors": self._errors,
            "hit_rate": (self._hits / total) if total else 0.0,
            "tokens_saved_in": self._tokens_saved_in,
            "tokens_saved_out": self._tokens_saved_out,
            "tokens_saved": self._tokens_saved_in + self._tokens_saved_out,
            "dollars_saved": round(self._dollars_saved, 6),
            "entries": len(self._store),
            "index_live": len(self._index),
            "index_capacity": capacity,
            "tombstones": num_deleted,
        }

    def invalidate(
        self,
        *,
        tenant: Optional[str] = None,
        older_than_seconds: Optional[float] = None,
        predicate: Optional[Callable[[dict], bool]] = None,
    ) -> int:
        """Delete matching entries. Returns the number of entries removed.

        This calls native `TurboQuantIndex.delete` on each matching slot,
        so the id becomes available for reuse on the next insert — no
        `compact()` call is needed.
        """
        now = time.time()
        victims: list[int] = []
        for k in list(self._store.keys()):
            e = self._store.get(k)
            if e is None:
                continue
            if tenant is not None and e.get("tenant") != tenant:
                continue
            if older_than_seconds is not None and now - float(e.get("created_at", now)) < older_than_seconds:
                continue
            if predicate is not None and not predicate(e):
                continue
            self._store.delete(k)
            victims.append(k)
        if victims:
            self._index.delete(victims)
        return len(victims)

    def save(self) -> None:
        if self._path is None:
            raise ValueError("SemanticCache was constructed without a path; nothing to save")
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._index.write(str(self._path.with_suffix(".tv")))
        if isinstance(self._store, PickleStore):
            self._store.save()

    def compact(self) -> None:
        """Rebuild the index from live store entries from scratch.

        Native tombstones + free-list reuse make this largely optional —
        slots are reclaimed as new entries arrive. Call ``compact()``
        when you want a defragmented index (e.g., after dropping a huge
        tenant) or to drop ``kind == "error"`` negative-cache entries.
        """
        live_entries = []
        for k in self._store.keys():
            e = self._store.get(k)
            if e is not None and e.get("kind") != "error":
                live_entries.append(e)

        dim = self._index.dim
        bit_width = self._index.bit_width
        new_index = TurboQuantIndex(dim, bit_width)
        new_store: PayloadStore
        if isinstance(self._store, InMemoryStore):
            new_store = InMemoryStore()
        else:
            new_store = self._store
            for i in list(self._store.keys()):
                self._store.delete(i)

        for entry in live_entries:
            key_text = self._key_fn(entry["messages"])
            vec = self._embedder.embed([self._redact(key_text)])
            new_ids = new_index.add(vec)
            new_store.set(int(new_ids[0]), entry)

        self._index = new_index
        self._store = new_store


__all__ = ["SemanticCache"]
