"""OpenAI-compatible FastAPI proxy wrapping SemanticCache.

Point any OpenAI-SDK application at this server to get caching for free::

    from openai import OpenAI
    client = OpenAI(base_url="http://localhost:8000/v1", api_key="anything")
    client.chat.completions.create(model="gpt-4o-mini", messages=[...])

Routes:
  * ``POST /v1/chat/completions`` — OpenAI-compatible (streaming supported).
  * ``GET /v1/models`` — reports the configured model.
  * ``GET /stats`` — cache stats JSON.
  * ``GET /metrics`` — Prometheus exposition (if ``prometheus_client`` installed).
  * ``POST /admin/invalidate`` — delete entries by tenant / age.

Tenant isolation:
  * If ``TenantResolver`` is configured, ``Authorization: Bearer <key>`` maps to a
    tenant id and the cache namespaces per tenant. Default resolver uses the
    bearer token itself as the tenant id (shared cache across requests with the
    same key).
"""
from __future__ import annotations

import json
import time
from typing import Any, Callable, Optional

try:
    from fastapi import FastAPI, Header, HTTPException, Request
    from fastapi.responses import JSONResponse, StreamingResponse, Response
except ImportError as exc:
    raise ImportError(
        "fastapi is required for turbovec.llm.proxy. Install with: pip install turbovec[llm-proxy]"
    ) from exc


from .cache import SemanticCache


TenantResolver = Callable[[Optional[str]], Optional[str]]


def _bearer_tenant(auth: Optional[str]) -> Optional[str]:
    if not auth:
        return None
    parts = auth.split()
    if len(parts) == 2 and parts[0].lower() == "bearer":
        import hashlib
        return hashlib.sha256(parts[1].encode()).hexdigest()[:16]
    return None


def build_app(
    cache: SemanticCache,
    *,
    tenant_resolver: TenantResolver = _bearer_tenant,
    enable_admin: bool = True,
) -> FastAPI:
    app = FastAPI(title="turbovec.llm proxy", version="1.0.0")

    @app.get("/v1/models")
    def models() -> dict:
        m = getattr(cache._llm, "model", "unknown")  # noqa: SLF001
        return {"object": "list", "data": [{"id": m, "object": "model", "owned_by": "turbovec"}]}

    @app.get("/stats")
    def stats() -> dict:
        return cache.stats()

    @app.get("/metrics")
    def metrics() -> Response:
        try:
            from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
        except ImportError:
            raise HTTPException(500, "prometheus_client not installed")
        reg = getattr(cache._metrics, "registry", None)  # noqa: SLF001
        return Response(generate_latest(reg) if reg is not None else generate_latest(),
                        media_type=CONTENT_TYPE_LATEST)

    if enable_admin:
        @app.post("/admin/invalidate")
        async def invalidate(req: Request) -> dict:
            body = await req.json() if (await req.body()) else {}
            removed = cache.invalidate(
                tenant=body.get("tenant"),
                older_than_seconds=body.get("older_than_seconds"),
            )
            return {"removed": removed}

    @app.post("/v1/chat/completions")
    async def chat_completions(
        req: Request,
        authorization: Optional[str] = Header(None),
    ) -> Any:
        body = await req.json()
        messages = body.get("messages") or []
        if not messages:
            raise HTTPException(400, "messages required")
        model = body.get("model")
        stream = bool(body.get("stream", False))
        tenant = tenant_resolver(authorization)

        kwargs = {k: v for k, v in body.items() if k not in {"messages", "stream", "model"}}
        if stream:
            async def gen():
                created = int(time.time())
                try:
                    async for chunk in cache.astream_complete(messages, tenant=tenant, **kwargs):
                        payload = {
                            "id": "chatcmpl-tv",
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model or getattr(cache._llm, "model", "unknown"),  # noqa: SLF001
                            "choices": [{"index": 0, "delta": {"content": chunk}, "finish_reason": None}],
                        }
                        yield f"data: {json.dumps(payload)}\n\n"
                    done = {
                        "id": "chatcmpl-tv",
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model or getattr(cache._llm, "model", "unknown"),  # noqa: SLF001
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    }
                    yield f"data: {json.dumps(done)}\n\n"
                    yield "data: [DONE]\n\n"
                except Exception as exc:
                    err = {"error": {"message": repr(exc), "type": type(exc).__name__}}
                    yield f"data: {json.dumps(err)}\n\n"
                    yield "data: [DONE]\n\n"
            return StreamingResponse(gen(), media_type="text/event-stream")

        try:
            result = await cache.acomplete(messages, tenant=tenant, **kwargs)
        except AttributeError:
            result = cache.complete(messages, tenant=tenant, **kwargs)

        cached = isinstance(result.raw, dict) and result.raw.get("cached", False)
        return JSONResponse(
            {
                "id": "chatcmpl-tv",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": result.model or model or "unknown",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": result.text},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": result.tokens_in,
                    "completion_tokens": result.tokens_out,
                    "total_tokens": result.tokens_in + result.tokens_out,
                },
                "turbovec": {"cached": cached},
            }
        )

    return app


__all__ = ["build_app", "TenantResolver"]
