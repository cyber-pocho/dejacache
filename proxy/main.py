"""The proxy itself: OpenAI-compatible routes in front of the cache.

Two rules govern everything here.

  * The customer's request must survive us. Every cache operation is wrapped;
    any failure means "forward upstream", never an error the customer sees.
  * The customer's key is theirs. It arrives in Authorization, it leaves in
    Authorization, and the only thing derived from it is a hash used as a
    namespace.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.background import BackgroundTask

from . import cache, config, embed, passthrough

log = logging.getLogger("dejacache")

CHAT_PATH = "/chat/completions"

# Per-tenant counters for this process. In-memory like the cache; a restart
# clears both.
_counters: dict[str, dict[str, int]] = defaultdict(
    lambda: {"requests": 0, "hit_l0": 0, "hit_l1": 0, "miss": 0, "bypass": 0, "tokens_saved": 0}
)

# Which cache partitions each tenant has touched, so /v1/stats can add them up.
_namespaces: dict[str, set[str]] = defaultdict(set)


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": self.formatTime(record),
            "level": record.levelname,
            "msg": record.getMessage(),
        }
        entry.update(getattr(record, "fields", {}))
        return json.dumps(entry)


@asynccontextmanager
async def lifespan(app: FastAPI):
    handler = logging.StreamHandler()
    handler.setFormatter(_JsonFormatter())
    log.handlers = [handler]
    log.setLevel(config.LOG_LEVEL)

    embed.warm()
    app.state.client = passthrough.client()
    log.info("started", extra={"fields": {"upstream": config.UPSTREAM_URL}})
    yield
    await app.state.client.aclose()


app = FastAPI(lifespan=lifespan)


def _api_key(request: Request) -> str | None:
    header = request.headers.get("authorization", "")
    return header[7:] if header.startswith("Bearer ") else None


def _tenant_id(api_key: str) -> str:
    return hashlib.sha256(api_key.encode()).hexdigest()[:32]


def _namespace(tenant: str, model: str, params: dict[str, Any]) -> str:
    """The cache partition a request may read from.

    The tenant alone is not enough. The semantic layer scores prompt text and
    nothing else, so two requests differing only in `temperature` would score
    1.0 against each other and one would be served the other's answer. Folding
    the model and the sampling params into the partition makes a different
    temperature a genuinely different request at both layers.
    """
    blob = json.dumps({"model": model, **params}, sort_keys=True, default=str)
    return f"{tenant}:{hashlib.sha256(blob.encode()).hexdigest()[:16]}"


def _unauthorized() -> JSONResponse:
    return JSONResponse(
        {"error": {"message": "Missing Authorization: Bearer <key>", "type": "invalid_request_error"}},
        status_code=401,
    )


def _cacheable(body: dict[str, Any]) -> bool:
    """v1 caches one narrow shape. Everything else passes through.

    Single-turn, non-streaming, one completion, no tools. Multi-turn and
    streaming are v2.
    """
    if body.get("stream") or body.get("n") not in (None, 1):
        return False
    if any(body.get(k) for k in ("tools", "functions", "tool_choice")):
        return False

    messages = body.get("messages")
    if not isinstance(messages, list):
        return False
    roles = [m.get("role") for m in messages if isinstance(m, dict)]
    if len(roles) != len(messages) or set(roles) - {"system", "user"}:
        return False
    if roles.count("user") != 1 or roles.count("system") > 1:
        return False
    return all(isinstance(m.get("content"), str) for m in messages)


def _key_parts(body: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    """The prompt we embed, plus everything that must change the exact hash."""
    contents = {m["role"]: m["content"] for m in body["messages"]}
    params = {
        "system": contents.get("system"),
        **{
            k: body.get(k)
            for k in ("temperature", "top_p", "max_tokens", "seed", "response_format")
        },
    }
    return body.get("model", ""), contents["user"], params


def _zeroed(usage: Any) -> Any:
    if isinstance(usage, dict):
        return {k: _zeroed(v) for k, v in usage.items()}
    return 0


def _replay(cached: str) -> tuple[dict[str, Any], int]:
    """Rebuild a fresh OpenAI response from a cached body.

    `model` and the choices (so `finish_reason`) come through untouched; only
    the parts that must not be replayed are rewritten. Returns the tokens the
    original call spent, which are the tokens this one saves.
    """
    payload = json.loads(cached)
    saved = int((payload.get("usage") or {}).get("total_tokens") or 0)
    payload["id"] = f"chatcmpl-{uuid.uuid4().hex}"
    payload["created"] = int(time.time())
    if "usage" in payload:
        payload["usage"] = _zeroed(payload["usage"])
    return payload, saved


async def _lookup(
    namespace: str, model: str, prompt: str, params: dict[str, Any]
) -> cache.Hit | None:
    """Read the cache under a hard deadline. A slow cache is a missing cache."""
    return await asyncio.wait_for(
        asyncio.to_thread(
            cache.lookup, namespace, model, prompt, params, config.SIMILARITY_THRESHOLD
        ),
        timeout=config.LOOKUP_TIMEOUT_MS / 1000,
    )


def _serve(tenant: str, hit: cache.Hit, started: float) -> JSONResponse:
    """Turn a cache hit into a response. Raising here means passing through."""
    payload, saved = _replay(hit.response)
    layer = f"hit-{hit.layer.lower()}"
    _counters[tenant][f"hit_{hit.layer.lower()}"] += 1
    _counters[tenant]["tokens_saved"] += saved
    added_ms = (time.perf_counter() - started) * 1000
    log.info(
        "served from cache",
        extra={"fields": {"tenant": tenant, "layer": layer, "added_ms": round(added_ms, 1)}},
    )
    return JSONResponse(
        payload,
        headers=_headers(layer, added_ms, hit.similarity if hit.layer == "L1" else None),
    )


def _store(
    namespace: str, model: str, prompt: str, response: str, params: dict[str, Any]
) -> None:
    """Runs after the response has been sent, so it can only cost savings."""
    try:
        cache.store(namespace, model, prompt, response, params)
    except Exception:
        log.exception("cache store failed", extra={"fields": {"namespace": namespace}})


def _headers(layer: str, added_ms: float, similarity: float | None = None) -> dict[str, str]:
    headers = {"X-Dejacache": layer, "X-Dejacache-Latency-Ms": f"{added_ms:.1f}"}
    if similarity is not None:
        headers["X-Dejacache-Similarity"] = f"{similarity:.4f}"
    return headers


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/stats")
async def stats(request: Request) -> Response:
    api_key = _api_key(request)
    if api_key is None:
        return _unauthorized()
    tenant = _tenant_id(api_key)
    entries = {"exact_entries": 0, "semantic_entries": 0}
    for namespace in _namespaces[tenant]:
        for field, count in cache.stats(namespace).items():
            entries[field] += count
    return JSONResponse({**entries, **_counters[tenant]})


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Response:
    started = time.perf_counter()
    api_key = _api_key(request)
    if api_key is None:
        return _unauthorized()

    tenant = _tenant_id(api_key)
    authorization = request.headers.get("authorization", "")
    content_type = request.headers.get("content-type", "application/json")
    raw = await request.body()
    _counters[tenant]["requests"] += 1

    # Everything cache-related lives in here. If any of it breaks we still have
    # `raw`, a key and an upstream, which is all a proxy needs.
    served = None
    parts = None
    namespace = None
    body: Any = None
    try:
        body = json.loads(raw)
        if isinstance(body, dict) and _cacheable(body):
            parts = _key_parts(body)
            namespace = _namespace(tenant, parts[0], parts[2])
            _namespaces[tenant].add(namespace)
            log.debug("lookup", extra={"fields": {"tenant": tenant, "prompt": parts[1]}})
            hit = await _lookup(namespace, *parts)
            if hit is not None:
                served = _serve(tenant, hit, started)
    except asyncio.TimeoutError:
        log.warning("lookup timed out", extra={"fields": {"tenant": tenant}})
    except Exception:
        log.exception("cache path failed, passing through", extra={"fields": {"tenant": tenant}})

    if served is not None:
        return served

    layer = "miss" if parts else "bypass"
    _counters[tenant][layer] += 1
    http: httpx.AsyncClient = request.app.state.client
    upstream_started = time.perf_counter()

    if isinstance(body, dict) and body.get("stream"):
        upstream = await passthrough.forward_stream(http, CHAT_PATH, raw, authorization, content_type)
        added_ms = (time.perf_counter() - started) * 1000 - (
            time.perf_counter() - upstream_started
        ) * 1000
        return StreamingResponse(
            upstream.aiter_raw(),
            status_code=upstream.status_code,
            media_type=upstream.headers.get("content-type"),
            headers=_headers("bypass", added_ms),
            background=BackgroundTask(upstream.aclose),
        )

    upstream = await passthrough.forward(http, CHAT_PATH, raw, authorization, content_type)
    upstream_ms = (time.perf_counter() - upstream_started) * 1000
    added_ms = (time.perf_counter() - started) * 1000 - upstream_ms

    # Store only a success, only a cacheable shape, and only once the customer
    # already has their bytes: BackgroundTask runs after the response is sent.
    store = None
    if parts and upstream.is_success:
        model, prompt, params = parts
        store = BackgroundTask(
            asyncio.to_thread, _store, namespace, model, prompt, upstream.text, params
        )

    log.info(
        "forwarded upstream",
        extra={
            "fields": {
                "tenant": tenant,
                "layer": layer,
                "status": upstream.status_code,
                "added_ms": round(added_ms, 1),
            }
        },
    )
    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        media_type=upstream.headers.get("content-type"),
        headers=_headers(layer, added_ms),
        background=store,
    )
