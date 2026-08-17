"""Two-layer cache: L0 exact hash, L1 semantic.

Fail-open is a contract, not a suggestion: lookup() returns None on ANY
error. A failure here must mean "no savings for a moment", never "the
customer's app is down".
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from typing import Any, NamedTuple

import numpy as np

from . import embed

log = logging.getLogger(__name__)

# Cosine similarity a candidate must clear to count as an L1 hit. One global
# number today; making it per-tenant (and eventually per-cluster) is the whole
# product.
SIMILARITY_THRESHOLD = 0.92

# Above this, brute-force search stops being free. Switch to hnswlib.
MAX_ENTRIES_PER_TENANT = 50_000


class Hit(NamedTuple):
    response: str
    layer: str
    similarity: float


class _TenantCache:
    """One isolated cache per customer.

    Invariant, held by writing only under `_lock`: `matrix` has exactly as
    many rows as `prompts`/`responses` have entries, and existing rows are
    never reordered or removed. That lets lookup() read a row index it
    computed from a snapshot of `matrix` straight out of `responses`.
    """

    def __init__(self) -> None:
        self.exact: dict[str, str] = {}
        self.matrix: np.ndarray | None = None
        self.prompts: list[str] = []
        self.responses: list[str] = []


_tenants: dict[str, _TenantCache] = {}
_lock = threading.Lock()


def _tenant(tenant_id: str) -> _TenantCache:
    with _lock:
        return _tenants.setdefault(tenant_id, _TenantCache())


def _exact_key(model: str, prompt: str, params: dict[str, Any]) -> str:
    """Hash the full request."""
    blob = json.dumps(
        {"model": model, "prompt": prompt, "params": params},
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(blob.encode()).hexdigest()


def lookup(
    tenant_id: str,
    model: str,
    prompt: str,
    params: dict[str, Any] | None = None,
    threshold: float | None = None,
) -> Hit | None:
    """Return a cached response, or None on a miss. Never raises.

    `threshold` overrides SIMILARITY_THRESHOLD for this call; pass 0.0 to see
    what the nearest neighbour actually scores when calibrating.
    """
    params = params or {}
    if threshold is None:
        threshold = SIMILARITY_THRESHOLD
    try:
        cache = _tenant(tenant_id)

        key = _exact_key(model, prompt, params)
        with _lock:
            response = cache.exact.get(key)
        if response is not None:
            return Hit(response, "L0", 1.0)

        with _lock:
            matrix = cache.matrix
            responses = cache.responses
        if matrix is None:
            return None

        # Embedding is the slow part — keep it off the lock.
        query = embed.embed(prompt)
        sims = matrix @ query
        best = int(np.argmax(sims))
        score = float(sims[best])

        if score >= threshold:
            return Hit(responses[best], "L1", score)
        return None
    except Exception:
        log.exception("cache lookup failed for tenant=%s, passing through", tenant_id)
        return None


def store(
    tenant_id: str,
    model: str,
    prompt: str,
    response: str,
    params: dict[str, Any] | None = None,
) -> None:
    """Record a fresh LLM response. Never raises."""
    params = params or {}
    try:
        cache = _tenant(tenant_id)
        key = _exact_key(model, prompt, params)

        with _lock:
            if len(cache.exact) < MAX_ENTRIES_PER_TENANT or key in cache.exact:
                cache.exact[key] = response
            if len(cache.prompts) >= MAX_ENTRIES_PER_TENANT:
                return

        vec = embed.embed(prompt).reshape(1, -1)

        with _lock:
            # Re-check under the lock: a concurrent store may have filled the
            # last slot while we were embedding.
            if len(cache.prompts) >= MAX_ENTRIES_PER_TENANT:
                return
            cache.matrix = (
                vec if cache.matrix is None else np.vstack([cache.matrix, vec])
            )
            cache.prompts.append(prompt)
            cache.responses.append(response)
    except Exception:
        log.exception("cache store failed for tenant=%s, skipping", tenant_id)


def stats(tenant_id: str) -> dict[str, int]:
    cache = _tenant(tenant_id)
    with _lock:
        return {
            "exact_entries": len(cache.exact),
            "semantic_entries": len(cache.prompts),
        }
