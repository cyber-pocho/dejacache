"""Forwarding to the upstream provider with the customer's own key.

The key is copied from the inbound Authorization header to the outbound one
and touched nowhere else: not logged, not stored, not put in an error body.
"""

from __future__ import annotations

import httpx

from . import config


def client() -> httpx.AsyncClient:
    """The one shared client. Created at startup, closed at shutdown."""
    return httpx.AsyncClient(
        base_url=config.UPSTREAM_URL,
        timeout=config.UPSTREAM_TIMEOUT_S,
    )


def _headers(authorization: str, content_type: str) -> dict[str, str]:
    return {"Authorization": authorization, "Content-Type": content_type}


async def forward(
    http: httpx.AsyncClient, path: str, body: bytes, authorization: str, content_type: str
) -> httpx.Response:
    """Send the request upstream and read the full response. No retries."""
    return await http.post(path, content=body, headers=_headers(authorization, content_type))


async def forward_stream(
    http: httpx.AsyncClient, path: str, body: bytes, authorization: str, content_type: str
) -> httpx.Response:
    """Start an upstream request and return before the body arrives.

    The caller owns the response and must aclose() it.
    """
    request = http.build_request(
        "POST", path, content=body, headers=_headers(authorization, content_type)
    )
    return await http.send(request, stream=True)
