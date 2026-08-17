"""Shared fixtures. The upstream is always an httpx.MockTransport."""

from __future__ import annotations

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from proxy import cache, main

KEY_A = "sk-tenant-a"
KEY_B = "sk-tenant-b"


def completion(text: str = "Free returns within 30 days.") -> dict:
    return {
        "id": "chatcmpl-upstream",
        "object": "chat.completion",
        "created": 1_700_000_000,
        "model": "gpt-4o",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20},
    }


def chat(prompt: str = "What's your return policy?", **extra) -> dict:
    return {"model": "gpt-4o", "messages": [{"role": "user", "content": prompt}], **extra}


class Upstream:
    """Records every forwarded request and replies with a canned response."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.status = 200
        self.payload: dict | None = completion()

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(self.status, json=self.payload)

    @property
    def calls(self) -> int:
        return len(self.requests)


@pytest.fixture
def upstream() -> Upstream:
    return Upstream()


@pytest.fixture
def client(upstream: Upstream):
    """A TestClient whose shared upstream client talks to the mock.

    The cache is process-global, so it is cleared per test.
    """
    cache._tenants.clear()
    main._counters.clear()
    main._namespaces.clear()
    with TestClient(main.app) as test_client:
        main.app.state.client = httpx.AsyncClient(
            transport=httpx.MockTransport(upstream.handle),
            base_url="https://upstream.test/v1",
        )
        yield test_client


def post(client: TestClient, body: dict, key: str = KEY_A) -> httpx.Response:
    return client.post(
        "/v1/chat/completions", json=body, headers={"Authorization": f"Bearer {key}"}
    )


def body_of(request: httpx.Request) -> dict:
    return json.loads(request.content)
