"""Cache behaviour and passthrough fidelity."""

from __future__ import annotations

import httpx

from conftest import KEY_A, body_of, chat, completion, post

SYSTEM = {"role": "system", "content": "You are terse."}


def test_passthrough_equivalence(client, upstream):
    """A request the cache never touches must be indistinguishable from a
    direct upstream call.

    "Cache disabled" here means a request outside the v1 cacheable shape (a
    multi-turn one), which takes the same no-cache code path.
    """
    multi_turn = chat()
    multi_turn["messages"] += [
        {"role": "assistant", "content": "Sure."},
        {"role": "user", "content": "And for shoes?"},
    ]
    response = post(client, multi_turn)

    assert response.status_code == 200
    assert response.headers["X-Dejacache"] == "bypass"
    assert response.json() == upstream.payload
    # The request itself reaches upstream unmodified, key included.
    assert body_of(upstream.requests[0]) == multi_turn
    assert upstream.requests[0].headers["authorization"] == f"Bearer {KEY_A}"


def test_l0_exact_hit(client, upstream):
    first = post(client, chat(**{"temperature": 0.2}))
    second = post(client, chat(**{"temperature": 0.2}))

    assert first.headers["X-Dejacache"] == "miss"
    assert second.headers["X-Dejacache"] == "hit-l0"
    assert upstream.calls == 1

    replayed = second.json()
    assert replayed["choices"] == first.json()["choices"]  # finish_reason preserved
    assert replayed["model"] == "gpt-4o"
    assert replayed["usage"] == {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    assert replayed["id"] != first.json()["id"]
    assert replayed["created"] >= first.json()["created"]


def test_l1_semantic_hit(client, upstream):
    post(client, chat("What's your return policy?"))
    hit = post(client, chat("How do I return something?"))

    assert hit.headers["X-Dejacache"] == "hit-l1"
    assert float(hit.headers["X-Dejacache-Similarity"]) >= 0.80
    assert hit.json()["choices"][0]["message"]["content"] == "Free returns within 30 days."
    assert upstream.calls == 1


def test_params_change_key(client, upstream):
    post(client, chat(temperature=0.0))
    changed = post(client, chat(temperature=1.0))

    assert changed.headers["X-Dejacache"] == "miss"
    assert upstream.calls == 2


def test_system_message_changes_key(client, upstream):
    post(client, chat())
    with_system = chat()
    with_system["messages"].insert(0, SYSTEM)

    assert post(client, with_system).headers["X-Dejacache"] == "miss"
    assert upstream.calls == 2


def test_tools_bypass(client, upstream):
    body = chat(tools=[{"type": "function", "function": {"name": "lookup_order"}}])
    assert post(client, body).headers["X-Dejacache"] == "bypass"
    assert post(client, body).headers["X-Dejacache"] == "bypass"
    assert upstream.calls == 2


def test_streaming_bypasses(client):
    chunks = [
        b'data: {"choices":[{"delta":{"content":"Free "}}]}\n\n',
        b'data: {"choices":[{"delta":{"content":"returns."}}]}\n\n',
        b"data: [DONE]\n\n",
    ]

    async def stream_body():
        for chunk in chunks:
            yield chunk

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=stream_body(), headers={"content-type": "text/event-stream"})

    from proxy import main

    main.app.state.client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="https://upstream.test/v1"
    )

    response = post(client, chat(stream=True))

    assert response.status_code == 200
    assert response.headers["X-Dejacache"] == "bypass"
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.content == b"".join(chunks)


def test_upstream_error_relayed(client, upstream):
    upstream.status = 429
    upstream.payload = {"error": {"message": "Rate limit reached", "type": "rate_limit_error"}}

    response = post(client, chat())

    assert response.status_code == 429
    assert response.json() == upstream.payload
    # A relayed error is never cached.
    assert post(client, chat()).status_code == 429
    assert upstream.calls == 2


def test_health_needs_no_auth(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_stats_reports_this_tenant(client):
    post(client, chat())
    post(client, chat())

    stats = client.get("/v1/stats", headers={"Authorization": f"Bearer {KEY_A}"}).json()

    assert stats["exact_entries"] == 1
    assert stats["requests"] == 2
    assert stats["hit_l0"] == 1
    assert stats["miss"] == 1
    assert stats["tokens_saved"] == completion()["usage"]["total_tokens"]
    assert client.get("/v1/stats").status_code == 401
