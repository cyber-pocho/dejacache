"""A broken cache must cost savings, never availability."""

from __future__ import annotations

from conftest import chat, post

from proxy import cache, config, embed, main


def _boom(*args, **kwargs):
    raise RuntimeError("cache is on fire")


def test_failopen_on_cache_error(client, upstream, monkeypatch):
    monkeypatch.setattr(cache, "lookup", _boom)

    response = post(client, chat())

    assert response.status_code == 200
    assert response.json() == upstream.payload
    assert response.headers["X-Dejacache"] == "miss"
    assert upstream.calls == 1


def test_failopen_on_embed_error(client, upstream, monkeypatch):
    monkeypatch.setattr(embed, "embed", _boom)

    first = post(client, chat())
    second = post(client, chat())

    assert (first.status_code, second.status_code) == (200, 200)
    assert first.json() == upstream.payload
    assert second.json()["choices"] == upstream.payload["choices"]
    # The exact layer needs no embedding, so L0 still works; L1 is simply gone.
    assert second.headers["X-Dejacache"] == "hit-l0"


def test_failopen_on_store_error(client, upstream, monkeypatch):
    monkeypatch.setattr(cache, "store", _boom)

    assert post(client, chat()).status_code == 200
    assert post(client, chat()).status_code == 200
    assert upstream.calls == 2


def test_lookup_timeout_passes_through(client, upstream, monkeypatch):
    import time

    def slow(*args, **kwargs):
        time.sleep(0.2)
        return None

    monkeypatch.setattr(config, "LOOKUP_TIMEOUT_MS", 20)
    monkeypatch.setattr(cache, "lookup", slow)

    response = post(client, chat())

    assert response.status_code == 200
    assert response.headers["X-Dejacache"] == "miss"
    assert upstream.calls == 1


def test_proxy_survives_without_the_cache_module(client, upstream, monkeypatch):
    """Killing the cache module entirely still leaves a working proxy."""
    monkeypatch.setattr(main, "cache", None)

    response = post(client, chat())

    assert response.status_code == 200
    assert response.json() == upstream.payload
