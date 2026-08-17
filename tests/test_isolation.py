"""One customer must never be served another customer's response."""

from __future__ import annotations

from conftest import KEY_A, KEY_B, chat, post

from proxy import main


def test_cross_tenant_isolation(client, upstream):
    stored = post(client, chat(), key=KEY_A)
    other = post(client, chat(), key=KEY_B)

    assert stored.headers["X-Dejacache"] == "miss"
    assert other.headers["X-Dejacache"] == "miss"
    assert upstream.calls == 2


def test_paraphrase_does_not_leak_across_tenants(client, upstream):
    post(client, chat("What's your return policy?"), key=KEY_A)
    other = post(client, chat("How do I return something?"), key=KEY_B)

    assert other.headers["X-Dejacache"] == "miss"
    assert upstream.calls == 2


def test_tenant_id_is_a_hash_of_the_key(client):
    tenant = main._tenant_id(KEY_A)

    assert len(tenant) == 32
    assert KEY_A not in tenant
    assert tenant != main._tenant_id(KEY_B)
