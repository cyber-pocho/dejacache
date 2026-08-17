"""Manual smoke: point it at a running dejacache and watch the cache work.

    uvicorn proxy.main:app &
    OPENAI_API_KEY=sk-... .venv/bin/python try_it.py

Every line is a real request through the proxy. The last one is the
interesting one: "Can I return shoes I wore outside?" is a *different*
question that looks a lot like the stored one, and a hit there is a wrong
answer served to a customer. Static thresholds cannot tell those apart; the
calibrator in v2 is the fix.
"""

from __future__ import annotations

import os
import time

import httpx

URL = os.getenv("DEJACACHE_URL", "http://localhost:8000")
KEY = os.getenv("OPENAI_API_KEY", "sk-missing")
MODEL = os.getenv("DEJACACHE_MODEL", "gpt-4o")

# (prompt, the layer the brief expects; None = either answer is defensible)
PROBES = [
    ("What's your return policy?", "miss"),
    ("What's your return policy?", "hit-l0"),
    ("How do I return something?", "hit-l1"),
    ("Can I send these back?", "hit-l1"),
    ("Do you ship to Portugal?", "miss"),
    ("Can I return shoes I wore outside?", None),
]

# A response is stored after it has been sent, so a request issued milliseconds
# later can legitimately miss a row that is still being embedded. Real traffic
# lives with that; a demo script should not race itself.
PAUSE_S = 0.3


def ask(client: httpx.Client, prompt: str) -> tuple[str, str, str]:
    response = client.post(
        "/v1/chat/completions",
        json={"model": MODEL, "messages": [{"role": "user", "content": prompt}]},
        headers={"Authorization": f"Bearer {KEY}"},
    )
    if response.status_code != 200:
        return f"HTTP {response.status_code}", "", response.text[:60]
    answer = response.json()["choices"][0]["message"]["content"]
    return (
        response.headers.get("X-Dejacache", "?"),
        response.headers.get("X-Dejacache-Similarity", ""),
        answer.replace("\n", " ")[:48],
    )


def main() -> None:
    with httpx.Client(base_url=URL, timeout=120) as client:
        print(f"{'prompt':<38} {'result':<8} {'sim':>6}  {'brief expects':<14} answer")
        print("-" * 112)
        diverged = []
        for prompt, expected in PROBES:
            layer, similarity, answer = ask(client, prompt)
            if expected and layer != expected:
                diverged.append((prompt, expected, layer, similarity))
            note = expected or "either"
            print(f"{prompt:<38} {layer:<8} {similarity:>6}  {note:<14} {answer}")
            time.sleep(PAUSE_S)

        for prompt, expected, actual, similarity in diverged:
            print(f"\n!!! {prompt!r} expected {expected}, got {actual} at {similarity or 'n/a'}")
            print("!!! The threshold, not the plumbing. Real scores are the point.")

        # `layer`/`similarity` still hold the last probe. Asking it again would
        # only hit L0 against itself.
        print()
        verdict = "FALSE POSITIVE" if layer.startswith("hit") else "correctly missed"
        print(f">>> {PROBES[-1][0]!r} -> {layer} ({similarity or 'n/a'}): {verdict}")
        print(">>> A hit here is a wrong answer served to a customer. This is what")
        print(">>> the v2 calibrator exists to fix; the static threshold cannot.")

        stats = client.get("/v1/stats", headers={"Authorization": f"Bearer {KEY}"}).json()
        print(f"\nstats: {stats}")


if __name__ == "__main__":
    main()
