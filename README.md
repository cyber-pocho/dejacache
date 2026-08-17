# dejacache

An OpenAI-compatible proxy that answers repeated and near-repeated prompts from
a local cache instead of paying for them twice.

```python
client = OpenAI(base_url="http://localhost:8000/v1")   # the only change
```

Your key goes upstream unchanged — we never substitute our own, never log it,
and never store it. It is hashed into a 32-character tenant id, which is the
only thing that reaches the cache. A different key is a different namespace.

## Quickstart

```bash
pip install -r requirements.txt
cp .env.example .env
uvicorn proxy.main:app                       # loads the embedding model at startup

OPENAI_API_KEY=sk-... python try_it.py       # watch hits and misses happen
```

## What comes back

Every response carries three headers:

| Header | Meaning |
|---|---|
| `X-Dejacache` | `hit-l0` (exact), `hit-l1` (semantic), `miss`, `bypass` |
| `X-Dejacache-Similarity` | cosine score, on `hit-l1` only |
| `X-Dejacache-Latency-Ms` | what *we* added, excluding the upstream call |

A hit rebuilds a valid response with a fresh `id` and `created`, keeps `model`
and `finish_reason`, and zeroes `usage` — those tokens genuinely were not spent.
`GET /v1/stats` reports your own entries and counters; `GET /health` needs no auth.

## Configuration

| Env var | Default | Notes |
|---|---|---|
| `DEJACACHE_UPSTREAM_URL` | `https://api.openai.com/v1` | any OpenAI-compatible endpoint |
| `DEJACACHE_SIMILARITY_THRESHOLD` | `0.80` | raise for fewer, safer hits |
| `DEJACACHE_LOOKUP_TIMEOUT_MS` | `50` | slower than this and we pass through |
| `DEJACACHE_UPSTREAM_TIMEOUT_S` | `120` | your call, your timeout |
| `DEJACACHE_LOG_LEVEL` | `INFO` | `DEBUG` logs prompt text; INFO never does |

## What v1 caches

Deliberately narrow: exactly one `user` message (an optional `system` message is
fine), no streaming, `n` of 1, no tools. Anything else is forwarded untouched and
marked `bypass`. Multi-turn and streaming caching are v2.

`model`, `temperature`, `top_p`, `max_tokens`, `seed`, `response_format` and the
system message all partition the cache, so a different temperature is a
different request at both layers.

## Honest numbers

Real-world semantic hit rates run **20–45%**, not the 90% you will see claimed
elsewhere. The ceiling is set by how much your traffic actually repeats itself,
and by how much paraphrase you dare accept before serving a wrong answer.

Measured against the shipped model (`BAAI/bge-small-en-v1.5`), with
*"What's your return policy?"* stored:

| Prompt | Score | At 0.80 |
|---|---|---|
| `"How do I return something?"` | 0.818 | hit |
| `"Can I send these back?"` | 0.713 | miss — a paraphrase we would like to catch |
| `"Can I return shoes I wore outside?"` | 0.687 | miss — a different question we must not answer |

Those last two rows are the whole problem: one static number cannot both catch
the second and reject the third. Tuning it per tenant is v2's job. Run
`try_it.py` to see your own scores before you trust a threshold.

## Failure behaviour

The cache can only cost you savings, never availability. Every cache operation
is wrapped; a lookup slower than `LOOKUP_TIMEOUT_MS` is abandoned, a raising
cache or embedding model is logged and skipped, and the request goes upstream
as if we were not here. Upstream errors are relayed with their status and body
unchanged — a 429 stays a 429. Deleting `proxy/cache.py` leaves a working proxy.

Storage is in-memory: a restart clears the cache.

## Tests

```bash
pip install pytest        # the only test-time dependency
python -m pytest          # upstream is an httpx.MockTransport; no real calls
```
