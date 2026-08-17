"""Configuration, all from env vars with defaults."""

from __future__ import annotations

import os

UPSTREAM_URL = os.getenv("DEJACACHE_UPSTREAM_URL", "https://api.openai.com/v1")

# Cosine similarity an L1 candidate must clear. Measured against
# BAAI/bge-small-en-v1.5: close paraphrases of a stored prompt land around
# 0.80-0.85, unrelated prompts around 0.65-0.70. See the README.
SIMILARITY_THRESHOLD = float(os.getenv("DEJACACHE_SIMILARITY_THRESHOLD", "0.80"))

# Budget for the whole cache read. Embedding one prompt costs ~5-12ms, so this
# is slack, not a target. Blow it and we pass through instead of waiting.
LOOKUP_TIMEOUT_MS = int(os.getenv("DEJACACHE_LOOKUP_TIMEOUT_MS", "50"))

# How long the customer's own upstream call may take. Their timeout, not ours.
UPSTREAM_TIMEOUT_S = float(os.getenv("DEJACACHE_UPSTREAM_TIMEOUT_S", "120"))

LOG_LEVEL = os.getenv("DEJACACHE_LOG_LEVEL", "INFO")
