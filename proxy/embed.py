"""Local embedding."""

from __future__ import annotations

import threading

import numpy as np
from fastembed import TextEmbedding

MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBED_DIM = 384

_model: TextEmbedding | None = None
_model_lock = threading.Lock()


def warm() -> None:
    """Load the model once at startup. Downloads the weights on first use."""
    global _model
    with _model_lock:
        if _model is None:
            _model = TextEmbedding(model_name=MODEL_NAME)


def embed(text: str) -> np.ndarray:
    """Return an L2-normalised float32 vector.

    Normalising means a plain dot product between two vectors is their
    cosine similarity, so the cache can score candidates with one matmul.
    """
    if _model is None:
        warm()
    vec = np.asarray(next(iter(_model.embed([text]))), dtype=np.float32)
    if vec.shape != (EMBED_DIM,):
        raise ValueError(f"expected a {EMBED_DIM}-dim embedding, got {vec.shape}")
    norm = float(np.linalg.norm(vec))
    return vec if norm == 0.0 else vec / norm
