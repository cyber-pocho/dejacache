"""Local embedding."""

from __future__ import annotations 

import numpy as np 
from fastembed import TextEmbedding

MODEL_NAME="BAAI/bge-small-en-v1.5"
EMBED_DIM=384

_model: TextEmbedding | None = None

def warm()-> None: 
    """Load the model once at startup. First downloads the weights"""
    global _model
    if _model is None: 
        _model = TextEmbedding(model_name=MODEL_NAME)

def embed(text:str)-> np.ndarray: 
    """Returns an L2-normalised float32 vector
    Normalising: dot product for related/similarity.
    """
    if _model is None:
        warm()
    vec = np.asarray(next(iter(_model.embed([text]))), dtype=float32)
    norm=float(np.linalg.norm(vec))
    return vec if norm==0.0 else vec/norm


