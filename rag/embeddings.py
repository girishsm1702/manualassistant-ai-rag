"""Sentence Transformer embeddings with a single cached model instance."""

from __future__ import annotations

import os
from functools import lru_cache

import numpy as np

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EXPECTED_DIMENSION = 384


@lru_cache(maxsize=2)
def load_embedding_model(model_name: str | None = None, device: str | None = None):
    """Load the embedding model once per process.

    Device is taken from EMBEDDING_DEVICE (cpu or cuda). We default to CPU
    until a later check proves CUDA works in this environment.
    """
    from sentence_transformers import SentenceTransformer

    name = model_name or os.getenv("EMBEDDING_MODEL_NAME", DEFAULT_MODEL_NAME)
    chosen_device = device or os.getenv("EMBEDDING_DEVICE", "cpu")
    return SentenceTransformer(name, device=chosen_device)


def describe_model(model_name: str | None = None, device: str | None = None) -> dict:
    model = load_embedding_model(model_name, device)
    actual_device = str(model.device)
    dimension_fn = getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension
    dimension = int(dimension_fn())
    return {
        "model_name": model_name or os.getenv("EMBEDDING_MODEL_NAME", DEFAULT_MODEL_NAME),
        "device": actual_device,
        "dimension": dimension,
        "cuda_requested": (device or os.getenv("EMBEDDING_DEVICE", "cpu")).lower() == "cuda",
        "cuda_in_use": actual_device.startswith("cuda"),
    }


def embed_texts(texts: list[str], model_name: str | None = None, device: str | None = None) -> np.ndarray:
    if not texts:
        return np.zeros((0, EXPECTED_DIMENSION), dtype=np.float32)
    model = load_embedding_model(model_name, device)
    vectors = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2:
        raise ValueError("Embedding output must be a 2D array.")
    return vectors


def embed_query(text: str, model_name: str | None = None, device: str | None = None) -> np.ndarray:
    vectors = embed_texts([text], model_name=model_name, device=device)
    return vectors[0]
