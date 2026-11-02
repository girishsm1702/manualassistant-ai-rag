"""Retrieve evidence for a question, filtered by machine in the vector query."""

from __future__ import annotations

import os

from typing import Protocol

from rag.embeddings import embed_query
from rag.types import RetrievedChunk
import numpy as np


class SupportsSearch(Protocol):
    def search(
        self,
        query_vector: np.ndarray,
        machine_id: str,
        top_k: int = 5,
        revision: str | None = None,
        min_score: float = 0.0,
        query_text: str | None = None,
    ) -> list[RetrievedChunk]: ...


def retrieve(
    question: str,
    machine_id: str,
    store: SupportsSearch,
    top_k: int | None = None,
    min_score: float | None = None,
    revision: str | None = None,
) -> list[RetrievedChunk]:
    if not question.strip():
        raise ValueError("question is required.")
    if not machine_id.strip():
        raise ValueError("machine_id is required.")

    k = top_k if top_k is not None else int(os.getenv("RETRIEVAL_TOP_K", "5"))
    threshold = min_score if min_score is not None else float(os.getenv("RETRIEVAL_MIN_SCORE", "0.35"))
    query_vector = embed_query(question)
    return store.search(
        query_vector=query_vector,
        machine_id=machine_id.strip(),
        top_k=k,
        revision=revision,
        min_score=threshold,
        query_text=question,
    )
