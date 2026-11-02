"""Keyword scoring for technical terms that embeddings can miss.

Part numbers, oil grades, and pressure values often match as exact tokens.
Machine filtering still happens before this function is called.
"""

from __future__ import annotations

import math
import re

from rag.types import TextChunk

TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[.-][A-Za-z0-9]+)*")
BM25_K1 = 1.5
BM25_B = 0.75


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text or "")]


def keyword_scores(question: str, chunks: list[TextChunk]) -> list[float]:
    """Return a 0-1 BM25-like score for each chunk against the question."""
    query_tokens = tokenize(question)
    if not query_tokens or not chunks:
        return [0.0] * len(chunks)

    documents = [tokenize(chunk.text) for chunk in chunks]
    doc_count = len(documents)
    avg_len = sum(len(tokens) for tokens in documents) / doc_count
    df: dict[str, int] = {}
    for tokens in documents:
        for token in set(tokens):
            df[token] = df.get(token, 0) + 1

    raw_scores: list[float] = []
    for tokens in documents:
        tf: dict[str, int] = {}
        for token in tokens:
            tf[token] = tf.get(token, 0) + 1
        score = 0.0
        doc_len = max(len(tokens), 1)
        for token in query_tokens:
            if token not in tf:
                continue
            idf = math.log(1.0 + (doc_count - df.get(token, 0) + 0.5) / (df.get(token, 0) + 0.5))
            freq = tf[token]
            denom = freq + BM25_K1 * (1 - BM25_B + BM25_B * doc_len / max(avg_len, 1.0))
            score += idf * (freq * (BM25_K1 + 1) / denom)
        if _has_phrase(question, tokens):
            score += 1.0
        raw_scores.append(score)

    peak = max(raw_scores) if raw_scores else 0.0
    if peak <= 0:
        return [0.0] * len(chunks)
    return [value / peak for value in raw_scores]


def _has_phrase(question: str, doc_tokens: list[str]) -> bool:
    query = tokenize(question)
    if len(query) < 2:
        return False
    window = " ".join(doc_tokens)
    return " ".join(query) in window
