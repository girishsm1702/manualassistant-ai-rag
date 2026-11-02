"""Local vector store with machine-filtered similarity search.

PostgreSQL + pgvector can replace this backend later without changing
ingestion, chunking, or the ask() service contract. Filtering by machine_id
happens in the query, not in the LLM prompt.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from rag.hybrid import keyword_scores
from rag.revisions import latest_revision_by_manual
from rag.types import RetrievedChunk, TextChunk

DEFAULT_STORE_DIR = Path("data") / "vectorstore"


def rank_candidates(
    candidate_chunks: list[TextChunk],
    candidate_vectors: np.ndarray,
    query_vector: np.ndarray,
    machine_id: str,
    top_k: int,
    min_score: float,
    query_text: str | None,
) -> list[RetrievedChunk]:
    """Score already machine-filtered chunks. Isolation must be done before this call."""
    if any(chunk.machine_id != machine_id for chunk in candidate_chunks):
        raise RuntimeError("Machine isolation failed before scoring.")
    if len(candidate_chunks) == 0:
        return []
    query = query_vector.astype(np.float32)
    query_norm = np.linalg.norm(query)
    if query_norm == 0:
        return []
    query = query / query_norm
    vector_scores = candidate_vectors @ query
    keyword = (
        np.array(keyword_scores(query_text, candidate_chunks), dtype=np.float32)
        if query_text
        else np.zeros(len(candidate_chunks), dtype=np.float32)
    )
    vector_weight = float(os.getenv("VECTOR_WEIGHT", "0.65"))
    keyword_weight = float(os.getenv("KEYWORD_WEIGHT", "0.35"))
    scores = (vector_weight * vector_scores) + (keyword_weight * keyword)
    ranked = np.argsort(scores)[::-1]
    results: list[RetrievedChunk] = []
    for rank in ranked[:top_k]:
        score = float(scores[rank])
        if score < min_score:
            continue
        chunk = candidate_chunks[int(rank)]
        if chunk.machine_id != machine_id:
            raise RuntimeError("Machine isolation failed: a foreign chunk was retrieved.")
        results.append(RetrievedChunk(chunk=chunk, score=score))
    return results


class LocalVectorStore:
    def __init__(self, store_dir: str | Path = DEFAULT_STORE_DIR) -> None:
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.store_dir / "chunks.jsonl"
        self.vector_path = self.store_dir / "embeddings.npy"
        self.chunks: list[TextChunk] = []
        self.embeddings: np.ndarray = np.zeros((0, 384), dtype=np.float32)
        self._load()

    def _load(self) -> None:
        if self.meta_path.exists():
            self.chunks = []
            with self.meta_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if line:
                        self.chunks.append(TextChunk.from_dict(json.loads(line)))
        if self.vector_path.exists():
            self.embeddings = np.load(self.vector_path)
        if len(self.chunks) != len(self.embeddings):
            raise ValueError(
                "Vector store is out of sync: chunk count does not match embedding count. "
                "Delete data/vectorstore and re-ingest manuals."
            )

    def _save(self) -> None:
        with self.meta_path.open("w", encoding="utf-8") as handle:
            for chunk in self.chunks:
                handle.write(json.dumps(chunk.to_dict(), ensure_ascii=True) + "\n")
        np.save(self.vector_path, self.embeddings)

    def upsert(self, chunks: list[TextChunk], embeddings: np.ndarray) -> int:
        if len(chunks) != len(embeddings):
            raise ValueError("Each chunk must have exactly one embedding.")
        existing_ids = {chunk.chunk_id: index for index, chunk in enumerate(self.chunks)}
        added = 0
        for chunk, vector in zip(chunks, embeddings):
            if chunk.chunk_id in existing_ids:
                index = existing_ids[chunk.chunk_id]
                self.chunks[index] = chunk
                self.embeddings[index] = vector
            else:
                self.chunks.append(chunk)
                if self.embeddings.size == 0:
                    self.embeddings = vector.reshape(1, -1)
                else:
                    self.embeddings = np.vstack([self.embeddings, vector])
                existing_ids[chunk.chunk_id] = len(self.chunks) - 1
                added += 1
        self._save()
        return added

    def search(
        self,
        query_vector: np.ndarray,
        machine_id: str,
        top_k: int = 5,
        revision: str | None = None,
        min_score: float = 0.0,
        query_text: str | None = None,
    ) -> list[RetrievedChunk]:
        """Return the most similar chunks that belong to the selected machine.

        Candidates are filtered by machine_id (and optional revision) before
        scoring. Keyword scores are blended with vector scores when query_text
        is provided, which helps exact terms such as oil grades and part codes.
        """
        if not machine_id.strip():
            raise ValueError("machine_id is required for retrieval.")
        if len(self.chunks) == 0:
            return []

        mask = np.array([chunk.machine_id == machine_id for chunk in self.chunks])
        if revision:
            mask = mask & np.array([chunk.revision == revision for chunk in self.chunks])
        else:
            latest = latest_revision_by_manual(self.chunks, machine_id)
            mask = mask & np.array(
                [chunk.revision == latest.get(chunk.manual_id, chunk.revision) for chunk in self.chunks]
            )
        candidate_indices = np.where(mask)[0]
        if candidate_indices.size == 0:
            return []

        candidate_chunks = [self.chunks[int(index)] for index in candidate_indices]
        return rank_candidates(
            candidate_chunks=candidate_chunks,
            candidate_vectors=self.embeddings[candidate_indices],
            query_vector=query_vector,
            machine_id=machine_id,
            top_k=top_k,
            min_score=min_score,
            query_text=query_text,
        )

    def count_for_machine(self, machine_id: str) -> int:
        return sum(1 for chunk in self.chunks if chunk.machine_id == machine_id)
