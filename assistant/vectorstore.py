"""Django-backed vector store.

Machine isolation is enforced in the database query (filter by machine_id)
before any similarity scoring. Embeddings are stored as float32 bytes so this
works on SQLite today. PostgreSQL + pgvector can replace the BinaryField later
without changing the ask() contract.
"""

from __future__ import annotations

import numpy as np
from django.db import transaction
from django.db.models import Q

from assistant.models import ManualChunk
from rag.revisions import latest_revision_by_manual
from rag.types import RetrievedChunk, TextChunk
from rag.vectorstore import rank_candidates


def vector_to_bytes(vector: np.ndarray) -> bytes:
    return np.asarray(vector, dtype=np.float32).tobytes()


def bytes_to_vector(payload: bytes) -> np.ndarray:
    return np.frombuffer(payload, dtype=np.float32)


class DjangoChunkStore:
    def upsert(self, chunks: list[TextChunk], embeddings: np.ndarray) -> int:
        if len(chunks) != len(embeddings):
            raise ValueError("Each chunk must have exactly one embedding.")
        added = 0
        with transaction.atomic():
            for chunk, vector in zip(chunks, embeddings):
                _, created = ManualChunk.objects.update_or_create(
                    chunk_id=chunk.chunk_id,
                    defaults={
                        "machine_id": chunk.machine_id,
                        "manual_id": chunk.manual_id,
                        "revision": chunk.revision,
                        "title": chunk.title,
                        "page_number": chunk.page_number,
                        "text": chunk.text,
                        "source_path": chunk.source_path,
                        "embedding": vector_to_bytes(vector),
                    },
                )
                if created:
                    added += 1
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
        if not machine_id.strip():
            raise ValueError("machine_id is required for retrieval.")

        queryset = ManualChunk.objects.filter(machine_id=machine_id)
        if revision:
            queryset = queryset.filter(revision=revision)
        else:
            latest = self._latest_revisions(machine_id)
            if not latest:
                return []
            allowed = list(latest.items())
            revision_filter = Q()
            for manual_id, rev in allowed:
                revision_filter |= Q(manual_id=manual_id, revision=rev)
            queryset = queryset.filter(revision_filter)

        rows = list(queryset)
        if not rows:
            return []
        if any(row.machine_id != machine_id for row in rows):
            raise RuntimeError("Machine isolation failed: SQL filter returned another machine.")

        chunks = [
            TextChunk(
                chunk_id=row.chunk_id,
                machine_id=row.machine_id,
                manual_id=row.manual_id,
                revision=row.revision,
                title=row.title,
                page_number=row.page_number,
                text=row.text,
                source_path=row.source_path,
            )
            for row in rows
        ]
        vectors = np.vstack([bytes_to_vector(bytes(row.embedding)) for row in rows])
        return rank_candidates(
            candidate_chunks=chunks,
            candidate_vectors=vectors,
            query_vector=query_vector,
            machine_id=machine_id,
            top_k=top_k,
            min_score=min_score,
            query_text=query_text,
        )

    def count_for_machine(self, machine_id: str) -> int:
        return ManualChunk.objects.filter(machine_id=machine_id).count()

    def _latest_revisions(self, machine_id: str) -> dict[str, str]:
        rows = ManualChunk.objects.filter(machine_id=machine_id).values("manual_id", "revision")
        chunks = [
            TextChunk(
                chunk_id="",
                machine_id=machine_id,
                manual_id=row["manual_id"],
                revision=row["revision"],
                title="",
                page_number=1,
                text="",
                source_path="",
            )
            for row in rows
        ]
        return latest_revision_by_manual(chunks, machine_id)
