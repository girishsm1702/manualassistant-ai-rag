"""Service interface for backend developers.

The Django API should call this class. It does not hard-code machines.
"""

from __future__ import annotations

from pathlib import Path
import time

from rag.chunking import chunk_document
from rag.embeddings import embed_texts
from rag.generation import generate_answer
from rag.ingestion import extract_pdf
from rag.retrieval import retrieve
from rag.types import AskResult, ExtractedDocument
from rag.vectorstore import LocalVectorStore


class ManualAssistantService:
    def __init__(self, store=None, store_dir: str | Path | None = None) -> None:
        self.store = store or LocalVectorStore(store_dir or Path("data") / "vectorstore")

    def ingest_pdf(
        self,
        pdf_path: str | Path,
        machine_id: str,
        manual_id: str,
        revision: str,
        title: str,
    ) -> dict:
        extracted = extract_pdf(
            pdf_path=pdf_path,
            machine_id=machine_id,
            manual_id=manual_id,
            revision=revision,
            title=title,
        )
        chunks = chunk_document(extracted)
        if not chunks:
            return {
                "status": "no_text",
                "machine_id": machine_id,
                "manual_id": manual_id,
                "revision": revision,
                "page_count": extracted.page_count,
                "chunk_count": 0,
                "warnings": extracted.warnings,
            }

        embeddings = embed_texts([chunk.text for chunk in chunks])
        added = self.store.upsert(chunks, embeddings)
        return {
            "status": "ingested",
            "machine_id": machine_id,
            "manual_id": manual_id,
            "revision": revision,
            "page_count": extracted.page_count,
            "chunk_count": len(chunks),
            "chunks_added": added,
            "warnings": extracted.warnings,
        }

    def ask(
        self,
        machine_id: str,
        question: str,
        revision: str | None = None,
        top_k: int | None = None,
        min_score: float | None = None,
    ) -> AskResult:
        started = time.perf_counter()
        evidence = retrieve(
            question=question,
            machine_id=machine_id,
            store=self.store,
            top_k=top_k,
            min_score=min_score,
            revision=revision,
        )
        result = generate_answer(question=question, machine_id=machine_id, evidence=evidence)
        result.latency_ms = round((time.perf_counter() - started) * 1000, 1)
        return result

    def inspect_extraction(self, pdf_path: str | Path, machine_id: str, manual_id: str) -> ExtractedDocument:
        return extract_pdf(
            pdf_path=pdf_path,
            machine_id=machine_id,
            manual_id=manual_id,
            revision="inspect",
            title=Path(pdf_path).stem,
        )
