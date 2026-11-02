"""Shared data types for the RAG pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class DocumentMetadata:
    machine_id: str
    manual_id: str
    revision: str
    title: str
    source_path: str


@dataclass
class PageText:
    page_number: int
    text: str
    is_empty: bool
    needs_ocr: bool
    char_count: int


@dataclass
class ExtractedDocument:
    metadata: DocumentMetadata
    page_count: int
    pages: list[PageText]
    warnings: list[str] = field(default_factory=list)


@dataclass
class TextChunk:
    chunk_id: str
    machine_id: str
    manual_id: str
    revision: str
    title: str
    page_number: int
    text: str
    source_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TextChunk":
        return cls(
            chunk_id=data["chunk_id"],
            machine_id=data["machine_id"],
            manual_id=data["manual_id"],
            revision=data["revision"],
            title=data["title"],
            page_number=int(data["page_number"]),
            text=data["text"],
            source_path=data["source_path"],
        )


@dataclass
class RetrievedChunk:
    chunk: TextChunk
    score: float


@dataclass
class Citation:
    manual_id: str
    title: str
    revision: str
    page_number: int
    excerpt: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AskResult:
    status: str
    answer: str
    citations: list[Citation]
    evidence: list[RetrievedChunk]
    machine_id: str
    question: str
    latency_ms: float | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "status": self.status,
            "answer": self.answer,
            "machine_id": self.machine_id,
            "question": self.question,
            "latency_ms": self.latency_ms,
            "citations": [citation.to_dict() for citation in self.citations],
            "evidence": [
                {
                    "score": round(item.score, 4),
                    "manual_id": item.chunk.manual_id,
                    "title": item.chunk.title,
                    "revision": item.chunk.revision,
                    "page_number": item.chunk.page_number,
                    "excerpt": item.chunk.text,
                }
                for item in self.evidence
            ],
        }
        return payload
