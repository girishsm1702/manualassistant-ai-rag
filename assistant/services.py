"""Django-facing wrapper around the RAG service."""

from __future__ import annotations

import logging
import re
from pathlib import Path

from django.conf import settings
from django.core.files.uploadedfile import UploadedFile
from django.db import connection
from django.db.utils import OperationalError

from assistant.models import ManualChunk, ManualRecord, QuestionRecord
from assistant.vectorstore import DjangoChunkStore
from rag.services import ManualAssistantService
from rag.types import AskResult

logger = logging.getLogger("assistant")
SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def get_rag_service() -> ManualAssistantService:
    return ManualAssistantService(store=DjangoChunkStore(), store_dir=settings.VECTORSTORE_DIR)


def save_uploaded_pdf(uploaded: UploadedFile, machine_id: str) -> Path:
    original = Path(uploaded.name or "manual.pdf").name
    cleaned = SAFE_NAME.sub("_", original).strip("._") or "manual.pdf"
    if not cleaned.lower().endswith(".pdf"):
        cleaned += ".pdf"
    dest_dir = Path(settings.MEDIA_ROOT) / SAFE_NAME.sub("_", machine_id)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / cleaned
    with dest.open("wb") as handle:
        for chunk in uploaded.chunks():
            handle.write(chunk)
    logger.info("Saved uploaded PDF for machine %s at %s", machine_id, dest)
    return dest


def ingest_manual(
    machine_id: str,
    manual_id: str,
    revision: str,
    title: str,
    pdf_path: str | None = None,
    file: UploadedFile | None = None,
) -> dict:
    path = save_uploaded_pdf(file, machine_id) if file is not None else Path(pdf_path)
    result = get_rag_service().ingest_pdf(
        pdf_path=path,
        machine_id=machine_id,
        manual_id=manual_id,
        revision=revision,
        title=title,
    )
    logger.info(
        "Ingest result status=%s machine=%s manual=%s chunks=%s",
        result["status"],
        machine_id,
        manual_id,
        result.get("chunk_count"),
    )
    if result["status"] == "ingested":
        ManualRecord.objects.update_or_create(
            machine_id=machine_id,
            manual_id=manual_id,
            revision=revision,
            defaults={
                "title": title,
                "source_path": str(path),
                "page_count": result["page_count"],
                "chunk_count": result["chunk_count"],
            },
        )
    return result


def ask_question(machine_id: str, question: str, revision: str | None = None) -> AskResult:
    result = get_rag_service().ask(
        machine_id=machine_id,
        question=question,
        revision=revision or None,
    )
    QuestionRecord.objects.create(
        machine_id=machine_id,
        question=question,
        status=result.status,
        answer=result.answer,
        latency_ms=result.latency_ms,
    )
    logger.info(
        "Ask status=%s machine=%s citations=%s latency_ms=%s",
        result.status,
        machine_id,
        len(result.citations),
        result.latency_ms,
    )
    return result


def list_manuals(machine_id: str | None = None) -> list[dict]:
    records = ManualRecord.objects.all().order_by("machine_id", "title")
    if machine_id:
        records = records.filter(machine_id=machine_id)
    return [
        {
            "machine_id": record.machine_id,
            "manual_id": record.manual_id,
            "revision": record.revision,
            "title": record.title,
            "page_count": record.page_count,
            "chunk_count": record.chunk_count,
        }
        for record in records
    ]


def health_status() -> dict:
    db_ok = True
    db_error = None
    try:
        connection.ensure_connection()
        connection.cursor().execute("SELECT 1")
    except OperationalError as exc:
        db_ok = False
        db_error = str(exc)
    return {
        "status": "ok" if db_ok else "degraded",
        "database": {"ok": db_ok, "engine": connection.vendor, "error": db_error},
        "vector_backend": "django_chunks",
        "pgvector_ready": connection.vendor == "postgresql",
        "chunk_count": ManualChunk.objects.count(),
        "manual_count": ManualRecord.objects.count(),
    }
