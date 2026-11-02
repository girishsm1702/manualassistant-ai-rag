import pytest
from django.urls import reverse
from rest_framework.test import APIClient

from assistant.models import ManualChunk
from assistant.vectorstore import DjangoChunkStore
from rag.sample_pdfs import create_sample_manuals
from rag.services import ManualAssistantService


@pytest.mark.django_db
def test_django_store_enforces_machine_isolation(tmp_path):
    pdfs = create_sample_manuals(tmp_path / "manuals")
    service = ManualAssistantService(store=DjangoChunkStore())
    service.ingest_pdf(pdfs["hp200"], "M-HP200", "MAN-HP200", "R2", "HP-200")
    service.ingest_pdf(pdfs["cncx1"], "M-CNCX1", "MAN-CNCX1", "A", "CNC-X1")

    assert ManualChunk.objects.filter(machine_id="M-HP200").exists()
    assert ManualChunk.objects.filter(machine_id="M-CNCX1").exists()

    result = service.ask("M-HP200", "spindle warm-up 1000 RPM", min_score=0.1)
    assert all(item.chunk.machine_id == "M-HP200" for item in result.evidence)
    combined = result.answer + " ".join(item.chunk.text for item in result.evidence)
    assert "1000 RPM" not in combined


@pytest.mark.django_db
def test_health_and_latency_in_ask_response(tmp_path, settings):
    settings.MEDIA_ROOT = tmp_path / "uploads"
    pdfs = create_sample_manuals(tmp_path / "manuals")
    client = APIClient()
    health = client.get(reverse("health"))
    assert health.status_code == 200
    assert health.data["database"]["ok"] is True
    assert health.data["vector_backend"] == "django_chunks"

    ingest = client.post(
        reverse("assistant-ingest"),
        {
            "machine_id": "M-HP200",
            "manual_id": "MAN-HP200",
            "revision": "R2",
            "title": "HP-200",
            "pdf_path": str(pdfs["hp200"]),
        },
        format="json",
    )
    assert ingest.status_code == 201, ingest.data
    ask = client.post(
        reverse("assistant-ask"),
        {"machine_id": "M-HP200", "question": "routine maintenance lubrication"},
        format="json",
    )
    assert ask.status_code == 200, ask.data
    assert ask.data["latency_ms"] is not None
    assert ask.data["latency_ms"] >= 0
    assert ManualChunk.objects.filter(machine_id="M-HP200").count() > 0
