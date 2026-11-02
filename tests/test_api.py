import pytest
from django.urls import reverse
from rest_framework.test import APIClient

from rag.sample_pdfs import create_sample_manuals


@pytest.mark.django_db
def test_ask_endpoint_contract(tmp_path, settings):
    settings.VECTORSTORE_DIR = tmp_path / "vectorstore"
    settings.SAMPLE_MANUAL_DIR = tmp_path / "manuals"
    pdfs = create_sample_manuals(settings.SAMPLE_MANUAL_DIR)
    client = APIClient()

    ingest = client.post(
        reverse("assistant-ingest"),
        {
            "machine_id": "M-HP200",
            "manual_id": "MAN-HP200",
            "revision": "R2",
            "title": "Hydraulic Press HP-200 Operator Manual",
            "pdf_path": str(pdfs["hp200"]),
        },
        format="json",
    )
    assert ingest.status_code == 201, ingest.data
    assert ingest.data["status"] == "ingested"

    ask = client.post(
        reverse("assistant-ask"),
        {
            "machine_id": "M-HP200",
            "question": "What does the manual say about routine maintenance?",
        },
        format="json",
    )
    assert ask.status_code == 200, ask.data
    body = ask.data
    assert "status" in body
    assert "answer" in body
    assert "citations" in body
    assert "evidence" in body
    if body["status"] == "grounded":
        assert body["citations"]
        assert all("page_number" in citation for citation in body["citations"])
        assert all(item["manual_id"] == "MAN-HP200" for item in body["evidence"])


@pytest.mark.django_db
def test_ingest_accepts_uploaded_pdf(tmp_path, settings):
    settings.VECTORSTORE_DIR = tmp_path / "vectorstore"
    settings.MEDIA_ROOT = tmp_path / "uploads"
    pdfs = create_sample_manuals(tmp_path / "manuals")
    client = APIClient()

    with pdfs["hp200"].open("rb") as handle:
        ingest = client.post(
            reverse("assistant-ingest"),
            {
                "machine_id": "M-HP200",
                "manual_id": "MAN-HP200",
                "revision": "R2",
                "title": "Hydraulic Press HP-200 Operator Manual",
                "file": handle,
            },
            format="multipart",
        )
    assert ingest.status_code == 201, ingest.data
    assert ingest.data["status"] == "ingested"

    manuals = client.get(reverse("assistant-manuals"), {"machine_id": "M-HP200"})
    assert manuals.status_code == 200
    assert manuals.data["manuals"]
    assert manuals.data["manuals"][0]["machine_id"] == "M-HP200"
