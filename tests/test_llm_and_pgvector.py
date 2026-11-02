import json
from unittest.mock import MagicMock, patch
import pytest
from django.db import connection

from assistant.services import health_status
from rag.generation import _try_llm_answer, generate_answer
from rag.types import RetrievedChunk, TextChunk


def test_try_llm_answer_fallback_when_unset(monkeypatch):
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    chunk = TextChunk(
        chunk_id="c1",
        machine_id="M-HP200",
        manual_id="MAN-HP200",
        revision="R2",
        title="HP200 Manual",
        page_number=3,
        text="ISO VG 68 oil required.",
        source_path="/path/test.pdf",
    )
    evidence = [RetrievedChunk(chunk=chunk, score=0.9)]
    ans = _try_llm_answer("What oil?", "M-HP200", evidence)
    assert ans is None


def test_try_llm_answer_success_with_mock_endpoint(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "http://127.0.0.1:11434/v1")
    monkeypatch.setenv("LLM_MODEL", "llama3.2")
    chunk = TextChunk(
        chunk_id="c1",
        machine_id="M-HP200",
        manual_id="MAN-HP200",
        revision="R2",
        title="HP200 Manual",
        page_number=3,
        text="ISO VG 68 oil required.",
        source_path="/path/test.pdf",
    )
    evidence = [RetrievedChunk(chunk=chunk, score=0.9)]

    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps({
        "choices": [{"message": {"content": "Use ISO VG 68 oil according to page 3."}}]
    }).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        ans = _try_llm_answer("What oil?", "M-HP200", evidence)
        assert ans == "Use ISO VG 68 oil according to page 3."


def test_llm_network_error_falls_back_to_extractive(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "http://127.0.0.1:99999/v1")
    monkeypatch.setenv("LLM_MODEL", "llama3.2")
    chunk = TextChunk(
        chunk_id="c1",
        machine_id="M-HP200",
        manual_id="MAN-HP200",
        revision="R2",
        title="HP200 Manual",
        page_number=3,
        text="ISO VG 68 oil required.",
        source_path="/path/test.pdf",
    )
    evidence = [RetrievedChunk(chunk=chunk, score=0.9)]
    
    result = generate_answer("What oil?", "M-HP200", evidence)
    assert result.status == "grounded"
    assert "Based only on the retrieved manual passages" in result.answer
    assert "ISO VG 68" in result.answer


@pytest.mark.django_db
def test_health_reports_pgvector_readiness(monkeypatch):
    payload = health_status()
    assert "pgvector_ready" in payload
    assert "vector_backend" in payload
    assert payload["database"]["ok"] is True
