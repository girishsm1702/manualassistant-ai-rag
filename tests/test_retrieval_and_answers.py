from rag.generation import GROUNDED_STATUS, INSUFFICIENT_STATUS


def test_supported_question_cites_correct_page(loaded_service):
    result = loaded_service.ask(
        machine_id="M-HP200",
        question="What does the manual say about routine maintenance lubrication?",
        min_score=0.25,
    )
    assert result.status == GROUNDED_STATUS
    assert result.citations
    assert any(citation.page_number == 3 for citation in result.citations)
    assert all(item.chunk.machine_id == "M-HP200" for item in result.evidence)
    assert "ISO VG 68" in result.answer or any("ISO VG 68" in item.chunk.text for item in result.evidence)


def test_machine_isolation(loaded_service):
    result = loaded_service.ask(
        machine_id="M-HP200",
        question="How long should the spindle warm-up run before cutting?",
        min_score=0.20,
    )
    assert all(item.chunk.machine_id == "M-HP200" for item in result.evidence)
    assert all(item.chunk.machine_id != "M-CNCX1" for item in result.evidence)
    combined = result.answer + " ".join(item.chunk.text for item in result.evidence)
    assert "1000 RPM" not in combined


def test_unsupported_question_does_not_invent(loaded_service):
    result = loaded_service.ask(
        machine_id="M-HP200",
        question="What is the factory Wi-Fi password for this press?",
        min_score=0.45,
    )
    assert result.status == INSUFFICIENT_STATUS
    assert result.citations == []
    assert "will not guess" in result.answer.lower() or "do not contain enough" in result.answer.lower()
