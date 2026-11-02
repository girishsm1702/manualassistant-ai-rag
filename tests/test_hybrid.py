from rag.hybrid import keyword_scores, tokenize
from rag.types import TextChunk


def _chunk(text: str, machine_id: str = "M-HP200", page: int = 3) -> TextChunk:
    return TextChunk(
        chunk_id=f"{machine_id}:{page}:{text[:12]}",
        machine_id=machine_id,
        manual_id="MAN-HP200",
        revision="R2",
        title="HP-200",
        page_number=page,
        text=text,
        source_path="sample.pdf",
    )


def test_tokenize_keeps_technical_tokens():
    tokens = tokenize("Use ISO VG 68 oil. Limit is 200 bar.")
    assert "iso" in tokens
    assert "vg" in tokens
    assert "68" in tokens
    assert "200" in tokens


def test_keyword_score_prefers_exact_oil_grade():
    chunks = [
        _chunk("Cover page and machine overview.", page=1),
        _chunk("Lubricate the ram guide using ISO VG 68 oil every 40 hours.", page=3),
        _chunk("Do not exceed 200 bar system pressure.", page=4),
    ]
    scores = keyword_scores("What oil grade is ISO VG 68?", chunks)
    assert scores[1] == max(scores)


def test_hybrid_retrieval_ranks_oil_grade_page(loaded_service):
    result = loaded_service.ask(
        machine_id="M-HP200",
        question="ISO VG 68",
        min_score=0.20,
    )
    assert result.evidence
    assert result.evidence[0].chunk.page_number == 3
    assert result.evidence[0].chunk.machine_id == "M-HP200"
    assert "ISO VG 68" in result.evidence[0].chunk.text
