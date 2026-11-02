from rag.revisions import revision_sort_key
from rag.sample_pdfs import HP200_R1_PAGES, write_manual_pdf


def test_revision_order():
    assert revision_sort_key("R2") > revision_sort_key("R1")
    assert revision_sort_key("R10") > revision_sort_key("R2")
    assert revision_sort_key("B") > revision_sort_key("A")


def test_ask_defaults_to_latest_revision(rag_service, tmp_path):
    r1 = write_manual_pdf(tmp_path / "r1.pdf", HP200_R1_PAGES)
    r2 = write_manual_pdf(
        tmp_path / "r2.pdf",
        [
            "HYDRAULIC PRESS HP-200\nRevision: R2",
            "3.1 Routine maintenance\nLubricate the ram guide using ISO VG 68 oil.",
        ],
    )
    rag_service.ingest_pdf(r1, "M-HP200", "MAN-HP200", "R1", "HP-200 R1")
    rag_service.ingest_pdf(r2, "M-HP200", "MAN-HP200", "R2", "HP-200 R2")

    latest = rag_service.ask("M-HP200", "What oil grade is used for lubrication?", min_score=0.2)
    combined_latest = latest.answer + " ".join(item.chunk.text for item in latest.evidence)
    assert "ISO VG 68" in combined_latest
    assert "ISO VG 32" not in combined_latest
    assert all(item.chunk.revision == "R2" for item in latest.evidence)

    old = rag_service.ask(
        "M-HP200",
        "What oil grade is used for lubrication?",
        revision="R1",
        min_score=0.2,
    )
    combined_old = old.answer + " ".join(item.chunk.text for item in old.evidence)
    assert "ISO VG 32" in combined_old
    assert all(item.chunk.revision == "R1" for item in old.evidence)
