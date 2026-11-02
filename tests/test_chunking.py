from rag.chunking import chunk_document
from rag.ingestion import extract_pdf


def test_chunks_keep_original_page_and_machine(sample_pdfs):
    extracted = extract_pdf(
        sample_pdfs["hp200"],
        machine_id="M-HP200",
        manual_id="MAN-HP200",
        revision="R2",
        title="HP-200 Manual",
    )
    chunks = chunk_document(extracted, max_chars=400, overlap_chars=40)
    assert chunks
    assert all(chunk.machine_id == "M-HP200" for chunk in chunks)
    assert all(chunk.revision == "R2" for chunk in chunks)
    maintenance = [chunk for chunk in chunks if "ISO VG 68" in chunk.text]
    assert maintenance
    assert all(chunk.page_number == 3 for chunk in maintenance)


def test_warning_line_is_not_dropped(sample_pdfs):
    extracted = extract_pdf(
        sample_pdfs["hp200"],
        machine_id="M-HP200",
        manual_id="MAN-HP200",
        revision="R2",
        title="HP-200 Manual",
    )
    chunks = chunk_document(extracted)
    assert any("WARNING" in chunk.text and chunk.page_number == 2 for chunk in chunks)
