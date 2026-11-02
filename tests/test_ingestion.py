from rag.ingestion import extract_pdf


def test_extracts_page_by_page_with_one_based_numbers(sample_pdfs):
    extracted = extract_pdf(
        sample_pdfs["hp200"],
        machine_id="M-HP200",
        manual_id="MAN-HP200",
        revision="R2",
        title="HP-200 Manual",
    )
    assert extracted.page_count == 4
    assert [page.page_number for page in extracted.pages] == [1, 2, 3, 4]
    assert "ISO VG 68" in extracted.pages[2].text
    assert extracted.pages[2].page_number == 3


def test_empty_pdf_is_flagged(sample_pdfs):
    extracted = extract_pdf(
        sample_pdfs["empty"],
        machine_id="M-HP200",
        manual_id="MAN-EMPTY",
        revision="R0",
        title="Empty Scan",
    )
    assert extracted.pages[0].is_empty
    assert any("no extractable text" in warning.lower() for warning in extracted.warnings)


def test_missing_file_raises(tmp_path):
    missing = tmp_path / "not-there.pdf"
    try:
        extract_pdf(missing, "M-HP200", "MAN-X", "R1", "Missing")
        raise AssertionError("Expected FileNotFoundError")
    except FileNotFoundError:
        pass
