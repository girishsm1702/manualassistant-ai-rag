import pytest

from rag.ingestion import extract_pdf
from rag.ocr import ocr_available


def test_image_only_pdf_is_detected_as_needing_ocr_when_engine_missing(sample_pdfs, monkeypatch):
    monkeypatch.setenv("OCR_ENABLED", "false")
    extracted = extract_pdf(
        sample_pdfs["scanned"],
        machine_id="M-HP200",
        manual_id="MAN-HP200-SCAN",
        revision="R2",
        title="Scanned warning",
    )
    assert extracted.pages[0].page_number == 1
    assert extracted.pages[0].needs_ocr or extracted.pages[0].is_empty


def test_ocr_recovers_scanned_warning_text(sample_pdfs):
    if not ocr_available():
        pytest.skip("RapidOCR is not installed")
    extracted = extract_pdf(
        sample_pdfs["scanned"],
        machine_id="M-HP200",
        manual_id="MAN-HP200-SCAN",
        revision="R2",
        title="Scanned warning",
    )
    text = extracted.pages[0].text.upper()
    assert extracted.pages[0].page_number == 1
    assert "LOCKOUT" in text or "WARNING" in text
    assert extracted.pages[0].is_empty is False
