"""PDF ingestion: page-by-page text extraction with original page numbers."""

from __future__ import annotations

from pathlib import Path

import pymupdf as fitz

from rag.ocr import ocr_pdf_page
from rag.types import DocumentMetadata, ExtractedDocument, PageText

# Pages with fewer than this many letters after cleaning are treated as empty.
# Scanned pages usually have 0 extractable characters.
MIN_TEXT_CHARS = 20


def extract_pdf(
    pdf_path: str | Path,
    machine_id: str,
    manual_id: str,
    revision: str,
    title: str,
) -> ExtractedDocument:
    """Extract text from each PDF page.

    Page numbers are 1-based so they match the page shown in a PDF reader.
    PyMuPDF's page.number is 0-based; we add 1 and never store the raw index.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file, got: {path.name}")
    if not machine_id.strip() or not manual_id.strip():
        raise ValueError("machine_id and manual_id are required.")

    metadata = DocumentMetadata(
        machine_id=machine_id.strip(),
        manual_id=manual_id.strip(),
        revision=revision.strip() or "unknown",
        title=title.strip() or path.stem,
        source_path=str(path.resolve()),
    )

    pages: list[PageText] = []
    warnings: list[str] = []

    with fitz.open(path) as document:
        if document.page_count == 0:
            warnings.append("The PDF contains no pages.")
            return ExtractedDocument(
                metadata=metadata,
                page_count=0,
                pages=[],
                warnings=warnings,
            )

        for page in document:
            raw_text = page.get_text("text") or ""
            cleaned = clean_page_text(raw_text)
            char_count = len(cleaned)
            is_empty = char_count < MIN_TEXT_CHARS
            has_images = bool(page.get_images())
            needs_ocr = is_empty and has_images
            pdf_page_number = page.number + 1
            ocr_used = False

            if needs_ocr:
                ocr_text, ocr_warning = ocr_pdf_page(page)
                if ocr_text:
                    cleaned = clean_page_text(ocr_text)
                    char_count = len(cleaned)
                    is_empty = char_count < MIN_TEXT_CHARS
                    needs_ocr = is_empty
                    ocr_used = not is_empty
                    warnings.append(f"Page {pdf_page_number} text recovered with OCR.")
                elif ocr_warning:
                    warnings.append(f"Page {pdf_page_number}: {ocr_warning}")

            if is_empty:
                warnings.append(
                    f"Page {pdf_page_number} has little or no extractable text."
                    + (" OCR may be required." if needs_ocr and not ocr_used else "")
                )

            pages.append(
                PageText(
                    page_number=pdf_page_number,
                    text=cleaned,
                    is_empty=is_empty,
                    needs_ocr=needs_ocr,
                    char_count=char_count,
                )
            )

    if all(page.is_empty for page in pages):
        warnings.append(
            "No extractable text was found in this PDF. "
            "If this is a scanned manual, OCR is required before retrieval."
        )

    return ExtractedDocument(
        metadata=metadata,
        page_count=len(pages),
        pages=pages,
        warnings=warnings,
    )


def clean_page_text(text: str) -> str:
    """Normalize whitespace without deleting headings, warnings, or table rows."""
    if not text:
        return ""

    lines = []
    for raw_line in text.splitlines():
        line = raw_line.replace("\u00a0", " ").strip()
        if line:
            lines.append(" ".join(line.split()))
    return "\n".join(lines)
