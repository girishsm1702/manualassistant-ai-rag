"""Create small text PDFs used for local tests. These are not real vendor manuals."""

from __future__ import annotations

from pathlib import Path

import pymupdf as fitz

HP200_PAGES = [
    "HYDRAULIC PRESS HP-200\nOperator and Maintenance Manual\nDocument ID: MAN-HP200\nRevision: R2",
    "WARNING: Isolate and lock out all energy sources before performing maintenance. "
    "Failure to follow lockout procedures can cause serious injury.",
    "3.1 Routine maintenance\n"
    "Lubricate the ram guide every 40 operating hours using ISO VG 68 oil.\n"
    "Inspect hydraulic hoses for cracks once every week.\n"
    "Record each inspection in the machine logbook.",
    "3.2 Operating limits\nDo not exceed 200 bar system pressure.\nMaximum ram temperature is 70 degrees Celsius.",
]

CNCX1_PAGES = [
    "CNC MILL X1\nProgramming and Service Manual\nDocument ID: MAN-CNCX1\nRevision: A",
    "WARNING: Keep hands away from the spindle while it is rotating.",
    "5.2 Spindle warm-up\nRun the spindle at 1000 RPM for 10 minutes before cutting.\n"
    "Do not begin a heavy cut until the spindle has reached operating temperature.",
]


def write_manual_pdf(path: Path, pages: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    document = fitz.open()
    for text in pages:
        page = document.new_page()
        page.insert_text((72, 72), text, fontsize=11)
    document.save(path)
    document.close()
    return path


HP200_R1_PAGES = [
    "HYDRAULIC PRESS HP-200\nOperator Manual\nRevision: R1",
    "3.1 Routine maintenance\nLubricate the ram guide using ISO VG 32 oil.",
]


def write_scanned_pdf(path: Path, text: str) -> Path:
    """Write an image-only PDF so text extraction requires OCR."""
    path.parent.mkdir(parents=True, exist_ok=True)
    source = fitz.open()
    page = source.new_page()
    page.insert_text((72, 120), text, fontsize=18)
    pixmap = page.get_pixmap(dpi=180)
    source.close()

    scanned = fitz.open()
    image_page = scanned.new_page(width=pixmap.width, height=pixmap.height)
    image_page.insert_image(image_page.rect, pixmap=pixmap)
    scanned.save(path)
    scanned.close()
    return path


def create_sample_manuals(output_dir: str | Path) -> dict[str, Path]:
    folder = Path(output_dir)
    return {
        "hp200": write_manual_pdf(folder / "HP200_manual_R2.pdf", HP200_PAGES),
        "hp200_r1": write_manual_pdf(folder / "HP200_manual_R1.pdf", HP200_R1_PAGES),
        "cncx1": write_manual_pdf(folder / "CNCX1_manual_A.pdf", CNCX1_PAGES),
        "empty": write_manual_pdf(folder / "empty_scan.pdf", [" "]),
        "scanned": write_scanned_pdf(
            folder / "HP200_scanned_warning.pdf",
            "WARNING LOCKOUT BEFORE MAINTENANCE USE ISO VG 68 OIL",
        ),
    }
