"""Optional OCR for scanned PDF pages.

Uses RapidOCR when installed. If the engine is missing, ingestion keeps the
page marked as needing OCR instead of inventing text.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache

import numpy as np

logger = logging.getLogger("rag")


def ocr_enabled() -> bool:
    return os.getenv("OCR_ENABLED", "true").lower() == "true"


@lru_cache(maxsize=1)
def load_ocr_engine():
    try:
        from rapidocr import RapidOCR
    except ImportError:
        try:
            from rapidocr_onnxruntime import RapidOCR
        except ImportError:
            logger.warning("RapidOCR is not installed. Scanned pages will not be OCR'd.")
            return None
    return RapidOCR()


def ocr_available() -> bool:
    return load_ocr_engine() is not None


def image_to_text(image: np.ndarray) -> str:
    engine = load_ocr_engine()
    if engine is None:
        return ""
    result = engine(image)
    lines = _lines_from_result(result)
    return "\n".join(line for line in lines if line.strip())


def pixmap_to_array(pixmap) -> np.ndarray:
    image = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(pixmap.height, pixmap.width, pixmap.n)
    if pixmap.n == 4:
        return image[:, :, :3]
    if pixmap.n == 1:
        return np.repeat(image, 3, axis=2)
    return image


def ocr_pdf_page(page, dpi: int = 200) -> tuple[str, str | None]:
    """OCR one PyMuPDF page. Returns (text, warning)."""
    if not ocr_enabled():
        return "", "OCR is disabled by OCR_ENABLED."
    if not ocr_available():
        return "", "OCR engine is not installed."
    try:
        pixmap = page.get_pixmap(dpi=dpi)
        image = pixmap_to_array(pixmap)
        text = image_to_text(image)
    except Exception as exc:  # OCR must never crash ingestion
        logger.exception("OCR failed")
        return "", f"OCR failed: {exc}"
    if not text.strip():
        return "", "OCR ran but found no readable text."
    return text, None


def _lines_from_result(result) -> list[str]:
    if result is None:
        return []
    if hasattr(result, "txts"):
        return [str(item) for item in (result.txts or ()) if item]
    payload = result
    if isinstance(result, tuple):
        payload = result[0]
    if payload is None:
        return []
    if isinstance(payload, dict):
        texts = payload.get("rec_texts") or payload.get("txts") or []
        return [str(item) for item in texts]
    lines = []
    for item in payload:
        if item is None:
            continue
        if isinstance(item, str):
            lines.append(item)
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            lines.append(str(item[1]))
        elif hasattr(item, "text"):
            lines.append(str(item.text))
    return lines
