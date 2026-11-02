"""Split extracted pages into retrieval chunks while keeping page metadata."""

from __future__ import annotations

import re

from rag.types import ExtractedDocument, TextChunk

DEFAULT_MAX_CHARS = 900
DEFAULT_OVERLAP_CHARS = 120


def chunk_document(
    document: ExtractedDocument,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
) -> list[TextChunk]:
    """Create text chunks from one extracted manual.

    Chunks never cross PDF page boundaries. That keeps citations exact:
    a chunk's page_number is always the original PDF reader page.
    """
    if max_chars < 100:
        raise ValueError("max_chars must be at least 100.")
    if overlap_chars < 0 or overlap_chars >= max_chars:
        raise ValueError("overlap_chars must be >= 0 and smaller than max_chars.")

    chunks: list[TextChunk] = []
    meta = document.metadata

    for page in document.pages:
        if page.is_empty or not page.text:
            continue
        page_parts = _split_page(page.text, max_chars, overlap_chars)
        for index, part in enumerate(page_parts, start=1):
            chunk_id = f"{meta.manual_id}:{meta.revision}:p{page.page_number}:c{index}"
            chunks.append(
                TextChunk(
                    chunk_id=chunk_id,
                    machine_id=meta.machine_id,
                    manual_id=meta.manual_id,
                    revision=meta.revision,
                    title=meta.title,
                    page_number=page.page_number,
                    text=part,
                    source_path=meta.source_path,
                )
            )
    return chunks


def _split_page(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    paragraphs = _paragraphs(text)
    assembled: list[str] = []
    current = ""

    for paragraph in paragraphs:
        candidate = paragraph if not current else f"{current}\n{paragraph}"
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            assembled.append(current)
            current = _overlap_suffix(current, overlap_chars)
            candidate = paragraph if not current else f"{current}\n{paragraph}"
        if len(candidate) <= max_chars:
            current = candidate
            continue
        assembled.extend(_split_long_block(paragraph, max_chars, overlap_chars))
        current = ""

    if current:
        assembled.append(current)
    return assembled


def _paragraphs(text: str) -> list[str]:
    """Keep headings, warnings, and table-like rows as separate paragraphs."""
    blocks = []
    buffer: list[str] = []

    def flush() -> None:
        if buffer:
            blocks.append("\n".join(buffer).strip())
            buffer.clear()

    for line in text.splitlines():
        if _is_heading_or_warning(line) or _looks_like_table_row(line):
            flush()
            blocks.append(line)
        elif not line.strip():
            flush()
        else:
            buffer.append(line)
    flush()
    return [block for block in blocks if block]


def _is_heading_or_warning(line: str) -> bool:
    stripped = line.strip()
    if re.match(r"^(WARNING|CAUTION|DANGER|NOTICE|NOTE)\b", stripped, re.I):
        return True
    if re.match(r"^\d+(\.\d+)*\s+\S+", stripped):
        return True
    if stripped.isupper() and len(stripped) <= 80:
        return True
    return False


def _looks_like_table_row(line: str) -> bool:
    return bool(re.search(r"\s{2,}|\t|\|", line)) and len(line.split()) >= 2


def _split_long_block(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    words = text.split()
    parts: list[str] = []
    current: list[str] = []
    length = 0
    for word in words:
        extra = len(word) + (1 if current else 0)
        if length + extra > max_chars and current:
            parts.append(" ".join(current))
            overlap = " ".join(current)[-overlap_chars:] if overlap_chars else ""
            current = overlap.split() if overlap.strip() else []
            length = len(" ".join(current))
        current.append(word)
        length += extra
    if current:
        parts.append(" ".join(current))
    return parts


def _overlap_suffix(text: str, overlap_chars: int) -> str:
    if overlap_chars <= 0 or len(text) <= overlap_chars:
        return text
    return text[-overlap_chars:].lstrip()
