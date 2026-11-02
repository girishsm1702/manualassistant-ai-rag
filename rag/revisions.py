"""Choose the latest document revision without hard-coding machine names."""

from __future__ import annotations

import re

from rag.types import TextChunk

REVISION_NUMBERS = re.compile(r"\d+")
REVISION_LETTERS = re.compile(r"[A-Za-z]+")


def revision_sort_key(revision: str) -> tuple:
    """Turn revision labels such as R1, R2, A, B into a comparable key.

    Higher numeric values rank later. Letter-only labels use alphabetical order.
    """
    text = (revision or "").strip().upper()
    numbers = tuple(int(part) for part in REVISION_NUMBERS.findall(text)) or (0,)
    letters = "".join(REVISION_LETTERS.findall(text))
    return (numbers, letters)


def latest_revision_by_manual(chunks: list[TextChunk], machine_id: str) -> dict[str, str]:
    latest: dict[str, str] = {}
    for chunk in chunks:
        if chunk.machine_id != machine_id:
            continue
        current = latest.get(chunk.manual_id)
        if current is None or revision_sort_key(chunk.revision) > revision_sort_key(current):
            latest[chunk.manual_id] = chunk.revision
    return latest
