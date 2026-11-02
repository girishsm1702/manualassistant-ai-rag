from pathlib import Path

import pytest

from rag.sample_pdfs import create_sample_manuals
from rag.services import ManualAssistantService


@pytest.fixture
def sample_pdfs(tmp_path: Path) -> dict[str, Path]:
    return create_sample_manuals(tmp_path / "manuals")


@pytest.fixture
def rag_service(tmp_path: Path) -> ManualAssistantService:
    return ManualAssistantService(store_dir=tmp_path / "vectorstore")


@pytest.fixture
def loaded_service(rag_service: ManualAssistantService, sample_pdfs: dict[str, Path]) -> ManualAssistantService:
    rag_service.ingest_pdf(
        pdf_path=sample_pdfs["hp200"],
        machine_id="M-HP200",
        manual_id="MAN-HP200",
        revision="R2",
        title="Hydraulic Press HP-200 Operator Manual",
    )
    rag_service.ingest_pdf(
        pdf_path=sample_pdfs["cncx1"],
        machine_id="M-CNCX1",
        manual_id="MAN-CNCX1",
        revision="A",
        title="CNC Mill X1 Service Manual",
    )
    return rag_service
