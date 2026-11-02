from django.conf import settings
from django.core.management.base import BaseCommand

from assistant.services import ingest_manual
from rag.sample_pdfs import create_sample_manuals


class Command(BaseCommand):
    help = "Create and ingest the two sample manuals used for local demos."

    def handle(self, *args, **options):
        paths = create_sample_manuals(settings.SAMPLE_MANUAL_DIR)
        hp = ingest_manual(
            pdf_path=str(paths["hp200"]),
            machine_id="M-HP200",
            manual_id="MAN-HP200",
            revision="R2",
            title="Hydraulic Press HP-200 Operator Manual",
        )
        cnc = ingest_manual(
            pdf_path=str(paths["cncx1"]),
            machine_id="M-CNCX1",
            manual_id="MAN-CNCX1",
            revision="A",
            title="CNC Mill X1 Service Manual",
        )
        self.stdout.write(self.style.SUCCESS(str(hp)))
        self.stdout.write(self.style.SUCCESS(str(cnc)))
