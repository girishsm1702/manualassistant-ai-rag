from django.core.management.base import BaseCommand, CommandError

from assistant.services import ingest_manual


class Command(BaseCommand):
    help = "Ingest a PDF manual into the RAG store."

    def add_arguments(self, parser):
        parser.add_argument("pdf_path")
        parser.add_argument("--machine-id", required=True)
        parser.add_argument("--manual-id", required=True)
        parser.add_argument("--revision", required=True)
        parser.add_argument("--title", required=True)

    def handle(self, *args, **options):
        try:
            result = ingest_manual(
                pdf_path=options["pdf_path"],
                machine_id=options["machine_id"],
                manual_id=options["manual_id"],
                revision=options["revision"],
                title=options["title"],
            )
        except (FileNotFoundError, ValueError) as exc:
            raise CommandError(str(exc)) from exc
        self.stdout.write(self.style.SUCCESS(str(result)))
