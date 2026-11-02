"""Setup and verify the pgvector extension on PostgreSQL."""

from __future__ import annotations

from django.core.management.base import BaseCommand
from django.db import connection


class Command(BaseCommand):
    help = "Enable and verify the pgvector extension on PostgreSQL."

    def handle(self, *args, **options):
        engine = connection.vendor
        self.stdout.write(f"Current database engine: {engine}")

        if engine != "postgresql":
            self.stdout.write(
                self.style.WARNING(
                    "SQLite is currently active. Embeddings are stored as float32 binary chunks in SQL.\n"
                    "To enable native pgvector:\n"
                    "  1. Start PostgreSQL (e.g. docker compose up -d)\n"
                    "  2. Set DATABASE_URL=postgresql://manuals:manuals@127.0.0.1:5432/manuals in .env\n"
                    "  3. Run: python manage.py migrate\n"
                    "  4. Re-run: python manage.py setup_pgvector"
                )
            )
            return

        with connection.cursor() as cursor:
            try:
                cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                cursor.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector';")
                row = cursor.fetchone()
                if row:
                    self.stdout.write(self.style.SUCCESS(f"pgvector extension is active (version {row[0]})."))
                else:
                    self.stdout.write(self.style.ERROR("pgvector extension could not be verified."))
            except Exception as exc:
                self.stdout.write(self.style.ERROR(f"Failed to create pgvector extension: {exc}"))
