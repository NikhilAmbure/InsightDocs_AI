"""
Management command to backfill search_vector on existing DocumentChunk rows.

Usage:
    python manage.py backfill_search_vector
    python manage.py backfill_search_vector --all
    python manage.py backfill_search_vector --doc 1
"""

import logging
from django.conf import settings
from django.contrib.postgres.search import SearchVector
from django.core.management.base import BaseCommand
from documents.models import DocumentChunk

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = "Backfill search_vector on DocumentChunk rows for PostgreSQL full-text search."

    def add_arguments(self, parser):
        parser.add_argument(
            "--doc",
            type=int,
            default=None,
            help="Backfill only chunks for a specific document ID.",
        )
        parser.add_argument(
            "--all",
            action="store_true",
            help="Recompute search_vector for ALL chunks (default: only chunks with null search_vector).",
        )

    def handle(self, *args, **options):
        doc_id = options.get("doc")
        recompute_all = options.get("all")
        search_config = getattr(settings, "SEARCH_CONFIG", "english")

        chunks_qs = DocumentChunk.objects.all()
        if doc_id:
            chunks_qs = chunks_qs.filter(document_id=doc_id)

        if not recompute_all:
            chunks_qs = chunks_qs.filter(search_vector__isnull=True)

        total_to_update = chunks_qs.count()
        if total_to_update == 0:
            self.stdout.write(self.style.SUCCESS("No DocumentChunk rows need search_vector backfill."))
            return

        self.stdout.write(
            f"Backfilling search_vector for {total_to_update} chunks using config='{search_config}'..."
        )

        updated_count = chunks_qs.update(
            search_vector=SearchVector("content", config=search_config)
        )

        self.stdout.write(
            self.style.SUCCESS(f"Successfully backfilled search_vector for {updated_count} chunks.")
        )

