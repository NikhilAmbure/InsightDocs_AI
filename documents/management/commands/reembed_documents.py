"""
Management command to re-embed all existing documents with the new 768-dim embeddings.

Usage:
    python manage.py reembed_documents          # re-embed all documents
    python manage.py reembed_documents --id 5   # re-embed only document #5
    python manage.py reembed_documents --dry-run # preview without changes
"""

import logging
import time

from django.core.management.base import BaseCommand

from documents.models import Document
from documents.utils.rag import process_document_for_rag

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = "Re-process all documents with updated embedding dimensions (768-dim) and larger chunks."

    def add_arguments(self, parser):
        parser.add_argument(
            "--id",
            type=int,
            default=None,
            help="Re-embed only a specific document by ID.",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Show what would be re-embedded without actually doing it.",
        )

    def handle(self, *args, **options):
        doc_id = options["id"]
        dry_run = options["dry_run"]

        if doc_id:
            documents = Document.objects.filter(id=doc_id)
        else:
            documents = Document.objects.all()

        total = documents.count()
        if total == 0:
            self.stdout.write(self.style.WARNING("No documents found."))
            return

        self.stdout.write(f"Found {total} document(s) to re-embed.")

        if dry_run:
            for doc in documents:
                chunk_count = doc.chunks.count()
                self.stdout.write(f"  [DRY-RUN] Doc #{doc.id} '{doc.title}' — {chunk_count} existing chunks")
            self.stdout.write(self.style.SUCCESS(f"Dry run complete. {total} document(s) would be re-embedded."))
            return

        success = 0
        failed = 0

        for idx, doc in enumerate(documents, 1):
            self.stdout.write(f"[{idx}/{total}] Processing doc #{doc.id} '{doc.title}'...")
            start = time.time()

            try:
                # Delete old chunks (old 3072-dim embeddings)
                old_count = doc.chunks.count()
                doc.chunks.all().delete()
                doc.is_processed = False
                doc.save(update_fields=["is_processed"])

                # Get file path
                try:
                    file_path = doc.file.path
                except NotImplementedError:
                    file_path = doc.file.url

                # Re-process with new batch embeddings + 768 dims
                process_document_for_rag(doc, file_path)

                elapsed = time.time() - start
                new_count = doc.chunks.count()
                self.stdout.write(
                    self.style.SUCCESS(
                        f"  ✓ Done in {elapsed:.1f}s — {old_count} old chunks → {new_count} new chunks"
                    )
                )
                success += 1

            except Exception as e:
                elapsed = time.time() - start
                self.stdout.write(
                    self.style.ERROR(f"  ✗ Failed after {elapsed:.1f}s — {e}")
                )
                logger.error(f"Re-embed failed for doc #{doc.id}: {e}", exc_info=True)
                failed += 1

        self.stdout.write("")
        self.stdout.write(self.style.SUCCESS(f"Re-embedding complete: {success} succeeded, {failed} failed."))

