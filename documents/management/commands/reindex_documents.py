"""
Management command to create and index ChildChunk records for documents from existing parent chunks.

Usage:
    python manage.py reindex_documents --doc <id>
    python manage.py reindex_documents --all
    python manage.py reindex_documents --limit 10
"""

import time
import logging
from django.core.management.base import BaseCommand, CommandError
from documents.models import Document, ChildChunk
from documents.utils.rag import create_children_for_document

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = "Generate, embed, and index ChildChunk records from existing parent DocumentChunk records."

    def add_arguments(self, parser):
        parser.add_argument(
            "--doc",
            type=int,
            default=None,
            help="Reindex only a specific document by ID.",
        )
        parser.add_argument(
            "--all",
            action="store_true",
            help="Reindex all documents, including those that already have child chunks.",
        )
        parser.add_argument(
            "--limit",
            type=int,
            default=None,
            help="Maximum number of documents to process.",
        )

    def handle(self, *args, **options):
        doc_id = options["doc"]
        reindex_all = options["all"]
        limit = options["limit"]

        if doc_id:
            documents = list(Document.objects.filter(id=doc_id))
            if not documents:
                raise CommandError(f"Document with ID {doc_id} not found.")
        else:
            qs = Document.objects.filter(is_processed=True).order_by("id")
            if not reindex_all:
                # Only documents that currently have NO children
                docs_with_children = ChildChunk.objects.values_list("document_id", flat=True).distinct()
                qs = qs.exclude(id__in=docs_with_children)
            if limit:
                qs = qs[:limit]
            documents = list(qs)

        total = len(documents)
        if total == 0:
            self.stdout.write(self.style.WARNING("No documents require reindexing."))
            return

        self.stdout.write(f"Starting reindex for {total} document(s)...")

        success_count = 0
        total_children_created = 0

        for idx, doc in enumerate(documents, start=1):
            parent_count = doc.chunks.count()
            if parent_count == 0:
                self.stdout.write(
                    self.style.WARNING(f"[{idx}/{total}] Doc #{doc.id} '{doc.title}': No parent chunks found, skipping.")
                )
                continue

            self.stdout.write(
                f"[{idx}/{total}] Doc #{doc.id} '{doc.title}' ({parent_count} parents) -> splitting and embedding children..."
            )
            t0 = time.time()
            try:
                children_created = create_children_for_document(doc)
                elapsed = time.time() - t0
                self.stdout.write(
                    self.style.SUCCESS(
                        f"  ✓ Doc #{doc.id}: created and indexed {children_created} child chunks in {elapsed:.2f}s"
                    )
                )
                success_count += 1
                total_children_created += children_created
            except Exception as e:
                elapsed = time.time() - t0
                self.stdout.write(
                    self.style.ERROR(
                        f"  ✗ Doc #{doc.id} failed after {elapsed:.2f}s: {e}"
                    )
                )
                logger.error(f"Reindex failed for doc {doc.id}: {e}", exc_info=True)

        self.stdout.write(
            self.style.SUCCESS(
                f"\nReindex complete: {success_count}/{total} documents processed successfully. "
                f"Total children created: {total_children_created}."
            )
        )

