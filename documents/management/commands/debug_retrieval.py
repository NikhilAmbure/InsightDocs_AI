"""
Management command to debug and compare legacy hybrid retrieval vs. RRF hybrid retrieval.

Usage:
    python manage.py debug_retrieval "<query>" --doc <id>
"""

import re
import time
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from documents.models import Document
from documents.utils.gemini_chat import (
    _get_query_embedding,
    _hybrid_retrieve_legacy,
    _hybrid_retrieve_rrf,
    _rerank_chunks_flashrank,
)


class Command(BaseCommand):
    help = "Compare legacy hybrid retrieval vs. PostgreSQL FTS + RRF vs. FlashRank local cross-encoder reranking."

    def add_arguments(self, parser):
        parser.add_argument("query", type=str, help="Search query string.")
        parser.add_argument(
            "--doc",
            type=int,
            required=True,
            help="Document ID to search within.",
        )
        parser.add_argument(
            "--candidates",
            type=int,
            default=None,
            help="Number of candidates to retrieve before reranking (default: settings.RERANK_CANDIDATES or 20).",
        )
        parser.add_argument(
            "--top-n",
            type=int,
            default=None,
            help="Top N reranked chunks to select (default: settings.RERANK_TOP_N or 5).",
        )

    def handle(self, *args, **options):
        query_text = options["query"]
        doc_id = options["doc"]
        candidates_k = options["candidates"] or getattr(settings, "RERANK_CANDIDATES", 20)
        top_n = options["top_n"] or getattr(settings, "RERANK_TOP_N", 5)

        try:
            document = Document.objects.get(id=doc_id)
        except Document.DoesNotExist:
            raise CommandError(f"Document with ID {doc_id} does not exist.")

        self.stdout.write(f"\n========================================================")
        self.stdout.write(f"DEBUG RETRIEVAL: doc_id={doc_id} ('{document.title}')")
        self.stdout.write(f"Query: \"{query_text}\"")
        self.stdout.write(f"========================================================\n")

        # Generate query embedding once for fair comparison
        query_embedding = _get_query_embedding(query_text)

        # 1. Run Legacy Hybrid Retrieval
        legacy_res = _hybrid_retrieve_legacy(
            document, query_text, top_k=10, query_embedding=query_embedding
        )

        self.stdout.write(self.style.WARNING("--- [1. OLD PATH] Legacy Weighted (0.75 vector + 0.25 icontains) ---"))
        self.stdout.write(
            f"{'Rank':<5} {'Chunk ID':<10} {'Page':<6} {'Score':<12} {'First 80 chars'}"
        )
        self.stdout.write("-" * 90)
        legacy_scored = legacy_res.get("scored", [])
        if not legacy_scored:
            self.stdout.write("No chunks retrieved.")
        else:
            for idx, item in enumerate(legacy_scored[:10], start=1):
                chunk = item["chunk"]
                score = item.get("hybrid_score", 0.0)
                clean_content = re.sub(r"\s+", " ", chunk.content[:80]).strip()
                page_str = str(chunk.page_number) if chunk.page_number is not None else "-"
                self.stdout.write(
                    f"{idx:<5} {chunk.id:<10} {page_str:<6} {score:<12.6f} {clean_content}"
                )

        # 2. Run New RRF Hybrid Retrieval (Retrieving candidates_k)
        rrf_res = _hybrid_retrieve_rrf(
            document, query_text, top_k=candidates_k, query_embedding=query_embedding
        )

        rrf_scored = rrf_res.get("scored", [])
        candidate_chunks = [item["chunk"] for item in rrf_scored]

        # 3. Run FlashRank Local Cross-Encoder Reranking
        t_rerank_start = time.time()
        reranked_chunks = _rerank_chunks_flashrank(
            query_text, candidate_chunks, top_n=len(candidate_chunks)
        )
        rerank_latency_ms = (time.time() - t_rerank_start) * 1000

        # Build position and score mapping for reranked chunks
        rerank_pos_by_id = {}
        rerank_score_by_id = {}
        for r_pos, c in enumerate(reranked_chunks, start=1):
            rerank_pos_by_id[c.id] = r_pos
            rerank_score_by_id[c.id] = getattr(c, "rerank_score", 0.0)

        self.stdout.write(self.style.SUCCESS(
            f"\n--- [2. RRF CANDIDATES & RERANK COMPARISON] (Retrieved {len(candidate_chunks)} candidates, Rerank latency: {rerank_latency_ms:.2f}ms) ---"
        ))
        self.stdout.write(
            f"{'RRF':<5} {'Rerank':<8} {'Chunk ID':<10} {'Page':<6} {'Dense':<7} {'Lex':<6} {'RRF Score':<11} {'Rerank Score':<14} {'First 80 chars'}"
        )
        self.stdout.write("-" * 105)
        if not rrf_scored:
            self.stdout.write("No chunks retrieved.")
        else:
            for idx, item in enumerate(rrf_scored[:10], start=1):
                chunk = item["chunk"]
                dense_rank = str(item.get("dense_rank")) if item.get("dense_rank") is not None else "-"
                lex_rank = str(item.get("lexical_rank")) if item.get("lexical_rank") is not None else "-"
                rrf_score = item.get("hybrid_score", 0.0)
                r_pos = str(rerank_pos_by_id.get(chunk.id, "-"))
                r_score = rerank_score_by_id.get(chunk.id)
                r_score_str = f"{r_score:.6f}" if r_score is not None else "-"
                clean_content = re.sub(r"\s+", " ", chunk.content[:80]).strip()
                page_str = str(chunk.page_number) if chunk.page_number is not None else "-"
                self.stdout.write(
                    f"{idx:<5} {r_pos:<8} {chunk.id:<10} {page_str:<6} {dense_rank:<7} {lex_rank:<6} {rrf_score:<11.6f} {r_score_str:<14} {clean_content}"
                )

        # 4. Top N Reranked Winners
        self.stdout.write(self.style.SUCCESS(
            f"\n--- [3. FINAL TOP-{top_n} RERANKED SELECTION] (FlashRank ms-marco-MiniLM-L-12-v2) ---"
        ))
        self.stdout.write(
            f"{'Rank':<5} {'Chunk ID':<10} {'Page':<6} {'Rerank Score':<14} {'First 80 chars'}"
        )
        self.stdout.write("-" * 80)
        for r_pos, c in enumerate(reranked_chunks[:top_n], start=1):
            r_score = getattr(c, "rerank_score", 0.0)
            clean_content = re.sub(r"\s+", " ", c.content[:80]).strip()
            page_str = str(c.page_number) if c.page_number is not None else "-"
            self.stdout.write(
                f"{r_pos:<5} {c.id:<10} {page_str:<6} {r_score:<14.6f} {clean_content}"
            )

        self.stdout.write("\n")
