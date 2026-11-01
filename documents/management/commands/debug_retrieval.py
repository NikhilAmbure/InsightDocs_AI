"""
Management command to debug and compare legacy hybrid retrieval vs. RRF hybrid retrieval.

Usage:
    python manage.py debug_retrieval "<query>" --doc <id>
"""

import re
import time
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from documents.models import Document, ChildChunk
from documents.utils.gemini_chat import (
    _get_query_embedding,
    _hybrid_retrieve_legacy,
    _hybrid_retrieve_rrf,
    _parent_child_retrieve,
    _rerank_chunks_flashrank,
    _multi_doc_retrieve,
)


class Command(BaseCommand):
    help = "Compare legacy hybrid retrieval vs. PostgreSQL FTS + RRF vs. FlashRank local cross-encoder reranking vs. Parent-Child retrieval."

    def add_arguments(self, parser):
        parser.add_argument("query", type=str, help="Search query string.")
        parser.add_argument(
            "--doc",
            type=int,
            default=None,
            help="Document ID to search within.",
        )
        parser.add_argument(
            "--docs",
            type=str,
            default=None,
            help="Comma-separated Document IDs for workspace retrieval debug (e.g. 1,4).",
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
        docs_str = options["docs"]
        candidates_k = options["candidates"] or getattr(settings, "RERANK_CANDIDATES", 20)
        top_n = options["top_n"] or getattr(settings, "RERANK_TOP_N", 5)

        if not doc_id and not docs_str:
            raise CommandError("Please specify either --doc <id> or --docs <id1,id2>.")

        # Workspace multi-doc debug mode
        if docs_str:
            doc_ids = [int(d.strip()) for d in docs_str.split(",") if d.strip().isdigit()]
            documents = list(Document.objects.filter(id__in=doc_ids))
            if not documents:
                raise CommandError(f"No documents found matching IDs: {docs_str}")

            self.stdout.write(f"\n========================================================")
            self.stdout.write(f"DEBUG WORKSPACE RETRIEVAL: {len(documents)} document(s)")
            for d in documents:
                has_c = ChildChunk.objects.filter(document=d).exists()
                c_count = d.child_chunks.count() if has_c else 0
                kind = f"Reindexed ({c_count} children)" if has_c else "Legacy (no children)"
                self.stdout.write(f"  - Doc #{d.id} '{d.title}': {kind}")
            self.stdout.write(f"Query: \"{query_text}\"")
            self.stdout.write(f"========================================================\n")

            query_embedding = _get_query_embedding(query_text)
            ws_details = _multi_doc_retrieve(
                documents, query_text, query_embedding=query_embedding, return_details=True
            )

            all_scored = ws_details.get("all_scored", [])
            all_unique = ws_details.get("all_unique", [])
            final_items = ws_details.get("final_items", [])
            min_rel = getattr(settings, "RERANK_MIN_RELATIVE_SCORE", 0.01)
            top_score = all_unique[0]["hybrid_score"] if all_unique else 0.0
            cutoff = top_score * min_rel if top_score > 0 else 0.0

            self.stdout.write(self.style.SUCCESS(
                f"--- [WORKSPACE MERGED CANDIDATES] (Retrieved {len(all_scored)} chunks across {len(documents)} docs) ---"
            ))
            self.stdout.write(
                f"{'Rank':<5} {'Doc ID':<8} {'Chunk ID':<10} {'Page':<6} {'Score':<12} {'Document Title':<25} {'First 60 chars'}"
            )
            self.stdout.write("-" * 105)
            for idx, item in enumerate(all_scored[:candidates_k], start=1):
                chunk = item["chunk"]
                clean_content = re.sub(r"\s+", " ", chunk.content[:60]).strip()
                page_str = str(chunk.page_number) if chunk.page_number is not None else "-"
                self.stdout.write(
                    f"{idx:<5} {item['doc_id']:<8} {chunk.id:<10} {page_str:<6} {item['hybrid_score']:<12.6f} {item['doc_title'][:23]:<25} {clean_content}"
                )

            self.stdout.write(self.style.SUCCESS(
                f"\n--- [WORKSPACE UNIQUE PARENTS & CUTOFF] (RERANK_MIN_RELATIVE_SCORE={min_rel}, Top Score={top_score:.6f}, Cutoff={cutoff:.6f}) ---"
            ))
            self.stdout.write(
                f"{'Rank':<5} {'Doc ID':<8} {'Chunk ID':<10} {'Page':<6} {'Rerank Score':<14} {'Status':<11} {'Document Title':<25} {'First 60 chars'}"
            )
            self.stdout.write("-" * 115)
            for r_pos, item in enumerate(all_unique, start=1):
                chunk = item["chunk"]
                clean_content = re.sub(r"\s+", " ", chunk.content[:60]).strip()
                page_str = str(chunk.page_number) if chunk.page_number is not None else "-"
                status_str = "[DROPPED]" if item.get("dropped") else "[KEPT]"
                self.stdout.write(
                    f"{r_pos:<5} {item['doc_id']:<8} {chunk.id:<10} {page_str:<6} {item['hybrid_score']:<14.6f} {status_str:<11} {item['doc_title'][:23]:<25} {clean_content}"
                )

            self.stdout.write(self.style.SUCCESS(
                f"\n--- [WORKSPACE FINAL ASSEMBLED CONTEXT] ({len(final_items)} chunks) ---"
            ))
            self.stdout.write(ws_details.get("context_text") or "No context assembled.")
            self.stdout.write("\n")
            return

        # Single document debug mode
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

        # 4. Parent-Child Retrieval (when doc has children or ENABLE_PARENT_CHILD is on)
        enable_parent_child = getattr(settings, "ENABLE_PARENT_CHILD", False)
        has_children = ChildChunk.objects.filter(document=document).exists()
        if enable_parent_child or has_children:
            self.stdout.write(self.style.SUCCESS(
                f"\n--- [4. PARENT-CHILD CANDIDATES] (Child pool, ENABLE_PARENT_CHILD={enable_parent_child}, has_children={has_children}) ---"
            ))
            pc_res = _parent_child_retrieve(
                document, query_text, top_k=top_n, query_embedding=query_embedding
            )
            child_candidates = pc_res.get("child_candidates", [])
            self.stdout.write(
                f"{'Child ID':<10} {'Parent ID':<11} {'Page':<6} {'Dense':<7} {'Lex':<6} {'RRF Score':<11} {'Rerank Score':<14} {'First 80 chars'}"
            )
            self.stdout.write("-" * 110)
            if not child_candidates:
                self.stdout.write("No child chunks retrieved.")
            else:
                for item in child_candidates[:candidates_k]:
                    child = item["child"]
                    dense_rank = str(item.get("dense_rank")) if item.get("dense_rank") is not None else "-"
                    lex_rank = str(item.get("lexical_rank")) if item.get("lexical_rank") is not None else "-"
                    rrf_score = item.get("rrf_score", 0.0)
                    r_score = item.get("rerank_score")
                    r_score_str = f"{r_score:.6f}" if r_score is not None else "-"
                    clean_content = re.sub(r"\s+", " ", child.content[:80]).strip()
                    page_str = str(child.page_number) if child.page_number is not None else "-"
                    self.stdout.write(
                        f"{child.id:<10} {child.parent_id:<11} {page_str:<6} {dense_rank:<7} {lex_rank:<6} {rrf_score:<11.6f} {r_score_str:<14} {clean_content}"
                    )

            # Final winning parent list with relative cutoff details
            min_rel = getattr(settings, "RERANK_MIN_RELATIVE_SCORE", 0.01)
            all_parents = pc_res.get("scored", [])
            top_p_score = all_parents[0]["hybrid_score"] if all_parents else 0.0
            cutoff_val = top_p_score * min_rel if top_p_score > 0 else 0.0

            self.stdout.write(self.style.SUCCESS(
                f"\n--- [5. FINAL DEDUPLICATED PARENT LIST] (RERANK_MIN_RELATIVE_SCORE={min_rel}, Top Score={top_p_score:.6f}, Cutoff={cutoff_val:.6f}) ---"
            ))
            self.stdout.write(
                f"{'Rank':<5} {'Parent ID':<11} {'Page':<6} {'Best Child ID':<15} {'Best Score':<12} {'Status':<11} {'First 80 chars'}"
            )
            self.stdout.write("-" * 105)
            for r_pos, p_item in enumerate(all_parents, start=1):
                parent = p_item["chunk"]
                child = p_item.get("child")
                child_id_str = str(child.id) if child else "-"
                best_score = p_item.get("hybrid_score", 0.0)
                status_str = "[DROPPED]" if p_item.get("dropped") else "[KEPT]"
                clean_content = re.sub(r"\s+", " ", parent.content[:80]).strip()
                page_str = str(parent.page_number) if parent.page_number is not None else "-"
                self.stdout.write(
                    f"{r_pos:<5} {parent.id:<11} {page_str:<6} {child_id_str:<15} {best_score:<12.6f} {status_str:<11} {clean_content}"
                )

        self.stdout.write("\n")
