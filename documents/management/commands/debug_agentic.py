"""
Management command to debug and inspect LangGraph Agentic Retrieval (Phase 2a).

Usage:
    python manage.py debug_agentic "<query>" --workspace <id> [--history "<previous turns>"]
    python manage.py debug_agentic "<query>" --docs <id1,id2> [--history "<previous turns>"]
"""

import asyncio
import json
import logging
import re
import time
from typing import Any, Dict, List

from django.core.management.base import BaseCommand, CommandError
from documents.models import Document, Workspace
from documents.utils.agentic_retrieval import run_agentic_retrieval

logger = logging.getLogger(__name__)


def _parse_history(raw_history: str) -> List[Dict[str, str]]:
    """Parse history from JSON string or pipe/newline-separated text."""
    if not raw_history or not raw_history.strip():
        return []

    raw = raw_history.strip()
    # Try parsing JSON first
    if (raw.startswith("[") and raw.endswith("]")) or (raw.startswith("{") and raw.endswith("}")):
        try:
            data = json.loads(raw)
            if isinstance(data, list):
                parsed = []
                for item in data:
                    if isinstance(item, dict):
                        role = item.get("role", "user")
                        content = item.get("content", "")
                        parsed.append({"role": role, "content": str(content)})
                    elif isinstance(item, str):
                        parsed.append({"role": "user", "content": item})
                return parsed
        except Exception:
            pass

    # Delimited by pipe or newline
    turns = [t.strip() for t in re.split(r"\||\n", raw) if t.strip()]
    history = []
    for turn in turns:
        if turn.lower().startswith("user:"):
            history.append({"role": "user", "content": turn[5:].strip()})
        elif turn.lower().startswith("assistant:") or turn.lower().startswith("model:"):
            content = turn[10:].strip() if turn.lower().startswith("assistant:") else turn[6:].strip()
            history.append({"role": "assistant", "content": content})
        else:
            history.append({"role": "user", "content": turn})

    return history


class Command(BaseCommand):
    help = "Run and debug LangGraph Agentic Retrieval (Node 1: rewrite_and_route, Node 2: retrieve, Node 3: build_context)."

    def add_arguments(self, parser):
        parser.add_argument("query", type=str, help="Search query string.")
        parser.add_argument(
            "--workspace",
            type=int,
            default=None,
            help="Workspace ID to retrieve documents from.",
        )
        parser.add_argument(
            "--docs",
            type=str,
            default=None,
            help="Comma-separated Document IDs (e.g. 1,4) if no workspace ID is specified.",
        )
        parser.add_argument(
            "--history",
            type=str,
            default="",
            help="Previous conversation turns in JSON or 'User: ... | Assistant: ...' format.",
        )

    def handle(self, *args, **options):
        query_text = options["query"]
        workspace_id = options["workspace"]
        docs_str = options["docs"]
        history_str = options["history"]

        documents: List[Document] = []
        workspace_name = "N/A"

        if workspace_id:
            try:
                ws = Workspace.objects.prefetch_related("documents").get(id=workspace_id)
                workspace_name = f"'{ws.name}' (ID: {ws.id})"
                documents = list(ws.documents.filter(is_processed=True))
                if not documents:
                    raise CommandError(f"Workspace {workspace_id} has no processed documents.")
            except Workspace.DoesNotExist:
                raise CommandError(f"Workspace with ID {workspace_id} does not exist.")
        elif docs_str:
            doc_ids = [int(d.strip()) for d in docs_str.split(",") if d.strip().isdigit()]
            documents = list(Document.objects.filter(id__in=doc_ids, is_processed=True))
            if not documents:
                raise CommandError(f"No processed documents found for IDs: {docs_str}")
        else:
            raise CommandError("Please specify either --workspace <id> or --docs <id1,id2>.")

        chat_history = _parse_history(history_str)

        self.stdout.write(f"\n========================================================")
        self.stdout.write(f"DEBUG AGENTIC RETRIEVAL (Agentic retrieval)")
        self.stdout.write(f"Workspace: {workspace_name}")
        self.stdout.write(f"Documents ({len(documents)}):")
        for d in documents:
            c_count = d.child_chunks.count()
            kind = f"Reindexed ({c_count} children)" if c_count > 0 else f"Legacy ({d.chunks.count()} parents)"
            self.stdout.write(f"  - Doc #{d.id} '{d.title}': {kind}")
        self.stdout.write(f"Input Query: \"{query_text}\"")
        if chat_history:
            self.stdout.write(f"History Turns ({len(chat_history)}):")
            for h in chat_history:
                self.stdout.write(f"  [{h['role'].upper()}]: {h['content']}")
        else:
            self.stdout.write("History: None")
        self.stdout.write(f"========================================================\n")

        # Run async agentic retrieval graph
        t_start = time.time()
        res = asyncio.run(run_agentic_retrieval(query_text, documents, chat_history))
        total_time_ms = (time.time() - t_start) * 1000

        route = res.get("route", "direct")
        standalone = res.get("standalone_query", query_text)
        sub_queries = res.get("sub_queries", [])
        retrieved_parents = res.get("retrieved_parents", {})
        timings = res.get("timings", {})
        tokens_used = res.get("tokens_used", 0)

        # 1. Routing & Rewrite Summary
        self.stdout.write(self.style.SUCCESS("--- [1. NODE 1: REWRITE & ROUTE] ---"))
        self.stdout.write(f"Route selected:    {route.upper()}")
        self.stdout.write(f"Standalone Query:  \"{standalone}\"")
        self.stdout.write(f"Tokens consumed:   {tokens_used}")
        self.stdout.write(f"Node Latency:      {timings.get('rewrite_and_route', 0.0):.2f}ms")

        if route == "decompose":
            self.stdout.write(self.style.SUCCESS(f"Sub-queries generated ({len(sub_queries)}):"))
            for sq in sub_queries:
                d_id = sq.get("document_id")
                d_obj = next((d for d in documents if d.id == d_id), None)
                d_title = d_obj.title if d_obj else f"Doc {d_id}"
                self.stdout.write(f"  - Doc #{d_id} ('{d_title}'): \"{sq.get('query')}\"")

        # 2. Per-Document Retrieval Breakdown
        self.stdout.write(self.style.SUCCESS(f"\n--- [2. NODE 2: RETRIEVE] (Latency: {timings.get('retrieve', 0.0):.2f}ms) ---"))
        for doc in documents:
            chunks = retrieved_parents.get(doc.id, [])
            self.stdout.write(self.style.WARNING(f"\nDocument #{doc.id} '{doc.title}' ({len(chunks)} parents retrieved):"))
            if not chunks:
                self.stdout.write("  (No chunks returned)")
            else:
                self.stdout.write(f"  {'Rank':<5} {'Chunk ID':<10} {'Page':<6} {'Score':<14} {'First 80 chars'}")
                self.stdout.write("  " + "-" * 95)
                for rank, c in enumerate(chunks, start=1):
                    score = getattr(c, "rerank_score", None)
                    if score is None:
                        score_str = "-"
                    else:
                        score_str = f"{float(score):.6f}"
                    clean_content = re.sub(r"\s+", " ", c.content[:80]).strip()
                    page_str = str(c.page_number) if c.page_number is not None else "-"
                    self.stdout.write(f"  {rank:<5} {c.id:<10} {page_str:<6} {score_str:<14} {clean_content}")

        # 3. Corrective Retry Summary (Phase 2b)
        retry_fired = res.get("retry_fired", False)
        first_pass_score = res.get("first_pass_score", 0.0)
        retry_pass_score = res.get("retry_pass_score", 0.0)
        alternative_query = res.get("alternative_query")
        retry_winner = res.get("retry_winner")
        retry_latency = timings.get("retry_retrieve", 0.0)

        self.stdout.write(self.style.SUCCESS(f"\n--- [3. CORRECTIVE RETRY (Phase 2b)] ---"))
        self.stdout.write(f"First Pass Top Score:  {first_pass_score:.6f}")
        self.stdout.write(f"Retry Fired:          {'YES' if retry_fired else 'NO'}")
        if retry_fired:
            self.stdout.write(f"Alternative Query:    \"{alternative_query}\"")
            self.stdout.write(f"Retry Pass Top Score: {retry_pass_score:.6f}")
            self.stdout.write(f"Winner:               {('RETRY PASS' if retry_winner == 'retry' else 'FIRST PASS')}")
            self.stdout.write(f"Added Retry Latency:  {retry_latency:.2f}ms")

        # 4. Context & Citations
        self.stdout.write(self.style.SUCCESS(f"\n--- [4. BUILD CONTEXT] (Latency: {timings.get('build_context', 0.0):.2f}ms) ---"))
        sources = res.get("sources", [])
        self.stdout.write(f"Total Sources Formatted: {len(sources)}")
        self.stdout.write(self.style.SUCCESS("\nAssembled Context Preview:"))
        context_preview = res.get("context_text", "")
        if context_preview:
            # Print first 500 chars of context
            if len(context_preview) > 500:
                self.stdout.write(context_preview[:500] + f"\n... [+{len(context_preview) - 500} more characters]")
            else:
                self.stdout.write(context_preview)
        else:
            self.stdout.write("(Empty context)")

        # 5. Latency Summary
        self.stdout.write(self.style.SUCCESS(f"\n--- [5. EXECUTION TIMINGS] ---"))
        self.stdout.write(f"Node 1 (rewrite_and_route): {timings.get('rewrite_and_route', 0.0):.2f}ms")
        self.stdout.write(f"Node 2 (retrieve):          {timings.get('retrieve', 0.0):.2f}ms")
        if "retry_retrieve" in timings:
            self.stdout.write(f"Node (retry_retrieve):      {timings.get('retry_retrieve', 0.0):.2f}ms")
        self.stdout.write(f"Node 3 (build_context):     {timings.get('build_context', 0.0):.2f}ms")
        self.stdout.write(f"Total Graph Wall Time:      {total_time_ms:.2f}ms\n")

