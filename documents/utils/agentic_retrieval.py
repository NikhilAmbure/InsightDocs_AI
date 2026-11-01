"""
Agentic Retrieval using LangGraph (Phase 2a).

Scope: Retrieval only. Runs before answer generation, providing:
1. Query rewriting with conversation history and routing (direct vs. decompose).
2. Decomposed multi-document concurrent retrieval (via asyncio.gather) or direct retrieval.
3. Unified context building and source tracking matching existing citation formats.
"""

import asyncio
import logging
import math
import re
import time
from typing import Any, Dict, List, Literal, Optional, TypedDict

import google.generativeai as genai
from django.conf import settings
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from .gemini_chat import _hybrid_retrieve, _multi_doc_retrieve, _safe_json_loads

logger = logging.getLogger(__name__)

# Fallback/default timeout for rewrite LLM call in seconds
REWRITE_TIMEOUT_SECONDS = 7.0


def _get_doc_id(doc: Any) -> Any:
    if isinstance(doc, dict):
        return doc.get("id")
    return getattr(doc, "id", None)


def _get_doc_title(doc: Any, default: str = "Document") -> str:
    if isinstance(doc, dict):
        return doc.get("title", default)
    return getattr(doc, "title", default)


class SubQuery(BaseModel):
    document_id: int
    query: str


class RewriteAndRouteOutput(BaseModel):
    standalone_query: str
    route: Literal["direct", "decompose"]
    sub_queries: List[SubQuery] = Field(default_factory=list)


class AgenticState(TypedDict):
    user_message: str
    chat_history: List[Dict[str, str]]
    documents: List[Any]  # List of Document objects or dicts with id, title
    standalone_query: str
    route: Literal["direct", "decompose"]
    sub_queries: List[Dict[str, Any]]  # [{"document_id": int, "query": str}]
    retrieved_parents: Dict[int, List[Any]]  # doc_id -> list of DocumentChunk
    sources: List[Dict[str, Any]]
    context_text: str
    timings: Dict[str, float]  # node_name -> latency_ms
    tokens_used: int  # tokens from rewrite LLM call
    error: Optional[str]


def _is_summary_query(query_text: str) -> bool:
    """Check if query matches global summary / list all bypass patterns."""
    query_lower = query_text.lower()
    global_keywords = [
        "list all", "all questions", "every question", "entire", "summarize",
        "summary", "full", "whole", "table of contents", "index", "complete list",
        "all the questions", "list questions", "list the questions"
    ]
    return any(w in query_lower for w in global_keywords)


async def rewrite_and_route_node(state: AgenticState) -> Dict[str, Any]:
    """
    Node 1: rewrite_and_route
    Rewrites the user question into a standalone query using conversation history,
    and routes to 'direct' or 'decompose' (when comparing/cross-referencing across 2+ documents).
    """
    t0 = time.time()
    user_message = state["user_message"]
    chat_history = state.get("chat_history") or []
    documents = state.get("documents") or []

    # Rule 1: Skip LLM call entirely if no chat history AND only 1 document is selected
    if len(chat_history) == 0 and len(documents) <= 1:
        elapsed_ms = (time.time() - t0) * 1000
        timings = dict(state.get("timings") or {})
        timings["rewrite_and_route"] = elapsed_ms
        return {
            "standalone_query": user_message,
            "route": "direct",
            "sub_queries": [],
            "tokens_used": 0,
            "timings": timings,
        }

    # Rule 2: Skip LLM call entirely for summary / list all bypass queries
    if _is_summary_query(user_message):
        elapsed_ms = (time.time() - t0) * 1000
        timings = dict(state.get("timings") or {})
        timings["rewrite_and_route"] = elapsed_ms
        return {
            "standalone_query": user_message,
            "route": "direct",
            "sub_queries": [],
            "tokens_used": 0,
            "timings": timings,
        }

    # Format history turns for prompt
    history_lines = []
    for msg in chat_history[-6:]:
        role = msg.get("role", "user")
        content = msg.get("content", "").strip()
        if content:
            history_lines.append(f"{role.capitalize()}: {content}")
    history_str = "\n".join(history_lines) if history_lines else "None"

    # Format documents for prompt
    doc_lines = []
    for d in documents:
        doc_id = _get_doc_id(d)
        doc_title = _get_doc_title(d, default=f"Doc {doc_id}")
        doc_lines.append(f"- ID {doc_id}: \"{doc_title}\"")
    docs_str = "\n".join(doc_lines) if doc_lines else "None"

    prompt = f"""You are a query analysis and routing module for an enterprise document QA system.
Analyze the user's latest question in the context of recent chat history and available documents.

AVAILABLE DOCUMENTS:
{docs_str}

CONVERSATION HISTORY:
{history_str}

LATEST USER QUESTION:
{user_message}

TASKS:
1. standalone_query: Rewrite the user question so it is completely self-contained. Resolve any pronouns, implicit references, or follow-up references using the conversation history. If no history is relevant, keep the user question.
2. route:
   - Set to "decompose" ONLY if the question asks to compare, contrast, correlate, or retrieve distinct information across 2 or more of the available documents.
   - Otherwise, set to "direct".
3. sub_queries:
   - If route is "decompose", generate exactly one focused, document-specific search query for each relevant document as: [{{"document_id": <int>, "query": "<focused search query>"}}].
   - If route is "direct", set sub_queries to [].

Return STRICT JSON matching this schema:
{{
  "standalone_query": "...",
  "route": "direct" | "decompose",
  "sub_queries": [
    {{"document_id": 1, "query": "..."}}
  ]
}}
"""

    tokens_used = 0
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")

        async def _call_gemini():
            return await model.generate_content_async(prompt)

        response = await asyncio.wait_for(_call_gemini(), timeout=REWRITE_TIMEOUT_SECONDS)

        # Track usage tokens from rewrite call
        usage_meta = getattr(response, "usage_metadata", None)
        if usage_meta is not None:
            tokens_used = int(getattr(usage_meta, "total_token_count", 0) or 0)

        raw_text = response.text or ""
        json_payload = _safe_json_loads(raw_text)
        if not json_payload or not isinstance(json_payload, dict):
            raise ValueError(f"Invalid JSON returned: {raw_text[:100]}")

        parsed = RewriteAndRouteOutput.model_validate(json_payload)

        # Guard: route can only be 'decompose' if 2+ documents are present
        route = parsed.route
        sub_queries = [sq.model_dump() for sq in parsed.sub_queries]
        if route == "decompose" and len(documents) < 2:
            route = "direct"
            sub_queries = []

        elapsed_ms = (time.time() - t0) * 1000
        timings = dict(state.get("timings") or {})
        timings["rewrite_and_route"] = elapsed_ms

        logger.info(
            f"rewrite_and_route succeeded in {elapsed_ms:.2f}ms: route='{route}', standalone_query='{parsed.standalone_query}' (tokens={tokens_used})"
        )

        return {
            "standalone_query": parsed.standalone_query,
            "route": route,
            "sub_queries": sub_queries,
            "tokens_used": tokens_used,
            "timings": timings,
        }

    except Exception as e:
        elapsed_ms = (time.time() - t0) * 1000
        timings = dict(state.get("timings") or {})
        timings["rewrite_and_route"] = elapsed_ms

        logger.warning(
            f"rewrite_and_route error after {elapsed_ms:.2f}ms, falling back to 'direct' route: {e}"
        )
        return {
            "standalone_query": user_message,
            "route": "direct",
            "sub_queries": [],
            "tokens_used": tokens_used,
            "timings": timings,
            "error": str(e),
        }


async def retrieve_node(state: AgenticState) -> Dict[str, Any]:
    """
    Node 2: retrieve
    Executes retrieval based on state['route']:
    - 'direct': runs standard retrieval pipeline with standalone_query.
    - 'decompose': runs concurrent per-document retrieval via asyncio.gather and guarantees
      each selected document receives at least its own top-k slots so no document starves.
    """
    t0 = time.time()
    route = state.get("route", "direct")
    standalone_query = state.get("standalone_query") or state.get("user_message")
    documents = state.get("documents") or []
    sub_queries = state.get("sub_queries") or []

    rerank_top_n = getattr(settings, "RERANK_TOP_N", 5)
    retrieved_parents: Dict[int, List[Any]] = {}

    if route == "decompose" and len(documents) >= 2:
        # Build document-to-query map
        doc_query_map: Dict[int, str] = {}
        for sq in sub_queries:
            doc_id = sq.get("document_id")
            q = sq.get("query")
            if doc_id is not None and q:
                doc_query_map[int(doc_id)] = q

        # Ensure every selected document has a query (use standalone_query as fallback)
        for d in documents:
            d_id = _get_doc_id(d)
            if d_id not in doc_query_map:
                doc_query_map[d_id] = standalone_query

        # Guarantee every selected document gets at least its own slots (min 2 slots)
        per_doc_slots = max(2, int(math.ceil(rerank_top_n / max(len(documents), 1))))

        def _retrieve_single(doc, query, top_k):
            return _hybrid_retrieve(doc, query, top_k=top_k)

        # Run concurrent single-document retrieval with asyncio.gather across thread workers
        tasks = [
            asyncio.to_thread(_retrieve_single, doc, doc_query_map[_get_doc_id(doc)], per_doc_slots)
            for doc in documents
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for doc, res in zip(documents, results):
            d_id = _get_doc_id(doc)
            if isinstance(res, Exception):
                logger.warning(f"Decompose retrieval error for doc {d_id}: {res}")
                retrieved_parents[d_id] = []
            elif isinstance(res, dict):
                retrieved_parents[d_id] = res.get("hybrid_chunks", [])[:per_doc_slots]
            else:
                retrieved_parents[d_id] = []

    else:
        # Direct route
        if len(documents) == 1:
            doc = documents[0]
            d_id = _get_doc_id(doc)

            def _retrieve_one(d, q, k):
                return _hybrid_retrieve(d, q, top_k=k)

            res = await asyncio.to_thread(_retrieve_one, doc, standalone_query, rerank_top_n)
            chunks = res.get("hybrid_chunks", []) if isinstance(res, dict) else []
            retrieved_parents[d_id] = chunks
        elif len(documents) > 1:
            def _retrieve_multi(docs, q):
                return _multi_doc_retrieve(docs, q, return_details=True)

            ws_details = await asyncio.to_thread(_retrieve_multi, documents, standalone_query)
            final_items = ws_details.get("final_items", []) if isinstance(ws_details, dict) else []
            for item in final_items:
                doc_id = item.get("doc_id")
                retrieved_parents.setdefault(doc_id, []).append(item["chunk"])

    elapsed_ms = (time.time() - t0) * 1000
    timings = dict(state.get("timings") or {})
    timings["retrieve"] = elapsed_ms

    total_chunks = sum(len(v) for v in retrieved_parents.values())
    logger.info(
        f"retrieve completed in {elapsed_ms:.2f}ms (route='{route}'): {total_chunks} chunks retrieved across {len(retrieved_parents)} doc(s)"
    )

    return {
        "retrieved_parents": retrieved_parents,
        "timings": timings,
    }


async def build_context_node(state: AgenticState) -> Dict[str, Any]:
    """
    Node 3: build_context
    Formats the merged parent chunks into the context string and source structure expected
    by the existing generation flow.
    """
    t0 = time.time()
    documents = state.get("documents") or []
    retrieved_parents = state.get("retrieved_parents") or {}

    parts: List[str] = []
    sources: List[Dict[str, Any]] = []

    if len(documents) > 1:
        # Multi-document workspace citation format: [Doc Title · Page X]
        for doc in documents:
            d_id = _get_doc_id(doc)
            d_title = _get_doc_title(doc, default="Document")
            chunks = retrieved_parents.get(d_id, [])
            for c in chunks:
                page_num = getattr(c, "page_number", None)
                page_info = f"Page {page_num}" if page_num is not None else "No page"
                parts.append(f"[{d_title} · {page_info}]: {c.content}")
                sources.append({
                    "doc_id": d_id,
                    "doc_title": d_title,
                    "page_number": page_num,
                    "chunk_id": getattr(c, "id", None),
                    "content": c.content,
                })
    else:
        # Single document citation format: [Page X]
        chunks = list(retrieved_parents.values())[0] if retrieved_parents else []
        for c in chunks:
            page_num = getattr(c, "page_number", None)
            if page_num is not None:
                parts.append(f"[Page {page_num}]: {c.content}")
            else:
                parts.append(c.content)
            sources.append({
                "doc_id": _get_doc_id(documents[0]) if documents else None,
                "doc_title": _get_doc_title(documents[0], default="Document") if documents else "Document",
                "page_number": page_num,
                "chunk_id": getattr(c, "id", None),
                "content": c.content,
            })

    context_text = "\n\n".join(parts)
    elapsed_ms = (time.time() - t0) * 1000
    timings = dict(state.get("timings") or {})
    timings["build_context"] = elapsed_ms

    return {
        "context_text": context_text,
        "sources": sources,
        "timings": timings,
    }


def build_agentic_retrieval_graph():
    """Build and compile the LangGraph StateGraph for agentic retrieval."""
    builder = StateGraph(AgenticState)
    builder.add_node("rewrite_and_route", rewrite_and_route_node)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("build_context", build_context_node)

    builder.set_entry_point("rewrite_and_route")
    builder.add_edge("rewrite_and_route", "retrieve")
    builder.add_edge("retrieve", "build_context")
    builder.add_edge("build_context", END)

    return builder.compile()


_AGENTIC_GRAPH = None


def get_agentic_retrieval_graph():
    """Module-level singleton for compiled LangGraph graph."""
    global _AGENTIC_GRAPH
    if _AGENTIC_GRAPH is None:
        _AGENTIC_GRAPH = build_agentic_retrieval_graph()
    return _AGENTIC_GRAPH


async def run_agentic_retrieval(
    user_message: str,
    documents: List[Any],
    chat_history: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """
    Run agentic retrieval graph end-to-end.
    Returns dict containing context_text, sources, standalone_query, route, sub_queries,
    retrieved_parents, timings, and tokens_used.
    """
    graph = get_agentic_retrieval_graph()
    initial_state: AgenticState = {
        "user_message": user_message,
        "chat_history": chat_history or [],
        "documents": documents,
        "standalone_query": user_message,
        "route": "direct",
        "sub_queries": [],
        "retrieved_parents": {},
        "sources": [],
        "context_text": "",
        "timings": {},
        "tokens_used": 0,
        "error": None,
    }

    result = await graph.ainvoke(initial_state)

    timings = result.get("timings", {})
    route = result.get("route", "direct")
    standalone = result.get("standalone_query", user_message)
    tokens_used = result.get("tokens_used", 0)

    timing_summary = ", ".join(f"{k}={v:.1f}ms" for k, v in timings.items())
    logger.info(
        f"[AgenticRetrieval] route='{route}' standalone='{standalone}' tokens={tokens_used} timings: {timing_summary}"
    )

    return result
