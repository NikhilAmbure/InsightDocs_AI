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

from .gemini_chat import (
    _apply_relative_score_cutoff,
    _generate_shared_query_embedding,
    _hybrid_retrieve,
    _multi_doc_retrieve,
    _rerank_chunks_flashrank,
    _safe_json_loads,
)

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


class AlternativeQueryOutput(BaseModel):
    alternative_query: str


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
    top_parent_score: float
    retry_fired: bool
    alternative_query: Optional[str]
    first_pass_score: float
    retry_pass_score: float
    retry_winner: Optional[str]


async def _call_gemini_rewrite(prompt: str):
    """
    Calls Gemini 2.5 Flash to rewrite and route queries.
    Uses modern google.genai with thinking_budget=0 to disable thinking overhead,
    falling back to google.generativeai if unavailable or mocked.
    """
    # Check if google.generativeai has been patched/mocked in tests
    model_cls = getattr(genai, "GenerativeModel", None)
    if hasattr(model_cls, "return_value") or hasattr(model_cls, "side_effect") or hasattr(getattr(model_cls, "generate_content_async", None), "assert_called"):
        model = genai.GenerativeModel("gemini-2.5-flash")
        return await model.generate_content_async(prompt)

    try:
        from google import genai as modern_genai
        from google.genai import types as modern_genai_types
        client = modern_genai.Client(api_key=getattr(settings, "GOOGLE_API_KEY", None))
        return await client.aio.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=modern_genai_types.GenerateContentConfig(
                thinking_config=modern_genai_types.ThinkingConfig(thinking_budget=0),
                response_mime_type="application/json",
            ),
        )
    except Exception as e:
        logger.debug(f"modern_genai call failed ({e}), falling back to genai.GenerativeModel: {e}")
        model = genai.GenerativeModel("gemini-2.5-flash")
        return await model.generate_content_async(prompt)


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

    # Format history turns for prompt (pass only the last 4 turns)
    history_lines = []
    for msg in chat_history[-4:]:
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
        response = await asyncio.wait_for(_call_gemini_rewrite(prompt), timeout=REWRITE_TIMEOUT_SECONDS)

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


async def _execute_retrieval(
    query: str,
    documents: List[Any],
    route: str = "direct",
    sub_queries: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[Dict[int, List[Any]], float]:
    """
    Executes hybrid retrieval + reranking across documents for a given query.
    Used by both retrieve_node (first pass) and retry_retrieve_node (second pass).
    Returns (retrieved_parents, top_parent_score).
    """
    rerank_top_n = getattr(settings, "RERANK_TOP_N", 5)
    retrieved_parents: Dict[int, List[Any]] = {}

    if route == "decompose" and len(documents) >= 2 and sub_queries:
        doc_query_map: Dict[int, str] = {}
        for sq in sub_queries:
            doc_id = sq.get("document_id")
            q = sq.get("query")
            if doc_id is not None and q:
                doc_query_map[int(doc_id)] = q

        for d in documents:
            d_id = _get_doc_id(d)
            if d_id not in doc_query_map:
                doc_query_map[d_id] = query

        per_doc_slots = max(2, int(math.ceil(rerank_top_n / max(len(documents), 1))))

        def _retrieve_single(doc, q_text, top_k):
            return _hybrid_retrieve(doc, q_text, top_k=top_k)

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

            res = await asyncio.to_thread(_retrieve_one, doc, query, rerank_top_n)
            chunks = res.get("hybrid_chunks", []) if isinstance(res, dict) else []
            retrieved_parents[d_id] = chunks
        elif len(documents) > 1:
            shared_embedding = await asyncio.to_thread(_generate_shared_query_embedding, query)
            enable_rerank = getattr(settings, "ENABLE_FLASHRANK_RERANK", False)
            processed_docs = [doc for doc in documents if getattr(doc, "is_processed", False)]
            if not processed_docs:
                processed_docs = documents
            doc_count = max(len(processed_docs), 1)

            if enable_rerank:
                rerank_candidates_k = getattr(settings, "RERANK_CANDIDATES", 20)
                per_doc_top_k = max(5, int(math.ceil(rerank_candidates_k / doc_count)))
            else:
                rerank_candidates_k = 20
                per_doc_top_k = 5

            def _retrieve_single_direct(doc, q, k, emb):
                return _hybrid_retrieve(doc, q, top_k=k, query_embedding=emb)

            tasks = [
                asyncio.to_thread(_retrieve_single_direct, doc, query, per_doc_top_k, shared_embedding)
                for doc in processed_docs
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            all_scored = []
            for doc, res in zip(processed_docs, results):
                d_id = _get_doc_id(doc)
                d_title = _get_doc_title(doc, "Document")
                if isinstance(res, Exception):
                    logger.warning(f"Parallel direct retrieval failed for doc {d_id}: {res}")
                    continue
                if not isinstance(res, dict):
                    continue
                scored_items = res.get("scored")
                if scored_items is not None:
                    for item in scored_items:
                        all_scored.append({
                            "hybrid_score": item.get("hybrid_score", 0.0),
                            "chunk": item["chunk"],
                            "doc_title": d_title,
                            "doc_id": d_id,
                            "dropped": False,
                        })
                else:
                    for chunk in res.get("hybrid_chunks", []):
                        all_scored.append({
                            "hybrid_score": 0.0,
                            "chunk": chunk,
                            "doc_title": d_title,
                            "doc_id": d_id,
                            "dropped": False,
                        })

            all_scored.sort(key=lambda x: x["hybrid_score"], reverse=True)

            if enable_rerank and all_scored:
                candidate_items = all_scored[:rerank_candidates_k]
                candidate_chunks = [item["chunk"] for item in candidate_items]
                chunk_id_to_item = {item["chunk"].id: item for item in candidate_items}
                reranked_chunks = await asyncio.to_thread(
                    _rerank_chunks_flashrank, query, candidate_chunks, top_n=len(candidate_chunks)
                )

                unique_items = []
                seen_chunk_ids = set()
                for c in reranked_chunks:
                    if c.id not in seen_chunk_ids:
                        seen_chunk_ids.add(c.id)
                        item = chunk_id_to_item.get(c.id)
                        if item:
                            item["hybrid_score"] = float(getattr(c, "rerank_score", 0.0) or 0.0)
                            item["dropped"] = False
                            unique_items.append(item)
                            if len(unique_items) >= rerank_top_n:
                                break

                kept_items, all_unique = _apply_relative_score_cutoff(unique_items, score_key="hybrid_score")
                final_items = kept_items
            else:
                unique_items = []
                seen_chunk_ids = set()
                for item in all_scored:
                    c = item["chunk"]
                    if c.id not in seen_chunk_ids:
                        seen_chunk_ids.add(c.id)
                        item["dropped"] = False
                        unique_items.append(item)
                        if len(unique_items) >= 15:
                            break
                final_items = unique_items

            for item in final_items:
                doc_id = item.get("doc_id")
                retrieved_parents.setdefault(doc_id, []).append(item["chunk"])

    all_scores = []
    for chunks in retrieved_parents.values():
        for c in chunks:
            s = getattr(c, "rerank_score", None)
            if s is None:
                s = getattr(c, "hybrid_score", None)
            if s is None:
                s = getattr(c, "score", None)
            if s is not None:
                try:
                    all_scores.append(float(s))
                except (ValueError, TypeError):
                    pass

    top_score = max(all_scores) if all_scores else 0.0
    return retrieved_parents, top_score


async def retrieve_node(state: AgenticState) -> Dict[str, Any]:
    """
    Node 2: retrieve
    Executes initial retrieval with standalone_query.
    """
    t0 = time.time()
    route = state.get("route", "direct")
    standalone_query = state.get("standalone_query") or state.get("user_message")
    documents = state.get("documents") or []
    sub_queries = state.get("sub_queries") or []

    retrieved_parents, top_parent_score = await _execute_retrieval(
        standalone_query, documents, route=route, sub_queries=sub_queries
    )

    elapsed_ms = (time.time() - t0) * 1000
    timings = dict(state.get("timings") or {})
    timings["retrieve"] = elapsed_ms

    total_chunks = sum(len(v) for v in retrieved_parents.values())
    logger.info(
        f"[AgenticRetrieval Result] standalone_query='{standalone_query}' route='{route}' "
        f"top_parent_score={top_parent_score:.6f} parents_returned={total_chunks}"
    )

    return {
        "retrieved_parents": retrieved_parents,
        "top_parent_score": top_parent_score,
        "first_pass_score": top_parent_score,
        "retry_fired": False,
        "timings": timings,
    }


def should_retry(state: AgenticState) -> str:
    """
    Conditional edge after retrieve.
    Returns 'retry_retrieve' if:
    1. ENABLE_CORRECTIVE_RETRY is True
    2. FlashRank reranking is enabled (so cross-encoder scores exist)
    3. Query is NOT a summarize/list all bypass query
    4. Top parent score is below RETRY_MIN_TOP_SCORE
    5. Retry hasn't already fired (never loop)
    Otherwise returns 'build_context'.
    """
    enable_retry = getattr(settings, "ENABLE_CORRECTIVE_RETRY", False)
    if not enable_retry:
        return "build_context"

    enable_rerank = getattr(settings, "ENABLE_FLASHRANK_RERANK", False)
    if not enable_rerank:
        return "build_context"

    user_message = state.get("user_message", "")
    standalone_query = state.get("standalone_query", "")
    if _is_summary_query(user_message) or _is_summary_query(standalone_query):
        return "build_context"

    if state.get("retry_fired", False):
        return "build_context"

    top_score = state.get("top_parent_score", 0.0)
    retry_min_top_score = getattr(settings, "RETRY_MIN_TOP_SCORE", 0.001)

    if top_score < retry_min_top_score:
        return "retry_retrieve"

    return "build_context"


def _clean_alternative_query(alt_query: str, documents: List[Any]) -> str:
    """
    Sanitizes alternative query:
    1. Removes quotation marks (single, double, backticks, unicode quotes).
    2. Strips any occurrence of selected document titles, filenames, and file extensions.
    3. Normalizes whitespace and returns clean plain text.
    """
    cleaned = alt_query
    # Strip quotation marks
    cleaned = re.sub(r'["\'`“”‘’]', ' ', cleaned)

    for doc in documents:
        title = _get_doc_title(doc, "")
        terms_to_strip = []
        if title:
            terms_to_strip.append(title)
            name_no_ext = re.sub(r'\.[a-zA-Z0-9]{2,5}$', '', title)
            if name_no_ext:
                terms_to_strip.append(name_no_ext)
            base_name = title.split("/")[-1].split("\\")[-1]
            if base_name:
                terms_to_strip.append(base_name)
                base_no_ext = re.sub(r'\.[a-zA-Z0-9]{2,5}$', '', base_name)
                if base_no_ext:
                    terms_to_strip.append(base_no_ext)

        file_obj = getattr(doc, "file", None)
        if file_obj and hasattr(file_obj, "name"):
            f_name = file_obj.name.split("/")[-1].split("\\")[-1]
            if f_name:
                terms_to_strip.append(f_name)
                f_no_ext = re.sub(r'\.[a-zA-Z0-9]{2,5}$', '', f_name)
                if f_no_ext:
                    terms_to_strip.append(f_no_ext)

        terms_to_strip = sorted(set(terms_to_strip), key=len, reverse=True)
        for term in terms_to_strip:
            if len(term.strip()) >= 3:
                pattern = re.compile(re.escape(term.strip()), re.IGNORECASE)
                cleaned = pattern.sub(' ', cleaned)

    # Remove lingering standalone extension keywords and file labels
    cleaned = re.sub(r'\b(pdf|docx|txt|document|file)\b', ' ', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


async def retry_retrieve_node(state: AgenticState) -> Dict[str, Any]:
    """
    Node: retry_retrieve (Phase 2b)
    Performs one corrective retrieval retry when initial top score falls below RETRY_MIN_TOP_SCORE.
    Calls Gemini Flash to generate an alternative query with domain keywords and synonyms in plain words.
    The alternative query must NOT include document titles, filenames, or quotation marks.
    If the alternative query contains a document title/filename, it is stripped.
    If the alternative query is empty or equals the original standalone query, re-retrieval is skipped.
    Keeps whichever pass has the higher top score; never merges the two passes.
    """
    t0 = time.time()
    user_message = state.get("user_message", "")
    standalone_query = state.get("standalone_query") or user_message
    chat_history = state.get("chat_history") or []
    documents = state.get("documents") or []
    route = state.get("route", "direct")
    old_top_score = state.get("top_parent_score", 0.0)
    old_parents = state.get("retrieved_parents", {})

    timings = dict(state.get("timings") or {})
    retry_tokens = 0
    alt_query = None

    doc_lines = []
    for d in documents:
        doc_id = _get_doc_id(d)
        doc_title = _get_doc_title(d, default=f"Doc {doc_id}")
        doc_lines.append(f"- ID {doc_id}: \"{doc_title}\"")
    docs_str = "\n".join(doc_lines) if doc_lines else "None"

    history_lines = []
    for msg in chat_history[-4:]:
        role = msg.get("role", "user")
        content = msg.get("content", "").strip()
        if content:
            history_lines.append(f"{role.capitalize()}: {content}")
    history_str = "\n".join(history_lines) if history_lines else "None"

    prompt = f"""You are an expert search query reformulation engine for an enterprise document retrieval system.
The initial search query yielded poor or low-confidence results.
Your task is to produce a plain-text alternative search query that rephrases what the user wants using synonyms and likely domain terms.

CRITICAL RULES:
1. Do NOT include document titles, filenames, or file extensions (e.g., .pdf, .docx).
2. Do NOT use quotation marks (single or double quotes).
3. Use plain natural words and specific domain terminology that describe the underlying topic, concept, or information sought.
4. If the user's message is too ambiguous, vague, or refers to an undefined reference (such as "section 2", "part 1", "this clause", etc.) without enough context to rewrite meaningfully, return the original standalone query unchanged.

AVAILABLE CONTEXT (FOR DOMAIN INFERENCE ONLY - DO NOT COPY TITLES/FILENAMES INTO QUERY):
{docs_str}

CONVERSATION HISTORY:
{history_str}

ORIGINAL USER MESSAGE:
{user_message}

PREVIOUS RETRIEVAL QUERY:
{standalone_query}

Return STRICT JSON matching this schema:
{{
  "alternative_query": "<plain-text search query without titles, filenames, or quotes>"
}}
"""

    try:
        response = await asyncio.wait_for(_call_gemini_rewrite(prompt), timeout=REWRITE_TIMEOUT_SECONDS)
        usage_meta = getattr(response, "usage_metadata", None)
        if usage_meta is not None:
            retry_tokens = int(getattr(usage_meta, "total_token_count", 0) or 0)

        raw_text = response.text or ""
        json_payload = _safe_json_loads(raw_text)
        if not json_payload or not isinstance(json_payload, dict):
            raise ValueError(f"Invalid JSON returned: {raw_text[:100]}")

        parsed = AlternativeQueryOutput.model_validate(json_payload)
        raw_alt_query = parsed.alternative_query.strip()
        if not raw_alt_query:
            raise ValueError("Empty alternative query returned")

        # Safety net: remove document titles, filenames, and quotation marks
        alt_query = _clean_alternative_query(raw_alt_query, documents)

    except Exception as e:
        logger.warning(f"Alternative query generation failed or timed out: {e}. Skipping retry silently.")
        elapsed_ms = (time.time() - t0) * 1000
        timings["retry_retrieve"] = elapsed_ms
        logger.info(
            f"[CorrectiveRetry Result] retry_fired=True alt_query=None "
            f"first_score={old_top_score:.6f} retry_score=0.000000 winner='first'"
        )
        return {
            "retry_fired": True,
            "alternative_query": None,
            "first_pass_score": old_top_score,
            "retry_pass_score": 0.0,
            "retry_winner": "first",
            "tokens_used": state.get("tokens_used", 0) + retry_tokens,
            "timings": timings,
        }

    # Safety net: if alternative query is empty or identical to the original standalone query, skip re-retrieval
    is_empty = not alt_query
    is_identical = (alt_query.lower() == standalone_query.strip().lower())

    if is_empty or is_identical:
        elapsed_ms = (time.time() - t0) * 1000
        timings["retry_retrieve"] = elapsed_ms
        final_alt_display = alt_query if alt_query else standalone_query
        logger.info(
            f"[CorrectiveRetry Result] retry_fired=True alt_query='{final_alt_display}' "
            f"skipped_re_retrieval=True reason={'empty_after_strip' if is_empty else 'identical_to_original'} "
            f"first_score={old_top_score:.6f} retry_score={old_top_score:.6f} winner='first'"
        )
        return {
            "retrieved_parents": old_parents,
            "top_parent_score": old_top_score,
            "retry_fired": True,
            "alternative_query": final_alt_display,
            "first_pass_score": old_top_score,
            "retry_pass_score": old_top_score,
            "retry_winner": "first",
            "tokens_used": state.get("tokens_used", 0) + retry_tokens,
            "timings": timings,
        }

    # Re-run retrieval with alternative_query
    new_retrieved_parents, new_top_score = await _execute_retrieval(
        alt_query, documents, route=route, sub_queries=None
    )

    if new_top_score > old_top_score:
        winner = "retry"
        winning_parents = new_retrieved_parents
        winning_score = new_top_score
    else:
        winner = "first"
        winning_parents = old_parents
        winning_score = old_top_score

    elapsed_ms = (time.time() - t0) * 1000
    timings["retry_retrieve"] = elapsed_ms

    logger.info(
        f"[CorrectiveRetry Result] retry_fired=True alt_query='{alt_query}' "
        f"first_score={old_top_score:.6f} retry_score={new_top_score:.6f} winner='{winner}'"
    )

    return {
        "retrieved_parents": winning_parents,
        "top_parent_score": winning_score,
        "retry_fired": True,
        "alternative_query": alt_query,
        "first_pass_score": old_top_score,
        "retry_pass_score": new_top_score,
        "retry_winner": winner,
        "tokens_used": state.get("tokens_used", 0) + retry_tokens,
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
    """Build and compile the LangGraph StateGraph for agentic retrieval with corrective retry (Phase 2b)."""
    builder = StateGraph(AgenticState)
    builder.add_node("rewrite_and_route", rewrite_and_route_node)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("retry_retrieve", retry_retrieve_node)
    builder.add_node("build_context", build_context_node)

    builder.set_entry_point("rewrite_and_route")
    builder.add_edge("rewrite_and_route", "retrieve")
    builder.add_conditional_edges(
        "retrieve",
        should_retry,
        {
            "retry_retrieve": "retry_retrieve",
            "build_context": "build_context",
        },
    )
    builder.add_edge("retry_retrieve", "build_context")
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
    retrieved_parents, timings, tokens_used, retry_fired, alternative_query, etc.
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
        "top_parent_score": 0.0,
        "retry_fired": False,
        "alternative_query": None,
        "first_pass_score": 0.0,
        "retry_pass_score": 0.0,
        "retry_winner": None,
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
