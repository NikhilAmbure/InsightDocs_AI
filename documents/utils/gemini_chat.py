import asyncio
import json
import logging
import math
import os
import re
import requests
import sys
import threading
import time

import google.generativeai as genai
from django.conf import settings
from django.contrib.postgres.search import SearchQuery, SearchRank, SearchVector
from django.db.models import F
from google.api_core.exceptions import ResourceExhausted
from pgvector.django import L2Distance

from ..models import DocumentChunk, ChildChunk

logger = logging.getLogger(__name__)

genai.configure(api_key=settings.GOOGLE_API_KEY)

TOP_K_PER_STEP = 5
MAX_FALLBACK_CHUNKS = 20
FALLBACK_CHUNK_SIZE = 180
GEMINI_COOLDOWN_UNTIL = 0.0

# Polished user-facing messages (no leaked developer configs, .env variables, or technical internals)
USER_FRIENDLY_RATE_LIMIT_MSG = "The AI service is currently experiencing high demand. Please wait a moment and try your question again."
USER_FRIENDLY_ERROR_MSG = "I was unable to complete this response due to temporary service load. Please try your question again in a moment."
DOCUMENT_UNREADABLE_MSG = "The document could not be read right now. Please try again in a moment."
WORKSPACE_DOCUMENTS_UNREADABLE_MSG = "The workspace documents could not be read right now. Please try again in a moment."

try:
    from sentence_transformers import CrossEncoder  # pyright: ignore[reportMissingImports]
except Exception:
    CrossEncoder = None

_CROSS_ENCODER = None
# Disabled by default to protect server memory (prevents 800MB-1.2GB PyTorch allocation per worker)
_CROSS_ENCODER_DISABLED = True




def _extract_retry_seconds(error_text):
    if not error_text:
        return 60
    match = re.search(r"seconds:\s*(\d+)", error_text)
    if match:
        try:
            return max(int(match.group(1)), 5)
        except ValueError:
            return 60
    return 60


def _set_gemini_cooldown_from_error(error_text):
    global GEMINI_COOLDOWN_UNTIL
    retry_after = _extract_retry_seconds(error_text)
    GEMINI_COOLDOWN_UNTIL = time.time() + retry_after
    logger.warning(f"Gemini cooldown enabled for ~{retry_after}s due to quota/rate limits.")


def _is_gemini_in_cooldown():
    return time.time() < GEMINI_COOLDOWN_UNTIL


def _safe_json_loads(text):
    if not text:
        return None
    raw = text.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()
    return json.loads(raw)


def _plan_query_steps(user_message):
    model = genai.GenerativeModel("gemini-2.5-flash")
    planner_prompt = f"""
You are a query planner for document QA.
Return STRICT JSON: {{"queries": ["...", "..."]}}
Rules:
- Include 1-3 focused retrieval queries.
- First query should be close to the original user intent.
- Keep each query under 14 words.
- Return JSON only.

USER QUESTION:
{user_message}
"""
    response = model.generate_content(planner_prompt)
    payload = _safe_json_loads(response.text)
    queries = payload.get("queries", []) if isinstance(payload, dict) else []
    cleaned = [q.strip() for q in queries if isinstance(q, str) and q.strip()]
    return cleaned[:3] or [user_message]

def _track_usage(usage_acc, response):
    """Append total_token_count from a Gemini response's usage_metadata, if present."""
    if usage_acc is None:
        return
    usage = getattr(response, "usage_metadata", None)
    if usage is not None:
        usage_acc.append(getattr(usage, "total_token_count", 0) or 0)

def _get_query_embedding(query_text):
    result = genai.embed_content(
        model="models/gemini-embedding-001",
        content=query_text,
        task_type="retrieval_query",
        output_dimensionality=768,
    )
    return result["embedding"]


def _hybrid_retrieve_legacy(document, query_text, top_k=TOP_K_PER_STEP, query_embedding=None):
    if query_embedding is None:
        query_embedding = _get_query_embedding(query_text)

    # 1. Sub-10ms HNSW approximate nearest neighbor search via pgvector
    vector_candidates = list(
        DocumentChunk.objects.filter(document=document)
        .annotate(distance=L2Distance("embedding", query_embedding))
        .order_by("distance")[: top_k * 2]
    )

    # 2. Fast keyword retrieval
    keyword_candidates = []
    try:
        # Extract keywords longer than 3 characters, filtering out common stop words
        words = [w for w in re.findall(r'\b\w{3,}\b', query_text) if w.lower() not in {
            'what', 'when', 'where', 'which', 'that', 'this', 'from', 'with', 'about', 'explain', 'tell', 'show'
        }]
        if words:
            from django.db.models import Q
            q_filter = Q()
            for w in words[:3]:
                q_filter |= Q(content__icontains=w)
            keyword_candidates = list(
                DocumentChunk.objects.filter(document=document).filter(q_filter)[: top_k * 2]
            )
    except Exception as kw_err:
        logger.debug(f"Keyword search skipped for doc {document.id}: {kw_err}")

    combined = {}
    for chunk in vector_candidates:
        combined.setdefault(
            chunk.id,
            {"chunk": chunk, "vector_score": 0.0, "keyword_score": 0.0},
        )
        distance = float(getattr(chunk, "distance", 1.0) or 1.0)
        combined[chunk.id]["vector_score"] = 1.0 / (1.0 + max(distance, 0.0))

    for chunk in keyword_candidates:
        combined.setdefault(
            chunk.id,
            {"chunk": chunk, "vector_score": 0.0, "keyword_score": 0.0},
        )
        combined[chunk.id]["keyword_score"] = 0.5

    scored = list(combined.values())
    for row in scored:
        row["hybrid_score"] = (0.75 * row["vector_score"]) + (0.25 * row["keyword_score"])

    scored.sort(key=lambda x: x["hybrid_score"], reverse=True)
    return {
        "vector_chunks": vector_candidates[:top_k],
        "keyword_chunks": keyword_candidates[:top_k],
        "hybrid_chunks": [row["chunk"] for row in scored[:top_k]],
        "scored": [
            {"chunk": row["chunk"], "hybrid_score": row["hybrid_score"]}
            for row in scored[:top_k]
        ],
    }


def _hybrid_retrieve_rrf(document, query_text, top_k=TOP_K_PER_STEP, query_embedding=None):
    if query_embedding is None:
        query_embedding = _get_query_embedding(query_text)

    # 1. Dense branch: L2Distance on HNSW vector index
    vector_candidates = list(
        DocumentChunk.objects.filter(document=document)
        .annotate(distance=L2Distance("embedding", query_embedding))
        .order_by("distance")[: top_k * 2]
    )

    # 2. Lexical branch: PostgreSQL full-text search with OR-combined terms
    words = [w for w in re.findall(r'\b\w{3,}\b', query_text) if w.lower() not in {
        'what', 'when', 'where', 'which', 'that', 'this', 'from', 'with', 'about', 'explain', 'tell', 'show'
    }]
    keyword_candidates = []
    if words:
        try:
            search_config = getattr(settings, "SEARCH_CONFIG", "english")
            combined_query = None
            for w in words:
                sq = SearchQuery(w, config=search_config)
                combined_query = sq if combined_query is None else combined_query | sq

            if combined_query is not None:
                keyword_candidates = list(
                    DocumentChunk.objects.filter(document=document)
                    .filter(search_vector=combined_query)
                    .annotate(rank=SearchRank(F("search_vector"), combined_query))
                    .order_by("-rank")[: top_k * 2]
                )
        except Exception as lex_err:
            logger.warning(f"PostgreSQL FTS search skipped for doc {document.id}: {lex_err}")

    # 3. Plain unweighted RRF: 1 / (RRF_K + rank)
    rrf_k = getattr(settings, "RRF_K", 60)
    scores = {}

    for rank, chunk in enumerate(vector_candidates, start=1):
        scores.setdefault(
            chunk.id,
            {"chunk": chunk, "dense_rank": None, "lexical_rank": None, "rrf_score": 0.0},
        )
        scores[chunk.id]["dense_rank"] = rank
        scores[chunk.id]["rrf_score"] += 1.0 / (rrf_k + rank)

    for rank, chunk in enumerate(keyword_candidates, start=1):
        scores.setdefault(
            chunk.id,
            {"chunk": chunk, "dense_rank": None, "lexical_rank": None, "rrf_score": 0.0},
        )
        scores[chunk.id]["lexical_rank"] = rank
        scores[chunk.id]["rrf_score"] += 1.0 / (rrf_k + rank)

    scored = list(scores.values())
    scored.sort(key=lambda x: x["rrf_score"], reverse=True)

    return {
        "vector_chunks": vector_candidates[:top_k],
        "keyword_chunks": keyword_candidates[:top_k],
        "hybrid_chunks": [row["chunk"] for row in scored[:top_k]],
        "scored": [
            {
                "chunk": row["chunk"],
                "hybrid_score": row["rrf_score"],
                "dense_rank": row["dense_rank"],
                "lexical_rank": row["lexical_rank"],
            }
            for row in scored[:top_k]
        ],
    }


_FLASHRANK_LOCK = threading.Lock()
_FLASHRANK_RANKER = None


def _is_celery_process():
    if not sys.argv:
        return False
    binary = os.path.basename(sys.argv[0]).lower()
    if "celery" in binary:
        return True
    if len(sys.argv) > 1 and sys.argv[1].lower() == "worker" and "celery" in "".join(sys.argv[:3]).lower():
        return True
    return False


def _get_flashrank_ranker():
    global _FLASHRANK_RANKER
    if _is_celery_process():
        return None
    if _FLASHRANK_RANKER is not None:
        return _FLASHRANK_RANKER
    with _FLASHRANK_LOCK:
        if _FLASHRANK_RANKER is not None:
            return _FLASHRANK_RANKER
        try:
            from flashrank import Ranker
            base_dir = getattr(settings, "BASE_DIR", None)
            default_cache = os.path.join(str(base_dir), ".flashrank_cache") if base_dir else ".flashrank_cache"
            cache_dir = getattr(settings, "FLASHRANK_CACHE_DIR", default_cache)
            os.makedirs(cache_dir, exist_ok=True)
            _FLASHRANK_RANKER = Ranker(
                model_name="ms-marco-MiniLM-L-12-v2",
                cache_dir=str(cache_dir),
            )
            return _FLASHRANK_RANKER
        except Exception as e:
            logger.warning(f"FlashRank Ranker initialization failed: {e}")
            return None


def _rerank_chunks_flashrank(query, chunks, top_n=5):
    """
    Rerank chunks using FlashRank local cross-encoder (ms-marco-MiniLM-L-12-v2).
    Returns up to top_n chunks, each carrying its rerank_score attribute.
    Falls back to original order on error or unavailability.
    """
    if not chunks:
        return []

    ranker = _get_flashrank_ranker()
    if ranker is None:
        logger.warning("FlashRank Ranker unavailable, falling back to original retrieval order.")
        return chunks[:top_n]

    try:
        from flashrank import RerankRequest
        t0 = time.time()
        passages = [
            {"id": idx, "text": chunk.content}
            for idx, chunk in enumerate(chunks)
        ]
        request = RerankRequest(query=query, passages=passages)
        results = ranker.rerank(request)
        latency_ms = (time.time() - t0) * 1000

        id_to_chunk = {idx: chunk for idx, chunk in enumerate(chunks)}
        reranked_chunks = []
        scores_for_log = []

        for item in results[:top_n]:
            idx = item["id"]
            chunk = id_to_chunk[idx]
            score = float(item["score"])
            setattr(chunk, "rerank_score", score)
            reranked_chunks.append(chunk)
            scores_for_log.append(f"chunk_{getattr(chunk, 'id', idx)}: {score:.4f}")

        logger.debug(
            f"FlashRank rerank completed in {latency_ms:.2f}ms for {len(chunks)} candidates -> top {len(reranked_chunks)}. Scores: {', '.join(scores_for_log)}"
        )
        return reranked_chunks
    except Exception as e:
        logger.warning(f"FlashRank rerank error, falling back to original retrieval order: {e}")
        return chunks[:top_n]


def _apply_relative_score_cutoff(scored_items, score_key="hybrid_score", min_relative_score=None):
    """
    Drop any item whose score is below min_relative_score * top_score,
    but always keep at least 2 items (or 1 if only 1 exists).
    Sets item['dropped'] = True/False.
    Returns (kept_items, all_items_with_dropped_flag).
    """
    if not scored_items:
        return [], []

    if min_relative_score is None:
        min_relative_score = getattr(settings, "RERANK_MIN_RELATIVE_SCORE", 0.01)

    # If cutoff is disabled (<= 0) or 2 or fewer items, keep all
    if min_relative_score <= 0 or len(scored_items) <= 2:
        for item in scored_items:
            item["dropped"] = False
        return list(scored_items), list(scored_items)

    top_score = scored_items[0].get(score_key, 0.0)
    cutoff = top_score * min_relative_score if top_score > 0 else 0.0

    kept = []
    for idx, item in enumerate(scored_items):
        score = item.get(score_key, 0.0)
        # Always keep the first 2 items (indices 0 and 1)
        if idx < 2:
            item["dropped"] = False
            kept.append(item)
        else:
            if cutoff > 0 and score < cutoff:
                item["dropped"] = True
            else:
                item["dropped"] = False
                kept.append(item)

    return kept, scored_items


def _parent_child_retrieve(document, query_text, top_k=None, query_embedding=None):
    """
    Retrieve chunks via parent-child chunking.
    1. Runs dense (L2Distance on HNSW) and lexical (Postgres FTS) on ChildChunk.
    2. Merges with unweighted RRF to get top RERANK_CANDIDATES children.
    3. If ENABLE_FLASHRANK_RERANK is on, reranks the short child texts.
    4. Walks down the ordered children until RERANK_TOP_N (or top_k) UNIQUE parents are found.
    5. Applies relative score cutoff (RERANK_MIN_RELATIVE_SCORE) when reranking is on.
    """
    if query_embedding is None:
        query_embedding = _get_query_embedding(query_text)

    rerank_candidates_k = getattr(settings, "RERANK_CANDIDATES", 20)
    rerank_top_n = getattr(settings, "RERANK_TOP_N", 5)
    limit_n = top_k if top_k is not None else rerank_top_n

    # 1. Dense branch: L2Distance on HNSW vector index of ChildChunk
    child_vector_candidates = list(
        ChildChunk.objects.filter(document=document)
        .select_related("parent")
        .annotate(distance=L2Distance("embedding", query_embedding))
        .order_by("distance")[: rerank_candidates_k * 2]
    )

    # 2. Lexical branch: PostgreSQL full-text search with OR-combined terms on ChildChunk.search_vector
    words = [w for w in re.findall(r'\b\w{3,}\b', query_text) if w.lower() not in {
        'what', 'when', 'where', 'which', 'that', 'this', 'from', 'with', 'about', 'explain', 'tell', 'show'
    }]
    child_keyword_candidates = []
    if words:
        try:
            search_config = getattr(settings, "SEARCH_CONFIG", "english")
            combined_query = None
            for w in words:
                sq = SearchQuery(w, config=search_config)
                combined_query = sq if combined_query is None else combined_query | sq

            if combined_query is not None:
                child_keyword_candidates = list(
                    ChildChunk.objects.filter(document=document)
                    .filter(search_vector=combined_query)
                    .select_related("parent")
                    .annotate(rank=SearchRank(F("search_vector"), combined_query))
                    .order_by("-rank")[: rerank_candidates_k * 2]
                )
        except Exception as lex_err:
            logger.warning(f"PostgreSQL FTS search on ChildChunk skipped for doc {document.id}: {lex_err}")

    # 3. Plain unweighted RRF on ChildChunk: 1 / (RRF_K + rank)
    rrf_k = getattr(settings, "RRF_K", 60)
    child_scores = {}

    for rank, child in enumerate(child_vector_candidates, start=1):
        child_scores.setdefault(
            child.id,
            {"child": child, "dense_rank": None, "lexical_rank": None, "rrf_score": 0.0},
        )
        child_scores[child.id]["dense_rank"] = rank
        child_scores[child.id]["rrf_score"] += 1.0 / (rrf_k + rank)

    for rank, child in enumerate(child_keyword_candidates, start=1):
        child_scores.setdefault(
            child.id,
            {"child": child, "dense_rank": None, "lexical_rank": None, "rrf_score": 0.0},
        )
        child_scores[child.id]["lexical_rank"] = rank
        child_scores[child.id]["rrf_score"] += 1.0 / (rrf_k + rank)

    scored_children = list(child_scores.values())
    scored_children.sort(key=lambda x: x["rrf_score"], reverse=True)

    # Take top RERANK_CANDIDATES children
    candidate_children_pool = scored_children[:rerank_candidates_k]

    # 4. Optional FlashRank cross-encoder reranking on child texts
    enable_rerank = getattr(settings, "ENABLE_FLASHRANK_RERANK", False)

    if enable_rerank and candidate_children_pool:
        child_objs = [item["child"] for item in candidate_children_pool]
        _rerank_chunks_flashrank(query_text, child_objs, top_n=len(child_objs))

        for item in candidate_children_pool:
            score = getattr(item["child"], "rerank_score", None)
            item["rerank_score"] = float(score) if isinstance(score, (int, float)) else None

        candidate_children_pool.sort(
            key=lambda x: x["rerank_score"] if x["rerank_score"] is not None else -9999.0,
            reverse=True,
        )
    else:
        for item in candidate_children_pool:
            item["rerank_score"] = None

    # 5. Walk down ordered children until limit_n UNIQUE parents are collected
    ordered_children = candidate_children_pool
    unique_parents = []
    seen_parent_ids = set()

    for item in ordered_children:
        child = item["child"]
        parent = child.parent
        if parent.id not in seen_parent_ids:
            seen_parent_ids.add(parent.id)
            score = item["rerank_score"] if (enable_rerank and item["rerank_score"] is not None) else item["rrf_score"]
            parent_entry = {
                "chunk": parent,
                "hybrid_score": score,
                "dense_rank": item["dense_rank"],
                "lexical_rank": item["lexical_rank"],
                "rrf_score": item["rrf_score"],
                "rerank_score": item["rerank_score"],
                "child": child,
                "dropped": False,
            }
            if item["rerank_score"] is not None:
                setattr(parent, "rerank_score", item["rerank_score"])
            unique_parents.append(parent_entry)
            if len(unique_parents) >= limit_n:
                break

    # 6. Apply relative score cutoff (only when reranking is on)
    if enable_rerank:
        kept_parents, all_scored_parents = _apply_relative_score_cutoff(
            unique_parents, score_key="hybrid_score"
        )
    else:
        kept_parents, all_scored_parents = unique_parents, unique_parents

    # Build unique parent lists for vector and keyword branches for return shape compatibility
    seen_v = set()
    vector_parents = []
    for c in child_vector_candidates:
        if c.parent_id not in seen_v:
            seen_v.add(c.parent_id)
            vector_parents.append(c.parent)

    seen_k = set()
    keyword_parents = []
    for c in child_keyword_candidates:
        if c.parent_id not in seen_k:
            seen_k.add(c.parent_id)
            keyword_parents.append(c.parent)

    return {
        "vector_chunks": vector_parents[:limit_n],
        "keyword_chunks": keyword_parents[:limit_n],
        "hybrid_chunks": [row["chunk"] for row in kept_parents],
        "scored": all_scored_parents,
        "child_candidates": candidate_children_pool,
    }


def _has_child_chunks(document):
    """Check if document has indexed ChildChunk records. Safe in tests and un-persisted environments."""
    try:
        doc_id = getattr(document, "id", None) or getattr(document, "pk", None)
        if not isinstance(doc_id, int):
            return False
        return ChildChunk.objects.filter(document_id=doc_id).exists()
    except Exception:
        return False


def _hybrid_retrieve(document, query_text, top_k=TOP_K_PER_STEP, query_embedding=None):
    if _has_child_chunks(document):
        return _parent_child_retrieve(document, query_text, top_k=top_k, query_embedding=query_embedding)
    if getattr(settings, "ENABLE_RRF_HYBRID_SEARCH", False):
        return _hybrid_retrieve_rrf(document, query_text, top_k=top_k, query_embedding=query_embedding)
    return _hybrid_retrieve_legacy(document, query_text, top_k=top_k, query_embedding=query_embedding)


def _get_cross_encoder():
    global _CROSS_ENCODER
    global _CROSS_ENCODER_DISABLED
    if _CROSS_ENCODER_DISABLED:
        return None
    if CrossEncoder is None:
        _CROSS_ENCODER_DISABLED = True
        return None
    if _CROSS_ENCODER is None:
        try:
            _CROSS_ENCODER = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        except Exception as e:
            logger.warning(f"Cross-encoder init failed, will use Gemini reranker fallback: {e}")
            _CROSS_ENCODER_DISABLED = True
            return None
    return _CROSS_ENCODER


def _rerank_chunks_cross_encoder(user_message, chunks):
    if not chunks:
        return []
    cross_encoder = _get_cross_encoder()
    if cross_encoder is None:
        return None
    try:
        pairs = [[user_message, chunk.content] for chunk in chunks]
        scores = cross_encoder.predict(pairs)
        ranked = sorted(zip(chunks, scores), key=lambda row: float(row[1]), reverse=True)
        return [chunk for chunk, _ in ranked]
    except Exception as e:
        logger.warning(f"Cross-encoder scoring failed, using Gemini reranker fallback: {e}")
        return None


def _rerank_chunks_gemini(user_message, chunks):
    if not chunks:
        return []

    model = genai.GenerativeModel("gemini-2.5-flash")
    formatted_chunks = []
    for idx, chunk in enumerate(chunks, start=1):
        page = f"Page {chunk.page_number}" if chunk.page_number is not None else "No page"
        formatted_chunks.append(f"{idx}. ({page}) {chunk.content}")

    rerank_prompt = f"""
You are a reranker. Score each chunk for relevance to the question.
Return STRICT JSON:
{{"scores": [{{"index": 1, "score": 0}}, {{"index": 2, "score": 100}}]}}
Rules:
- Score all chunks.
- Higher score means more relevant.
- Return JSON only.

QUESTION:
{user_message}

CHUNKS:
{chr(10).join(formatted_chunks)}
"""
    response = model.generate_content(rerank_prompt)
    payload = _safe_json_loads(response.text)
    scores = payload.get("scores", []) if isinstance(payload, dict) else []
    score_by_index = {}
    for row in scores:
        try:
            idx = int(row.get("index"))
            score = float(row.get("score", 0))
            score_by_index[idx] = score
        except (AttributeError, TypeError, ValueError):
            continue

    ranked = sorted(
        enumerate(chunks, start=1),
        key=lambda pair: score_by_index.get(pair[0], 0.0),
        reverse=True,
    )
    return [chunk for _, chunk in ranked] or chunks


def _compress_context(user_message, chunks):
    if not chunks:
        return ""

    model = genai.GenerativeModel("gemini-2.5-flash")
    context_parts = []
    for c in chunks:
        if c.page_number is not None:
            context_parts.append(f"[Page {c.page_number}] {c.content}")
        else:
            context_parts.append(c.content)

    prompt = f"""
You are a contextual compression module for RAG.
Compress the context specifically for answering the user question.
Requirements:
- Keep document-grounded facts only.
- Preserve page references like [Page X] when present.
- Keep it concise but complete.
- Do not invent facts.

QUESTION:
{user_message}

CONTEXT:
{chr(10).join(context_parts)}
"""
    response = model.generate_content(prompt)
    return (response.text or "").strip()


def _build_system_parts(base_instruction, context_text):
    if context_text:
        return [f"{base_instruction}\n\nDOCUMENT CONTEXT:\n{context_text}"]
    return [
        base_instruction,
        (
            "Note: The document has not finished processing yet or could not be read. "
            "Tell the user to try again shortly."
        ),
    ]


def _build_history_messages(system_parts, chat_history):
    messages = [
        {"role": "system", "content": "\n\n".join(system_parts)},
    ]
    for msg in chat_history:
        role = "user" if msg.get("role") == "user" else "assistant"
        content = msg.get("content", "").strip()
        if content:
            messages.append({"role": role, "content": content})
    return messages


def _build_base_instruction(is_global_query=False):
    base_instruction = (
        "You are InsightDocs AI, an intelligent document assistant.\n\n"
        "You may use THREE sources of knowledge in priority order:\n"
        "1. The provided document context (highest priority)\n"
        "2. General knowledge (if the document does not contain the answer)\n"
        "3. Logical reasoning based on the user's question\n\n"
        "Rules:\n"
        "- If the answer is found in the document context, answer strictly from it.\n"
        "- Include inline citations using [Page X] where applicable.\n"
        "- If the document does NOT contain the answer, answer using general knowledge.\n"
        "- If using general knowledge, explicitly say: "
        "'This answer is based on general knowledge, not the document.'\n"
        "- Do NOT hallucinate document-specific facts.\n\n"
        "Formatting & Structure Guidelines:\n"
        "- Structure your answers using clean, readable Markdown layout.\n"
        "- Use relevant level-3 headers (###) to separate distinct topics, sections, or categories. Do not use H1 or H2.\n"
        "- Use bolding (**word**) to highlight critical names, terms, metrics, numbers, or section labels.\n"
        "- Present lists using bullet points (- ) or numbered lists (1. ) instead of dense, multi-sentence paragraphs.\n"
        "- When comparing data, key parameters, or presenting structured lists of metrics/properties, organize them in a clean Markdown table.\n"
        "- Wrap direct quotes or critical definitions from the document in blockquotes (> quote).\n"
        "- For programming code, commands, database queries, or structured JSON configurations, wrap them in code blocks specifying the language name (e.g. ```python, ```sql).\n"
        "- Keep paragraphs short and concise (2-3 sentences max) to improve readability.\n"
    )
    if is_global_query:
        base_instruction += (
            "- The user is asking for a comprehensive list or summary. "
            "You must be EXHAUSTIVE and list EVERY single item, question, or detail found in the context. "
            "Do NOT truncate the list, omit items, or skip numbers. List all of them completely.\n"
        )
    else:
        base_instruction += "- Be concise, clear, and helpful.\n"
    return base_instruction


def _get_document_fallback_context(document):
    if not getattr(document, "is_processed", False):
        return None
    chunks = DocumentChunk.objects.filter(document=document).order_by("chunk_index")[:MAX_FALLBACK_CHUNKS]
    if chunks.exists():
        parts = []
        for c in chunks:
            if c.page_number is not None:
                parts.append(f"[Page {c.page_number}]: {c.content}")
            else:
                parts.append(c.content)
        return "\n\n".join(parts)
    return None


def _build_workspace_base_instruction(documents):
    doc_names = ", ".join(str(getattr(d, "title", "Document")) for d in documents[:6])
    return (
        f"You are InsightDocs AI, analyzing multiple documents in this workspace: {doc_names}.\n\n"
        "Rules:\n"
        "- Answer using the provided document contexts.\n"
        "- You MUST cite every source using [Document Title · Page X] format for each factual claim.\n"
        "- When comparing or synthesizing across documents, clearly state which document each point originates from.\n"
        "- If a question cannot be answered from the provided documents, state so clearly.\n"
        "- Format with clean Markdown (headers, bullet points, comparison tables where helpful).\n"
        "- Be concise, professional, and clear.\n"
    )


def _get_workspace_fallback_context(documents, chunks_per_doc=3):
    parts = []
    for doc in documents:
        if not getattr(doc, "is_processed", False):
            continue
        chunks = DocumentChunk.objects.filter(document=doc).order_by("chunk_index")[:chunks_per_doc]
        for c in chunks:
            page_info = f"Page {c.page_number}" if c.page_number else "No page"
            parts.append(f"[{doc.title} · {page_info}]: {c.content}")
    if parts:
        return "\n\n".join(parts)
    return None


def _call_groq_fallback(user_message, system_parts, chat_history):
    groq_api_key = getattr(settings, "GROQ_API_KEY", "")
    if not groq_api_key or not groq_api_key.strip():
        return None

    groq_model = getattr(settings, "GROQ_MODEL", "openai/gpt-oss-120b")
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {groq_api_key.strip()}",
        "Content-Type": "application/json",
    }
    messages = _build_history_messages(system_parts, chat_history) + [{"role": "user", "content": user_message}]

    # Primary model followed by active fallbacks in case configured model returns 404 (decommissioned/not found)
    candidate_models = [groq_model]
    for fallback_candidate in ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "llama-3.3-70b-versatile"]:
        if fallback_candidate not in candidate_models:
            candidate_models.append(fallback_candidate)

    for model_name in candidate_models:
        payload = {
            "model": model_name,
            "messages": messages,
            "temperature": 0.2,
        }
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            if response.status_code == 200:
                data = response.json()
                choices = data.get("choices", [])
                if choices:
                    return choices[0].get("message", {}).get("content", "")
            else:
                logger.warning(
                    f"Groq API call with model '{model_name}' failed ({response.status_code}): {response.text}"
                )
                if response.status_code == 404:
                    continue
                break
        except Exception as e:
            logger.warning(f"Groq fallback request error with model '{model_name}': {e}")
            break

    return None


async def _yield_in_chunks(text, size=FALLBACK_CHUNK_SIZE):
    if not text:
        return
    for i in range(0, len(text), size):
        yield text[i : i + size]


async def get_gemini_response(user_message, document, chat_history, progress_callback=None, usage_callback=None, user=None):
    """
    Async generator that streams response using agentic hybrid RAG.
    Pro & Industry users are powered by DeepSeek v4 Flash (with seamless Gemini fallback).
    Free tier users are routed to Gemini 2.5 Flash.
    """
    context_text = None
    system_parts = None
    base_instruction = None
    try:
        logger.info(f"Starting generation for doc {document.id}...")
        model = genai.GenerativeModel("gemini-2.5-flash")

        # Detect global query vs local query
        query_lower = user_message.lower()
        global_keywords = [
            "list all", "all questions", "every question", "entire", "summarize", 
            "summary", "full", "whole", "table of contents", "index", "complete list",
            "all the questions", "list questions", "list the questions"
        ]
        is_global_query = any(w in query_lower for w in global_keywords)
        base_instruction = _build_base_instruction(is_global_query)

        if progress_callback:
            await progress_callback("Analyzing question intent...")

        # If we recently hit a Gemini quota/rate-limit, avoid repeated failed calls.
        if _is_gemini_in_cooldown():
            logger.warning("Gemini is in cooldown, attempting Groq fallback directly.")
            if progress_callback:
                await progress_callback("Gemini limited, switching model...")
            context_text = await asyncio.to_thread(_get_document_fallback_context, document)
            if not context_text:
                yield DOCUMENT_UNREADABLE_MSG
                return
            system_parts = _build_system_parts(base_instruction, context_text)
            groq_text = await asyncio.to_thread(_call_groq_fallback, user_message, system_parts, chat_history)
            if groq_text:
                async for part in _yield_in_chunks(groq_text):
                    yield part
                return
            logger.warning("Gemini rate limit cooldown active; Groq fallback not configured (set GROQ_API_KEY in .env to enable fallback).")
            yield USER_FRIENDLY_RATE_LIMIT_MSG
            return

        def _retrieve_pipeline():
            if not document.is_processed:
                logger.warning(f"Doc {document.id} not yet processed for RAG.")
                return None

            if is_global_query:
                # For global/summary/comprehensive queries, fetch up to 60 chunks in chronological order to ensure full recall.
                chunks_to_use = list(DocumentChunk.objects.filter(document=document).order_by("chunk_index")[:60])
            else:
                # Normal local query: retrieve top hybrid chunks.
                has_children = _has_child_chunks(document)
                enable_rerank = getattr(settings, "ENABLE_FLASHRANK_RERANK", False)

                if has_children:
                    # _parent_child_retrieve already retrieves child candidates,
                    # walks down to unique parents, reranks child texts (if FlashRank enabled),
                    # and applies the relative score cutoff.
                    retrieval = _parent_child_retrieve(document, user_message, top_k=getattr(settings, "RERANK_TOP_N", 5))
                    chunks_to_use = retrieval["hybrid_chunks"]
                elif enable_rerank:
                    rerank_candidates_k = getattr(settings, "RERANK_CANDIDATES", 20)
                    rerank_top_n = getattr(settings, "RERANK_TOP_N", 5)
                    retrieval = _hybrid_retrieve(document, user_message, top_k=rerank_candidates_k)
                    reranked = _rerank_chunks_flashrank(
                        user_message, retrieval["hybrid_chunks"], top_n=rerank_candidates_k
                    )
                    unique_items = []
                    seen_ids = set()
                    for c in reranked:
                        if c.id not in seen_ids:
                            seen_ids.add(c.id)
                            unique_items.append({
                                "chunk": c,
                                "hybrid_score": float(getattr(c, "rerank_score", 0.0) or 0.0),
                                "dropped": False,
                            })
                            if len(unique_items) >= rerank_top_n:
                                break
                    kept, _ = _apply_relative_score_cutoff(unique_items, score_key="hybrid_score")
                    chunks_to_use = [item["chunk"] for item in kept]
                else:
                    retrieval = _hybrid_retrieve(document, user_message, top_k=10)
                    chunks_to_use = retrieval["hybrid_chunks"]

            if not chunks_to_use:
                return None

            # Format chunks with page markers
            parts = []
            for c in chunks_to_use:
                if c.page_number is not None:
                    parts.append(f"[Page {c.page_number}]: {c.content}")
                else:
                    parts.append(c.content)
            return "\n\n".join(parts)

        try:
            enable_agentic = getattr(settings, "ENABLE_AGENTIC_RETRIEVAL", False)
            if enable_agentic and chat_history and not is_global_query:
                if progress_callback:
                    await progress_callback("Analyzing query intent and rewriting with context...")
                from .agentic_retrieval import run_agentic_retrieval
                agentic_res = await run_agentic_retrieval(user_message, [document], chat_history)
                context_text = agentic_res.get("context_text")
                tokens_used = agentic_res.get("tokens_used", 0)
                if tokens_used > 0 and usage_callback:
                    try:
                        await usage_callback(tokens_used)
                    except Exception as u_err:
                        logger.warning(f"Failed to record agentic rewrite tokens: {u_err}")
            else:
                if progress_callback:
                    await progress_callback("Retrieving and ranking sources...")
                context_text = await asyncio.to_thread(_retrieve_pipeline)
        except Exception as retrieval_error:
            logger.warning(f"Retrieval pipeline failed, using chunk fallback: {retrieval_error}")
            err_text = str(retrieval_error).lower()
            if "429" in err_text or "quota" in err_text or "rate" in err_text:
                _set_gemini_cooldown_from_error(str(retrieval_error))
            context_text = None

        if not context_text:
            logger.info("RAG pipeline returned no context. Falling back to first chunks.")
            context_text = await asyncio.to_thread(_get_document_fallback_context, document)

        if not context_text:
            yield DOCUMENT_UNREADABLE_MSG
            return

        if progress_callback:
            await progress_callback("Refining context for final answer...")

        system_parts = _build_system_parts(base_instruction, context_text)

        history = [
            {"role": "user", "parts": system_parts},
            {"role": "model", "parts": ["Understood. I will prioritize document-grounded answers."]},
        ]

        for msg in chat_history:
            role = "user" if msg.get("role") == "user" else "model"
            content = msg.get("content", "").strip()
            if content:
                history.append({"role": role, "parts": [content]})

        # PRO & INDUSTRY PLANS: ROUTE TO DEEPSEEK-V4-FLASH
        if user and getattr(user, 'is_premium', False):
            if progress_callback:
                await progress_callback("Synthesizing answer with DeepSeek v4 Flash...")
            try:
                system_prompt = f"{base_instruction}\n\nDOCUMENT CONTEXT:\n{context_text}"
                ds_messages = []
                for msg in chat_history:
                    role = "user" if msg.get("role") == "user" else "assistant"
                    content = msg.get("content", "").strip()
                    if content:
                        ds_messages.append({"role": role, "content": content})
                ds_messages.append({"role": "user", "content": user_message})

                from .deepseek_client import stream_deepseek_chat
                async for chunk in stream_deepseek_chat(
                    system_prompt=system_prompt,
                    messages=ds_messages,
                    usage_callback=usage_callback,
                ):
                    yield chunk
                return
            except Exception as ds_err:
                logger.warning(f"DeepSeek v4 Flash fallback to Gemini for doc {document.id}: {ds_err}")
                if progress_callback:
                    await progress_callback("DeepSeek busy, routing to Gemini...")

        try:
            if progress_callback:
                await progress_callback("Generating answer...")
            chat = model.start_chat(history=history)
            response_stream = await chat.send_message_async(user_message, stream=True)

            last_usage = None
            async for chunk in response_stream:
                try:
                    if chunk.text:
                        yield chunk.text
                except (ValueError, AttributeError):
                    pass
                last_usage = getattr(chunk, "usage_metadata", None) or last_usage

            if usage_callback and last_usage is not None:
                prompt_tokens = getattr(last_usage, "prompt_token_count", 0) or 0
                candidates_tokens = getattr(last_usage, "candidates_token_count", 0) or 0
                total_tokens = prompt_tokens + candidates_tokens
                await usage_callback(total_tokens)

            usage_acc = []
            usage_reported = False

            async def _report_usage(extra_tokens=0):
                nonlocal usage_reported
                if usage_reported or not usage_callback:
                    return
                total_tokens = sum(usage_acc) + (extra_tokens or 0)
                if total_tokens <= 0:
                    return
                usage_reported = True
                try:
                    await usage_callback(total_tokens)
                except Exception as usage_error:
                    logger.warning(f"usage_callback failed for doc {document.id}: {usage_error}")
            
        except ResourceExhausted:
            _set_gemini_cooldown_from_error("seconds: 60")
            logger.warning(f"Gemini API limit reached for doc {document.id}, switching to Groq.")
            if progress_callback:
                await progress_callback("Gemini limit reached, switching to Groq...")
            groq_text = await asyncio.to_thread(_call_groq_fallback, user_message, system_parts, chat_history)
            if groq_text:
                async for part in _yield_in_chunks(groq_text):
                    yield part
                return
            logger.warning("Gemini API limit reached; Groq fallback not configured (set GROQ_API_KEY in .env to enable fallback).")
            yield USER_FRIENDLY_RATE_LIMIT_MSG
        except Exception as model_error:
            err_text = str(model_error).lower()
            fallback_allowed = any(
                token in err_text
                for token in ["api key", "invalid", "expired", "quota", "permission", "unauthorized", "429"]
            )
            if fallback_allowed:
                _set_gemini_cooldown_from_error(str(model_error))
                logger.warning(f"Gemini auth/quota issue, switching to Groq: {model_error}")
                if progress_callback:
                    await progress_callback("Gemini unavailable, switching to Groq...")
                groq_text = await asyncio.to_thread(_call_groq_fallback, user_message, system_parts, chat_history)
                if groq_text:
                    async for part in _yield_in_chunks(groq_text):
                        yield part
                    return
            raise

    except ResourceExhausted:
        _set_gemini_cooldown_from_error("seconds: 60")
        logger.warning(f"Gemini API limit exceeded for doc {document.id}.")
        if not system_parts:
            if not context_text:
                context_text = await asyncio.to_thread(_get_document_fallback_context, document)
            if context_text:
                if not base_instruction:
                    base_instruction = _build_base_instruction(is_global_query=False)
                system_parts = _build_system_parts(base_instruction, context_text)
        if not system_parts:
            yield DOCUMENT_UNREADABLE_MSG
            return
        groq_text = await asyncio.to_thread(_call_groq_fallback, user_message, system_parts, chat_history)
        if groq_text:
            async for part in _yield_in_chunks(groq_text):
                yield part
        else:
            logger.warning("Gemini API limit exceeded; Groq fallback not configured.")
            yield USER_FRIENDLY_RATE_LIMIT_MSG
    except Exception as e:
        logger.error(f"Gemini Error: {e}", exc_info=True)
        if not system_parts:
            if not context_text:
                context_text = await asyncio.to_thread(_get_document_fallback_context, document)
            if context_text:
                if not base_instruction:
                    base_instruction = _build_base_instruction(is_global_query=False)
                system_parts = _build_system_parts(base_instruction, context_text)
        if not system_parts:
            yield DOCUMENT_UNREADABLE_MSG
            return
        groq_text = await asyncio.to_thread(_call_groq_fallback, user_message, system_parts, chat_history)
        if groq_text:
            async for part in _yield_in_chunks(groq_text):
                yield part
        else:
            yield USER_FRIENDLY_ERROR_MSG


def _multi_doc_retrieve(documents, user_message, query_embedding=None, return_details=False):
    """Retrieve relevant chunks across all workspace documents with a single shared query embedding."""
    shared_embedding = query_embedding
    if shared_embedding is None:
        try:
            res = genai.embed_content(
                model="models/gemini-embedding-001",
                content=user_message,
                task_type="retrieval_query",
                output_dimensionality=768,
            )
            shared_embedding = res.get("embedding")
        except Exception as emb_err:
            logger.warning(f"Failed to generate shared embedding for workspace query: {emb_err}")

    enable_rerank = getattr(settings, "ENABLE_FLASHRANK_RERANK", False)
    processed_docs = [doc for doc in documents if getattr(doc, "is_processed", False)]
    doc_count = max(len(processed_docs), 1)

    if enable_rerank:
        rerank_candidates_k = getattr(settings, "RERANK_CANDIDATES", 20)
        rerank_top_n = getattr(settings, "RERANK_TOP_N", 5)
        per_doc_top_k = max(5, int(math.ceil(rerank_candidates_k / doc_count)))
    else:
        rerank_top_n = getattr(settings, "RERANK_TOP_N", 5)
        per_doc_top_k = 5

    all_scored = []
    for doc in processed_docs:
        try:
            retrieval = _hybrid_retrieve(doc, user_message, top_k=per_doc_top_k, query_embedding=shared_embedding)
            scored_items = retrieval.get("scored")
            if scored_items is not None:
                for item in scored_items:
                    all_scored.append({
                        "hybrid_score": item.get("hybrid_score", 0.0),
                        "chunk": item["chunk"],
                        "doc_title": getattr(doc, "title", "Document"),
                        "doc_id": getattr(doc, "id", None),
                        "dropped": False,
                    })
            else:
                for chunk in retrieval.get("hybrid_chunks", []):
                    all_scored.append({
                        "hybrid_score": 0.0,
                        "chunk": chunk,
                        "doc_title": getattr(doc, "title", "Document"),
                        "doc_id": getattr(doc, "id", None),
                        "dropped": False,
                    })
        except Exception as e:
            logger.warning(f"Retrieval failed for doc {getattr(doc, 'id', None)}: {e}")

    if not all_scored:
        return None if not return_details else {"context_text": None, "final_items": [], "all_unique": [], "all_scored": []}

    # Sort all candidates across documents by hybrid_score descending
    all_scored.sort(key=lambda x: x["hybrid_score"], reverse=True)

    if enable_rerank:
        candidate_items = all_scored[:rerank_candidates_k]
        candidate_chunks = [item["chunk"] for item in candidate_items]
        chunk_id_to_item = {item["chunk"].id: item for item in candidate_items}
        reranked_chunks = _rerank_chunks_flashrank(user_message, candidate_chunks, top_n=len(candidate_chunks))

        # Walk down reranked chunks to collect unique parents/chunks up to rerank_top_n
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
        all_unique = unique_items
        final_items = unique_items

    parts = []
    for item in final_items:
        chunk = item["chunk"]
        doc_title = item["doc_title"]
        page_info = f"Page {chunk.page_number}" if getattr(chunk, "page_number", None) else "No page"
        parts.append(f"[{doc_title} · {page_info}]: {chunk.content}")

    context_text = "\n\n".join(parts)
    if return_details:
        return {
            "context_text": context_text,
            "final_items": final_items,
            "all_unique": all_unique,
            "all_scored": all_scored,
        }
    return context_text


async def get_workspace_response(user_message, documents, chat_history, progress_callback=None, usage_callback=None, user=None, query_embedding=None):
    """
    Async generator that streams response using multi-document hybrid RAG.
    Retrieves relevant chunks across ALL documents in the workspace.
    Pro & Industry users are powered by DeepSeek v4 Flash (with Gemini fallback).
    """
    context_text = None
    system_parts = None
    base_instruction = _build_workspace_base_instruction(documents)
    try:
        logger.info(f"Starting workspace generation across {len(documents)} documents...")
        model = genai.GenerativeModel("gemini-2.5-flash")

        if progress_callback:
            await progress_callback("Searching across workspace documents...")

        if _is_gemini_in_cooldown():
            context_text = await asyncio.to_thread(_get_workspace_fallback_context, documents)
            if not context_text:
                yield WORKSPACE_DOCUMENTS_UNREADABLE_MSG
                return
            system_parts = _build_system_parts(base_instruction, context_text)
            groq_text = await asyncio.to_thread(_call_groq_fallback, user_message, system_parts, chat_history)
            if groq_text:
                async for part in _yield_in_chunks(groq_text):
                    yield part
                return
            yield USER_FRIENDLY_RATE_LIMIT_MSG
            return

        try:
            enable_agentic = getattr(settings, "ENABLE_AGENTIC_RETRIEVAL", False)
            if enable_agentic:
                if progress_callback:
                    await progress_callback("Analyzing query intent and routing retrieval...")
                from .agentic_retrieval import run_agentic_retrieval
                agentic_res = await run_agentic_retrieval(user_message, documents, chat_history)
                context_text = agentic_res.get("context_text")
                tokens_used = agentic_res.get("tokens_used", 0)
                if tokens_used > 0 and usage_callback:
                    try:
                        await usage_callback(tokens_used)
                    except Exception as u_err:
                        logger.warning(f"Failed to record agentic rewrite tokens: {u_err}")
            else:
                if progress_callback:
                    await progress_callback("Retrieving context from multiple sources...")
                context_text = await asyncio.to_thread(_multi_doc_retrieve, documents, user_message, query_embedding=query_embedding)
        except Exception as e:
            logger.warning(f"Multi-doc retrieval failed: {e}")
            context_text = None

        if not context_text:
            context_text = await asyncio.to_thread(_get_workspace_fallback_context, documents)
        if not context_text:
            yield WORKSPACE_DOCUMENTS_UNREADABLE_MSG
            return

        system_parts = _build_system_parts(base_instruction, context_text)
        history = [
            {"role": "user", "parts": system_parts},
            {"role": "model", "parts": ["Understood. I will answer across the workspace documents with clear citations."]},
        ]
        for msg in chat_history:
            role = "user" if msg.get("role") == "user" else "model"
            content = msg.get("content", "").strip()
            if content:
                history.append({"role": role, "parts": [content]})

        # PRO & INDUSTRY PLANS: ROUTE TO DEEPSEEK-V4-FLASH
        if user and getattr(user, 'is_premium', False):
            if progress_callback:
                await progress_callback("Synthesizing multi-document analysis with DeepSeek v4 Flash...")
            try:
                system_prompt = f"{base_instruction}\n\nWORKSPACE DOCUMENT CONTEXTS:\n{context_text}"
                ds_messages = []
                for msg in chat_history:
                    role = "user" if msg.get("role") == "user" else "assistant"
                    content = msg.get("content", "").strip()
                    if content:
                        ds_messages.append({"role": role, "content": content})
                ds_messages.append({"role": "user", "content": user_message})

                from .deepseek_client import stream_deepseek_chat
                async for chunk in stream_deepseek_chat(
                    system_prompt=system_prompt,
                    messages=ds_messages,
                    usage_callback=usage_callback,
                ):
                    yield chunk
                return
            except Exception as ds_err:
                logger.warning(f"DeepSeek v4 Flash workspace fallback to Gemini: {ds_err}")
                if progress_callback:
                    await progress_callback("Routing to Gemini...")

        try:
            chat = model.start_chat(history=history)
            response_stream = await chat.send_message_async(user_message, stream=True)

            last_usage = None
            async for chunk in response_stream:
                try:
                    if chunk.text:
                        yield chunk.text
                except (ValueError, AttributeError):
                    pass
                last_usage = getattr(chunk, "usage_metadata", None) or last_usage

            if usage_callback and last_usage is not None:
                prompt_tokens = getattr(last_usage, "prompt_token_count", 0) or 0
                candidates_tokens = getattr(last_usage, "candidates_token_count", 0) or 0
                total_tokens = prompt_tokens + candidates_tokens
                if total_tokens > 0:
                    await usage_callback(total_tokens)

        except ResourceExhausted:
            _set_gemini_cooldown_from_error("seconds: 60")
            if not system_parts:
                if not context_text:
                    context_text = await asyncio.to_thread(_get_workspace_fallback_context, documents)
                if context_text:
                    system_parts = _build_system_parts(base_instruction, context_text)
            if not system_parts:
                yield WORKSPACE_DOCUMENTS_UNREADABLE_MSG
                return
            groq_text = await asyncio.to_thread(_call_groq_fallback, user_message, system_parts, chat_history)
            if groq_text:
                async for part in _yield_in_chunks(groq_text):
                    yield part
            else:
                yield USER_FRIENDLY_RATE_LIMIT_MSG

    except Exception as e:
        logger.error(f"Workspace Gemini Error: {e}", exc_info=True)
        yield "I encountered a temporary issue analyzing the workspace documents. Please try asking your question again in a moment."


