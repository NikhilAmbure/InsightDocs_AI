import asyncio
import json
import logging
import re
import requests
import time

import google.generativeai as genai
from django.conf import settings
from django.contrib.postgres.search import SearchQuery, SearchRank, SearchVector
from google.api_core.exceptions import ResourceExhausted
from pgvector.django import L2Distance

from ..models import DocumentChunk

logger = logging.getLogger(__name__)

genai.configure(api_key=settings.GOOGLE_API_KEY)

TOP_K_PER_STEP = 5
MAX_FALLBACK_CHUNKS = 20
FALLBACK_CHUNK_SIZE = 180
GEMINI_COOLDOWN_UNTIL = 0.0

try:
    from sentence_transformers import CrossEncoder  # pyright: ignore[reportMissingImports]
except Exception:
    CrossEncoder = None

_CROSS_ENCODER = None
_CROSS_ENCODER_DISABLED = False




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

def _hybrid_retrieve(document, query_text, top_k=TOP_K_PER_STEP):
    result = genai.embed_content(
        model="models/gemini-embedding-001",
        content=query_text,
        task_type="retrieval_query",
        output_dimensionality=3072,
    )
    query_embedding = result["embedding"]

    vector_candidates = list(
        DocumentChunk.objects.filter(document=document)
        .annotate(distance=L2Distance("embedding", query_embedding))
        .order_by("distance")[: top_k * 2]
    )

    keyword_vector = SearchVector("content")
    keyword_query = SearchQuery(query_text, search_type="websearch")
    keyword_candidates = list(
        DocumentChunk.objects.filter(document=document)
        .annotate(keyword_rank=SearchRank(keyword_vector, keyword_query))
        .filter(keyword_rank__gt=0)
        .order_by("-keyword_rank")[: top_k * 2]
    )

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
        combined[chunk.id]["keyword_score"] = float(getattr(chunk, "keyword_rank", 0.0) or 0.0)

    scored = list(combined.values())
    for row in scored:
        row["hybrid_score"] = (0.7 * row["vector_score"]) + (0.3 * row["keyword_score"])

    scored.sort(key=lambda x: x["hybrid_score"], reverse=True)
    return {
        "vector_chunks": vector_candidates[:top_k],
        "keyword_chunks": keyword_candidates[:top_k],
        "hybrid_chunks": [row["chunk"] for row in scored[:top_k]],
    }


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


def _call_groq_fallback(user_message, system_parts, chat_history):
    groq_api_key = getattr(settings, "GROQ_API_KEY", "")
    if not groq_api_key:
        return None

    groq_model = getattr(settings, "GROQ_MODEL", "llama-3.3-70b-versatile")
    url = "https://api.groq.com/openai/v1/chat/completions"
    payload = {
        "model": groq_model,
        "messages": _build_history_messages(system_parts, chat_history) + [{"role": "user", "content": user_message}],
        "temperature": 0.2,
    }
    headers = {
        "Authorization": f"Bearer {groq_api_key}",
        "Content-Type": "application/json",
    }
    response = requests.post(url, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    data = response.json()
    choices = data.get("choices", [])
    if not choices:
        return None
    return choices[0].get("message", {}).get("content", "")


async def _yield_in_chunks(text, size=FALLBACK_CHUNK_SIZE):
    if not text:
        return
    for i in range(0, len(text), size):
        yield text[i : i + size]


async def get_gemini_response(user_message, document, chat_history, progress_callback=None, usage_callback=None):
    """
    Async generator that streams response using agentic hybrid RAG.
    """
    try:
        logger.info(f"Starting Gemini generation for doc {document.id}...")
        model = genai.GenerativeModel("gemini-2.5-flash")
        if progress_callback:
            await progress_callback("Analyzing question intent...")

        # If we recently hit a Gemini quota/rate-limit, avoid repeated failed calls.
        if _is_gemini_in_cooldown():
            logger.warning("Gemini is in cooldown, attempting Groq fallback directly.")
            if progress_callback:
                await progress_callback("Gemini limited, switching model...")
            groq_text = await asyncio.to_thread(_call_groq_fallback, user_message, [""], chat_history)
            if groq_text:
                async for part in _yield_in_chunks(groq_text):
                    yield part
                return
            yield "Gemini is temporarily rate-limited and Groq fallback is not configured. Set GROQ_API_KEY in .env and retry."
            return

        def _retrieve_pipeline():
            if not document.is_processed:
                logger.warning(f"Doc {document.id} not yet processed for RAG.")
                return None

            query_steps = _plan_query_steps(user_message)
            logger.info(f"Planned retrieval steps: {query_steps}")

            selected = []
            seen = set()
            for step_query in query_steps:
                retrieval = _hybrid_retrieve(document, step_query, top_k=TOP_K_PER_STEP)

                # 1) Take top pgvector chunks (primary semantic retrieval).
                vector_chunks = retrieval["vector_chunks"]
                cross_reranked = _rerank_chunks_cross_encoder(user_message, vector_chunks)
                if cross_reranked is None:
                    cross_reranked = _rerank_chunks_gemini(user_message, vector_chunks)

                # 2) Add keyword chunks from PostgreSQL FTS to keep exact-term recall high.
                fts_chunks = retrieval["keyword_chunks"]
                step_chunks = (cross_reranked[:TOP_K_PER_STEP] if cross_reranked else []) + fts_chunks

                for chunk in step_chunks:
                    if chunk.id not in seen:
                        selected.append(chunk)
                        seen.add(chunk.id)

            if not selected:
                return None

            final_reranked = _rerank_chunks_cross_encoder(user_message, selected[:10])
            if final_reranked is None:
                final_reranked = _rerank_chunks_gemini(user_message, selected[:10])

            # Contextual compression before answer generation.
            return _compress_context(user_message, final_reranked[:TOP_K_PER_STEP])

        try:
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

            def _fallback_chunks():
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

            context_text = await asyncio.to_thread(_fallback_chunks)

        if progress_callback:
            await progress_callback("Refining context for final answer...")

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
            "- Do NOT hallucinate document-specific facts.\n"
            "- Be concise, clear, and helpful.\n"
        )

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
                total_tokens = getattr(last_usage, "total_token_count", 0) or 0
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
            yield "API limit reached and Groq fallback is not configured. Add GROQ_API_KEY in .env to enable fallback."
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
        groq_text = await asyncio.to_thread(_call_groq_fallback, user_message, [""], chat_history)
        if groq_text:
            async for part in _yield_in_chunks(groq_text):
                yield part
        else:
            yield "API limit reached. Configure GROQ_API_KEY in .env for automatic fallback."
    except Exception as e:
        logger.error(f"Gemini Error: {e}", exc_info=True)
        groq_text = await asyncio.to_thread(_call_groq_fallback, user_message, [""], chat_history)
        if groq_text:
            async for part in _yield_in_chunks(groq_text):
                yield part
        else:
            yield "An error occurred while processing your request. Please try again later."

