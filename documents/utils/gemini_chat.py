import google.generativeai as genai
import logging
import asyncio

from google.api_core.exceptions import ResourceExhausted

from django.conf import settings
from pgvector.django import L2Distance
from ..models import DocumentChunk

logger = logging.getLogger(__name__)

genai.configure(api_key=settings.GOOGLE_API_KEY)

async def get_gemini_response(user_message, document, chat_history):
    """
    Async Generator that streams response using RAG -> Fallback with chunk text.
    """
    try:
        logger.info(f"Starting Gemini generation for doc {document.id}...")
        
        model = genai.GenerativeModel("gemini-2.5-flash")
        
        context_text = ""

        # --- STRATEGY 1: RAG Vector Search ---
        def perform_rag_search():
            if not document.is_processed:
                logger.warning(f"Doc {document.id} not yet processed for RAG.")
                return None
            
            result = genai.embed_content(
                model="models/gemini-embedding-001",  # ✅ Same model as indexing
                content=user_message,
                task_type="retrieval_query"
            )
            query_embedding = result['embedding']

            chunks = DocumentChunk.objects.filter(document=document) \
                .annotate(distance=L2Distance('embedding', query_embedding)) \
                .order_by('distance')[:5]
            
            if chunks.exists():
                context_parts = []
                for c in chunks:
                    if c.page_number is not None:
                        context_parts.append(f"[Page {c.page_number}]: {c.content}")
                    else:
                        context_parts.append(f"{c.content}")
                return "\n\n".join(context_parts)
            return None

        context_text = await asyncio.to_thread(perform_rag_search)

        # --- STRATEGY 2: Fallback — use ALL chunks if RAG returned nothing ---
        # This handles cases where the document is processed but query embedding
        # doesn't match well (e.g. very short docs, first-time queries).
        if not context_text:
            logger.info("⚠️ RAG returned no context. Falling back to full chunk text.")

            def get_all_chunks():
                chunks = DocumentChunk.objects.filter(document=document).order_by('chunk_index')[:20]
                if chunks.exists():
                    context_parts = []
                    for c in chunks:
                        if c.page_number is not None:
                            context_parts.append(f"[Page {c.page_number}]: {c.content}")
                        else:
                            context_parts.append(f"{c.content}")
                    return "\n\n".join(context_parts)
                return None

            context_text = await asyncio.to_thread(get_all_chunks)

        if context_text:
            logger.info("✅ Context found (RAG or fallback).")
        else:
            logger.warning("⚠️ No context available. Document may not be processed yet.")

        # --- BUILD SYSTEM PROMPT ---
        base_instruction = (
            "You are InsightDocs AI, an intelligent document assistant.\n\n"
            "You may use THREE sources of knowledge in priority order:\n"
            "1. The provided document context (highest priority)\n"
            "2. General knowledge (if the document does not contain the answer)\n"
            "3. Logical reasoning based on the user's question\n\n"
            "Rules:\n"
            "- If the answer is found in the document context, answer strictly from it.\n"
            "- Please include inline citations to the document pages when answering. If the context has [Page X]: ..., cite it as [Page X] at the end of the relevant sentence.\n"
            "- If the document does NOT contain the answer, answer using general knowledge.\n"
            "- If answering from general knowledge, explicitly say: "
            "'This answer is based on general knowledge, not the document.'\n"
            "- If you truly do not know the answer, say so honestly.\n"
            "- Do NOT hallucinate document-specific facts.\n"
            "- Be concise, clear, and helpful.\n"
        )

        if context_text:
            system_parts = [
                f"{base_instruction}\n\nDOCUMENT CONTEXT:\n{context_text}",
            ]
        else:
            # No chunks at all — document still processing
            system_parts = [
                base_instruction,
                (
                    "Note: The document has not finished processing yet or could not be read. "
                    "Let the user know their document may still be indexing and to try again shortly. "
                    "Do NOT say you need content provided — the document was already uploaded."
                )
            ]

        # --- BUILD CHAT HISTORY ---
        history = []
        history.append({"role": "user", "parts": system_parts})
        history.append({"role": "model", "parts": ["Understood. I'll answer based on the document context provided."]})

        for msg in chat_history:
            role = "user" if msg.get("role") == "user" else "model"
            content = msg.get("content", "").strip()
            if content:
                history.append({"role": role, "parts": [content]})

        # --- STREAM RESPONSE ---
        chat = model.start_chat(history=history)
        
        response_stream = await chat.send_message_async(
            user_message, 
            stream=True
        )

        async for chunk in response_stream:
            if chunk.text:
                yield chunk.text

    except ResourceExhausted:
        logger.warning(f"Gemini API rate limit exceeded for doc {document.id}.")
        yield "⚠️ **API Limit Reached.** The AI service is currently experiencing high demand. Please try asking your question again in a few moments."
    except Exception as e:
        logger.error(f"Gemini Error: {e}", exc_info=True)
        yield f"⚠️ **An error occurred.** We could not process your request at this time. Please try again later."