import google.generativeai as genai
import logging
import asyncio

from django.conf import settings
from pgvector.django import L2Distance
from ..models import DocumentChunk

logger = logging.getLogger(__name__)

# Configure Gemini API
genai.configure(api_key=settings.GOOGLE_API_KEY)

async def get_gemini_response(user_message, document, chat_history):
    """
    Async Generator that streams response using RAG -> Fallback -> File Upload.
    """
    try:
        logger.info(f"Starting Gemini generation for doc {document.id}...")
        
        # Initialize Gemini Model (Async)
        model = genai.GenerativeModel("gemini-2.5-flash")
        
        context_text = ""
        gemini_file = None

        # --- STRATEGY 1: RAG (Run in Thread) ---
        # wrap DB calls in to_thread so they don't block the WebSocket
        def perform_rag_search():
            if not document.is_processed:
                return None
            
            # 1. Embed Query
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=user_message,
                task_type="retrieval_query"
            )
            query_embedding = result['embedding']

            # 2. Vector Search (L2 Distance)
            chunks = DocumentChunk.objects.filter(document=document) \
                .annotate(distance=L2Distance('embedding', query_embedding)) \
                .order_by('distance')[:5]
            
            if chunks.exists():
                return "\n\n".join([c.content for c in chunks])
            return None

        # Execute RAG Search
        context_text = await asyncio.to_thread(perform_rag_search)
        
        if context_text:
            logger.info("✅ RAG Context found.")
        else:
            logger.info("⚠️ No RAG context. Using Fallback strategy.")

        # --- STRATEGY 2: CONTEXT PREPARATION ---
        system_parts = []
        
        base_instruction = (
            "You are InsightDocs AI, an intelligent document assistant.\n\n"

            "You may use THREE sources of knowledge in priority order:\n"
            "1. The provided document context (highest priority)\n"
            "2. General knowledge (if the document does not contain the answer)\n"
            "3. Logical reasoning based on the user's question\n\n"

            "Rules:\n"
            "- If the answer is found in the document context, answer strictly from it.\n"
            "- If the document does NOT contain the answer, answer using general knowledge.\n"
            "- If answering from general knowledge, explicitly say: "
            "'This answer is based on general knowledge, not the document.'\n"
            "- If you truly do not know the answer, say so honestly.\n"
            "- Do NOT hallucinate document-specific facts.\n"
            "- Be concise, clear, and helpful.\n"
        )


        if context_text:
            # RAG Prompt
            system_parts = [
                f"{base_instruction}\n\nCONTEXT:\n{context_text}",
            ]
        else:
            # Fallback: Just general knowledge (File upload removed for speed/simplicity in async)
            # If you really need file upload here, it will slow down response time significantly.
            system_parts = [
                base_instruction,
                "Note: No specific document context was found for this query."
            ]

        # --- STRATEGY 3: BUILD HISTORY ---
        history = []
        # System prompt injected as first user turn
        history.append({"role": "user", "parts": system_parts})
        history.append({"role": "model", "parts": ["Understood."]})

        for msg in chat_history:
            role = "user" if msg.get("role") == "user" else "model"
            content = msg.get("content", "").strip()
            if content:
                history.append({"role": role, "parts": [content]})

        # --- STRATEGY 4: STREAMING RESPONSE ---
        chat = model.start_chat(history=history)
        
        # Async stream
        response_stream = await chat.send_message_async(
            user_message, 
            stream=True
        )

        async for chunk in response_stream:
            if chunk.text:
                yield chunk.text

    except Exception as e:
        logger.error(f"Gemini Error: {e}", exc_info=True)
        yield f"Error: {str(e)}"