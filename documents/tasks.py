import requests
import tempfile
import os
import logging 
import json
import google.generativeai as genai

from celery import shared_task
from django.conf import settings
from .models import Document, DocumentInsights
from .utils.rag import process_document_for_rag
from .utils.storage import prepare_local_document

logger = logging.getLogger(__name__)
genai.configure(api_key=settings.GOOGLE_API_KEY)


def _safe_json_loads(raw_text):
    text = (raw_text or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    return json.loads(text)


def _fallback_questions(document):
    title = (document.title or "this document").strip()
    return [
        f"What are the key takeaways from {title}?",
        "Can you summarize the most important sections?",
        "What questions should I ask next based on this document?",
    ]

@shared_task
def process_document_task(document_id):

    try:
        document = Document.objects.get(id=document_id)
        local_path, cleanup = prepare_local_document(document)
        try:
            process_document_for_rag(document, local_path)
            analyze_document_task.delay(document_id)
        finally:
            cleanup()

        logger.info(f"Task completed: Document {document_id} processed successfully")
    except Document.DoesNotExist:
        logger.info(f"Task failed: Document {document_id} not found.")
    except Exception as e:
        logger.error(f"Task failed for document {document_id}: {str(e)}")
        raise e


@shared_task
def analyze_document_task(document_id):
    """
    Agentic analysis step run post-indexing:
    - executive summary
    - entity extraction
    - follow-up questions
    """
    try:
        document = Document.objects.get(id=document_id)
        insights, _ = DocumentInsights.objects.get_or_create(document=document)
        insights.status = DocumentInsights.STATUS_PROCESSING
        insights.save(update_fields=["status", "generated_at"])

        chunks = list(document.chunks.order_by("chunk_index").values_list("content", flat=True)[:25])
        if not chunks:
            insights.status = DocumentInsights.STATUS_FAILED
            insights.save(update_fields=["status", "generated_at"])
            logger.warning(f"No chunks available for document analysis: {document_id}")
            return

        context = "\n\n".join(chunks)
        model = genai.GenerativeModel("gemini-2.5-flash")
        prompt = f"""
You are a document-analysis agent.
Analyze the document context and return STRICT JSON with this schema:
{{
  "summary": "string",
  "entities": {{
    "dates": ["..."],
    "amounts": ["..."],
    "names": ["..."],
    "organizations": ["..."]
  }},
  "suggested_questions": ["...", "...", "..."]
}}
Rules:
- Keep summary to 120-220 words.
- Only include entities grounded in the context.
- Suggested questions must be useful next user prompts.
- Return valid JSON only.

DOCUMENT CONTEXT:
{context}
"""
        response = model.generate_content(prompt)
        text = (response.text or "").strip()
        payload = _safe_json_loads(text)

        insights.summary = payload.get("summary", "")
        insights.entities = payload.get("entities", {})
        suggested = payload.get("suggested_questions", [])
        if not isinstance(suggested, list):
            suggested = []
        suggested = [q.strip() for q in suggested if isinstance(q, str) and q.strip()]
        insights.suggested_questions = suggested[:6] if suggested else _fallback_questions(document)
        insights.status = DocumentInsights.STATUS_COMPLETED
        insights.save()
        logger.info(f"Document analysis completed: {document_id}")
    except Document.DoesNotExist:
        logger.info(f"Analyze skipped; document not found: {document_id}")
    except Exception as e:
        logger.error(f"Analyze task failed for document {document_id}: {e}", exc_info=True)
        DocumentInsights.objects.filter(document_id=document_id).update(
            status=DocumentInsights.STATUS_FAILED
        )