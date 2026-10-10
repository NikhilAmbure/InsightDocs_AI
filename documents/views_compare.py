import json
import logging

import google.generativeai as genai
from django.conf import settings
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.http import require_POST

from .models import Document, DocumentChunk

logger = logging.getLogger(__name__)
genai.configure(api_key=settings.GOOGLE_API_KEY)


@login_required(login_url='login')
def compare_select_view(request):
    """Page to select two documents for side-by-side comparison."""
    documents = Document.objects.filter(owner=request.user, is_processed=True).order_by('-uploaded_at')
    return render(request, 'compare_select.html', {
        'documents': documents,
    })


@login_required(login_url='login')
def compare_view(request):
    """Side-by-side comparison view."""
    doc1_id = request.GET.get('doc1')
    doc2_id = request.GET.get('doc2')

    if not doc1_id or not doc2_id or doc1_id == doc2_id:
        messages.error(request, 'Please select two distinct processed documents to compare.')
        return redirect('compare_select')

    doc1 = get_object_or_404(Document, id=doc1_id, owner=request.user, is_processed=True)
    doc2 = get_object_or_404(Document, id=doc2_id, owner=request.user, is_processed=True)

    return render(request, 'compare_result.html', {
        'doc1': doc1,
        'doc2': doc2,
    })


@login_required(login_url='login')
@require_POST
def compare_analyze_api(request):
    """API endpoint to generate AI-powered comparison between two documents."""
    try:
        body = json.loads(request.body)
        doc1_id = body.get('doc1_id')
        doc2_id = body.get('doc2_id')
        aspect = body.get('aspect', 'general')
    except (json.JSONDecodeError, AttributeError):
        return JsonResponse({'error': 'Invalid request body.'}, status=400)

    doc1 = get_object_or_404(Document, id=doc1_id, owner=request.user, is_processed=True)
    doc2 = get_object_or_404(Document, id=doc2_id, owner=request.user, is_processed=True)

    # Fetch representative chunks from both documents
    chunks1 = DocumentChunk.objects.filter(document=doc1).order_by('chunk_index')[:16]
    chunks2 = DocumentChunk.objects.filter(document=doc2).order_by('chunk_index')[:16]

    context1_parts = [
        f"[Page {c.page_number}]: {c.content}" if c.page_number else c.content
        for c in chunks1
    ]
    context2_parts = [
        f"[Page {c.page_number}]: {c.content}" if c.page_number else c.content
        for c in chunks2
    ]

    context1 = "\n\n".join(context1_parts)[:6000]
    context2 = "\n\n".join(context2_parts)[:6000]

    aspect_instructions = {
        'general': 'Provide an in-depth executive comparison covering scope, key findings, similarities, and differences.',
        'key_terms': 'Focus specifically on terminology, key definitions, metrics, and clauses used in each document.',
        'structure': 'Compare organizational structure, methodology, flow of arguments, and presentation style.',
        'arguments': 'Critically evaluate the core claims, conclusions, strengths, and limitations presented in each document.',
    }
    instruction = aspect_instructions.get(aspect, aspect_instructions['general'])

    prompt = f"""You are an elite document intelligence analyst comparing two documents side-by-side.

DOCUMENT 1: "{doc1.title}"
CONTEXT 1:
{context1}

DOCUMENT 2: "{doc2.title}"
CONTEXT 2:
{context2}

TASK:
{instruction}

RULES:
- Cite specific pages using [Doc 1 · Page X] and [Doc 2 · Page Y].
- Structure using these clean Markdown sections:
  ### Executive Summary
  ### Key Similarities
  ### Critical Differences
  ### Unique to "{doc1.title}"
  ### Unique to "{doc2.title}"
  ### Side-by-Side Comparison Matrix (Markdown table)
- Be objective, grounded strictly in the context, and highlight actionable insights.
"""

    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt)
        comparison_text = (response.text or "").strip()
        return JsonResponse({
            'status': 'success',
            'comparison': comparison_text,
            'aspect': aspect,
        })
    except Exception as e:
        logger.error(f"Document comparison error: {e}", exc_info=True)
        return JsonResponse({'error': 'Failed to generate comparison. Please try again.'}, status=500)

