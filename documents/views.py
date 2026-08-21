# views.py
import logging
import os
import json

from django.conf import settings
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.shortcuts import get_object_or_404, redirect, render
from django.http import HttpResponse, Http404, JsonResponse

from .forms import DocumentUploadForm
from .models import Document, ChatSession, ChatMessage, DocumentInsights
from .utils.rate_limit import check_rate_limit
from .tasks import process_document_task, analyze_document_task
from .utils.rag import process_document_for_rag

logger = logging.getLogger(__name__)

def _default_follow_up_questions(document):
    title = (document.title or "this document").strip()
    return [
        f"What are the main insights in {title}?",
        "Can you explain this in simpler terms?",
        "Which section should I review next?",
    ]


def landing_page_view(request):
    """Public marketing landing page."""
    if request.user.is_authenticated:
        return redirect("upload")
    return render(request, "landing_page.html")


@login_required(login_url='login')
def upload_view(request):
    """Handle document upload and display user's documents."""
    form = DocumentUploadForm(request.POST or None, request.FILES or None, user=request.user)

    if request.method == "POST":
        # Free users: 5 uploads per min; Premium users: 20 uploads per min
        limit = 20 if request.user.is_premium else 5
        window = 60
        
        limit_result = check_rate_limit(
            request,
            scope="upload",
            limit=limit,
            window=window,
        )

        if limit_result.limited:
            messages.error(
                request,
                f"You have reached the upload rate limit. "
                f"Please wait {limit_result.retry_after} seconds and try again.",
            )
            return redirect("upload")
        
        # CHECK UPLOAD COUNT LIMIT 
        current_doc_count = Document.objects.filter(owner=request.user).count()
        
        if request.user.is_premium:
            doc_limit = 50
        else:
            doc_limit = 10
        
        if current_doc_count >= doc_limit:
            messages.error(request, 
                f"You have reached your document limit of {doc_limit}. "
                "Please delete existing documents or upgrade your plan."
            )
            return redirect("upload")

        if form.is_valid():
            document = form.save(commit=False)
            document.owner = request.user
            document.original_name = document.file.name
            
            if not document.title:
                document.title = document.original_name
            document.save()

            try:
                process_document_task.delay(document.id)
            except Exception:
                # Celery/Redis not available — process synchronously
                logger.info(f"Celery unavailable, processing doc {document.id} synchronously.")
                try:
                    file_path = document.file.path
                except NotImplementedError:
                    file_path = document.file.url
                process_document_for_rag(document, file_path)
                analyze_document_task(document.id)
            
            # Create a chat session
            ChatSession.objects.create(document=document, user=request.user)
            return redirect("chat", document_id=document.id)
        else:
            # If the file is too big, the error will be in form.errors
            messages.error(request, "Upload failed. Please check the file requirements.")

    documents = Document.objects.filter(owner=request.user).order_by('-uploaded_at')
    context = {"form": form, "recent_documents": documents}
    return render(request, "upload.html", context)


@login_required(login_url='login')
def subscription_view(request):
    """Subscription / billing page view."""
    from payments.models import SubscriptionPlan
    pro_plan = SubscriptionPlan.objects.filter(slug="pro", is_active=True).first()
    return render(request, 'subscription.html', {'pro_plan': pro_plan})


@login_required(login_url='login')
def chat_view(request, document_id):
    """
    Main chat page - render initial UI with chat history.
    All real-time communication is handled via WebSocket in consumers.py
    """
    # Verify user has permission to access this document
    document = get_object_or_404(Document, id=document_id, owner=request.user)

    # Ensure session exists
    session, created = ChatSession.objects.get_or_create(
        document=document,
        user=request.user
    )

    # Get chat history for initial page load
    chat_history_qs = ChatMessage.objects.filter(session=session).order_by('created_at')
    
    # Get recent documents for sidebar
    recent_docs = Document.objects.filter(owner=request.user).exclude(id=document.id).order_by('-uploaded_at')[:5]
    insights = DocumentInsights.objects.filter(document=document).first()
    follow_up_questions = []
    if insights and isinstance(insights.suggested_questions, list):
        follow_up_questions = [q for q in insights.suggested_questions if isinstance(q, str) and q.strip()]
    if not follow_up_questions:
        follow_up_questions = _default_follow_up_questions(document)

    return render(request, "chat.html", {
        "document": document,
        "chat_history": chat_history_qs,
        "recent_documents": recent_docs,
        "document_insights": insights,
        "follow_up_questions": follow_up_questions,
    })

@login_required(login_url='login')
def serve_document_view(request, document_id):
    """
    Redirects to the cloudinary URL to serve the document file. 
    Since files are hosted externally, we don't serve bytes; we just point the browser to the file.
    """
    document = get_object_or_404(Document, id=document_id, owner=request.user)

    return redirect(document.file.url)


def coming_soon(request):
    return render(request, 'coming-soon.html')


@login_required(login_url='login')
def delete_document_view(request, document_id):
    """
    Delete a document and its associated chat session/messages.
    """
    if request.method == "POST":
        document = get_object_or_404(Document, id=document_id, owner=request.user)
        
        if document.file:
            try:
                document.file.delete(save=False)
            except Exception as e:
                logger.error(f"Error deleting file for doc {document.id}: {e}")

        # This will cascade delete ChatSession and ChatMessages due to on_delete=models.CASCADE
        document.delete()
        
        messages.success(request, "Chat and document deleted successfully.")
        return redirect("upload")
    
    return redirect("upload")


@login_required(login_url='login')
def document_status_view(request, document_id):
    document = get_object_or_404(Document, id=document_id, owner=request.user)
    
    is_processed = document.is_processed
    insights_ready = False
    try:
        insights = DocumentInsights.objects.get(document=document)
        if insights.status in [DocumentInsights.STATUS_COMPLETED, DocumentInsights.STATUS_FAILED]:
            insights_ready = True
    except DocumentInsights.DoesNotExist:
        # If it hasn't been created yet, it's not ready
        pass

    return JsonResponse({"is_processed": is_processed and insights_ready})