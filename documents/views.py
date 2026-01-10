# views.py
import logging
import os

from django.conf import settings
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.shortcuts import get_object_or_404, redirect, render
from django.http import HttpResponse, Http404

from .forms import DocumentUploadForm
from .models import Document, ChatSession, ChatMessage
from .utils.rate_limit import check_rate_limit

logger = logging.getLogger(__name__)


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
        upload_rate_limit = getattr(settings, "RATE_LIMITS", {}).get(
            "upload",
            {"limit": 5, "window": 60},
        )
        limit_result = check_rate_limit(
            request,
            scope="upload",
            limit=upload_rate_limit.get("limit", 5),
            window=upload_rate_limit.get("window", 60),
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
    return render(request, 'subscription.html')


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

    return render(request, "chat.html", {
        "document": document,
        "chat_history": chat_history_qs,
        "recent_documents": recent_docs,
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
        
        # Delete the actual file from storage (CLoudinary)
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