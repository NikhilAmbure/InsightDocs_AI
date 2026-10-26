import html
import logging
import os
import json

from django.conf import settings
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.shortcuts import get_object_or_404, redirect, render
from django.http import HttpResponse, Http404, JsonResponse, FileResponse
from django.utils import timezone

from django.urls import reverse
from .forms import DocumentUploadForm
from .models import Document, ChatSession, ChatMessage, DocumentInsights, Workspace
from .utils.rate_limit import check_rate_limit
from .tasks import process_document_task, analyze_document_task
from .utils.rag import process_document_for_rag
from .utils.storage import prepare_local_document

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
    """Handle single or multiple document uploads and display user's documents."""
    form = DocumentUploadForm(request.POST or None, request.FILES or None, user=request.user)

    if request.method == "POST":
        # Free users: 20 uploads per min; Premium users: 60 uploads per min
        limit = 60 if getattr(request.user, 'is_premium', False) else 20
        window = 60
        
        limit_result = check_rate_limit(
            request,
            scope="upload",
            limit=limit,
            window=window,
        )

        is_json = (
            request.headers.get("x-requested-with") == "XMLHttpRequest"
            or request.content_type == "application/json"
            or request.GET.get("format") == "json"
            or "application/json" in request.headers.get("Accept", "")
        )

        if limit_result.limited:
            msg = f"Upload rate limit reached. Please wait {limit_result.retry_after}s and try again."
            if is_json:
                return JsonResponse({"error": msg}, status=429)
            messages.error(request, msg)
            return redirect("upload")
        
        # Collect all uploaded files (handles both 'files' and 'file')
        raw_files = request.FILES.getlist('files') or request.FILES.getlist('file')
        if not raw_files and 'file' in request.FILES:
            raw_files = [request.FILES['file']]

        if not raw_files:
            msg = "Please choose at least one file to upload."
            if is_json:
                return JsonResponse({"error": msg}, status=400)
            messages.error(request, msg)
            return redirect("upload")

        # CHECK STRICT UPLOAD COUNT & FILE SIZE LIMITS
        current_doc_count = Document.objects.filter(owner=request.user).count()
        from payments.models import Subscription
        sub = Subscription.objects.filter(user=request.user).select_related('plan').first()
        is_sub_active = sub and sub.is_active
        if is_sub_active and sub.plan:
            doc_limit = sub.plan.doc_limit
            max_size_mb = sub.plan.max_file_size_mb
        elif getattr(request.user, 'is_premium', False):
            doc_limit = 50
            max_size_mb = 50
        else:
            # STRICT FREE PLAN LIMITS
            doc_limit = 10
            max_size_mb = 10

        remaining_slots = max(0, doc_limit - current_doc_count)

        if current_doc_count >= doc_limit or len(raw_files) > remaining_slots:
            msg = (
                f"Document limit reached: Your current plan allows up to {doc_limit} documents ({current_doc_count} uploaded). "
                f"Upgrade to Pro (₹199 - 50 docs) or Industry (₹1,999 - 200 docs) to upload more."
            )
            if is_json:
                return JsonResponse({"error": msg}, status=403)
            messages.error(request, msg)
            return redirect("upload")

        # Allowed size & extensions
        max_size = max_size_mb * 1024 * 1024
        allowed_extensions = {
            ext.lower().lstrip(".")
            for ext in getattr(settings, "ALLOWED_UPLOAD_EXTENSIONS", [])
        }

        # Optional workspace association if provided
        workspace_id = request.POST.get('workspace_id')
        target_workspace = None
        if workspace_id:
            target_workspace = Workspace.objects.filter(id=workspace_id, owner=request.user).first()

        created_docs = []
        errors = []

        for f in raw_files:
            _, ext = os.path.splitext(f.name)
            clean_ext = ext.lower().lstrip(".")
            if allowed_extensions and clean_ext not in allowed_extensions:
                errors.append(f"'{f.name}': Unsupported file type '.{clean_ext}'.")
                continue
            if f.size > max_size:
                errors.append(f"'{f.name}': Exceeds {max_size_mb} MB limit. Free plan limit is 10 MB. Upgrade to Pro (50 MB) or Industry (100 MB).")
                continue

            # Read file bytes for cross-container cloud availability (Railway Web <-> Celery)
            f.seek(0)
            file_bytes = f.read()
            f.seek(0)

            doc = Document(
                owner=request.user,
                file=f,
                file_content=file_bytes,
                title=f.name,
                original_name=f.name,
            )
            doc.save()
            created_docs.append(doc)

            if target_workspace:
                target_workspace.documents.add(doc)
                target_workspace.save(update_fields=['updated_at'])

            try:
                process_document_task.delay(doc.id)
            except Exception:
                logger.info(f"Celery unavailable, processing doc {doc.id} synchronously.")
                try:
                    local_path, cleanup = prepare_local_document(doc)
                    try:
                        process_document_for_rag(doc, local_path)
                        analyze_document_task(doc.id)
                    finally:
                        cleanup()
                except Exception as proc_err:
                    logger.error(f"Synchronous processing failed for doc {doc.id}: {proc_err}")

            ChatSession.objects.get_or_create(document=doc, user=request.user)

        if not created_docs and errors:
            msg = "Upload failed: " + " ".join(errors)
            if is_json:
                return JsonResponse({"error": msg, "errors": errors}, status=400)
            messages.error(request, msg)
            return redirect("upload")

        if is_json:
            return JsonResponse({
                "status": "success",
                "message": f"{len(created_docs)} file(s) uploaded successfully.",
                "documents": [
                    {
                        "id": d.id,
                        "title": d.title,
                        "extension": d.extension,
                        "url": reverse("serve_document", kwargs={"document_id": d.id}),
                        "chat_url": reverse("chat", kwargs={"document_id": d.id}),
                    }
                    for d in created_docs
                ],
                "errors": errors,
            })

        if errors:
            messages.warning(request, f"Uploaded {len(created_docs)} document(s) with warnings: {' '.join(errors)}")
        else:
            if len(created_docs) > 1:
                messages.success(request, f"Successfully uploaded {len(created_docs)} files!")
            else:
                messages.success(request, f"'{created_docs[0].title}' uploaded successfully!")

        if target_workspace:
            return redirect("workspace_chat", workspace_id=target_workspace.id)

        return redirect("chat", document_id=created_docs[0].id)

    documents = Document.objects.filter(owner=request.user).order_by('-uploaded_at')
    doc_count = documents.count()
    from payments.models import Subscription
    sub = Subscription.objects.filter(user=request.user).select_related('plan').first()
    if request.user.is_premium:
        doc_limit = sub.plan.doc_limit if (sub and sub.plan) else 50
        max_file_size_mb = sub.plan.max_file_size_mb if (sub and sub.plan) else 50
    else:
        doc_limit = 10
        max_file_size_mb = 10

    context = {
        "form": form,
        "recent_documents": documents,
        "doc_count": doc_count,
        "doc_limit": doc_limit,
        "max_file_size_mb": max_file_size_mb,
        "subscription": sub,
    }
    return render(request, "upload.html", context)


@login_required(login_url='login')
def subscription_view(request):
    """Subscription / billing page view."""
    from payments.models import SubscriptionPlan, Subscription, Payment
    from payments.services import ensure_default_plans

    pro_plan = SubscriptionPlan.objects.filter(slug="pro", is_active=True).first()
    industry_plan = SubscriptionPlan.objects.filter(slug="industry", is_active=True).first()
    if not pro_plan or not industry_plan or getattr(pro_plan, 'price_inr', None) != 199 or getattr(industry_plan, 'price_inr', None) != 1999:
        ensure_default_plans()
        pro_plan = SubscriptionPlan.objects.filter(slug="pro", is_active=True).first()
        industry_plan = SubscriptionPlan.objects.filter(slug="industry", is_active=True).first()

    free_plan = SubscriptionPlan.objects.filter(slug="free", is_active=True).first()
    subscription = Subscription.objects.filter(user=request.user).select_related('plan').first()

    # If user is marked premium but has no subscription record yet, link to Pro plan
    if not subscription and request.user.is_premium:
        subscription, _ = Subscription.objects.get_or_create(
            user=request.user,
            defaults={
                'plan': pro_plan,
                'plan_type': 'pro',
                'status': 'active',
                'tokens_allocated': pro_plan.token_quota if pro_plan else 2000000,
                'tokens_used': 0,
            }
        )

    recent_payments = Payment.objects.filter(user=request.user).select_related('plan').order_by('-created_at')[:5]

    return render(request, 'subscription.html', {
        'pro_plan': pro_plan,
        'industry_plan': industry_plan,
        'free_plan': free_plan,
        'subscription': subscription,
        'recent_payments': recent_payments,
    })


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
    
    # Get recent documents & workspaces for sidebar tabs
    recent_docs = Document.objects.filter(owner=request.user).order_by('-uploaded_at')[:20]
    recent_workspaces = Workspace.objects.filter(owner=request.user).order_by('-updated_at')[:20]
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
        "recent_workspaces": recent_workspaces,
        "document_insights": insights,
        "follow_up_questions": follow_up_questions,
    })

@login_required(login_url='login')
def serve_document_view(request, document_id):
    """
    Serves the document file directly (inline streaming for in-browser PDF preview).
    Streams locally with CORS headers for PDF.js.
    """
    document = get_object_or_404(Document, id=document_id, owner=request.user)

    content_type = "application/pdf" if document.is_pdf else "application/octet-stream"
    
    # 1. Resolve local disk path
    from documents.utils.storage import resolve_local_file_path
    local_path = resolve_local_file_path(document)
    if local_path and os.path.exists(local_path):
        response = FileResponse(open(local_path, 'rb'), content_type=content_type)
        response['Content-Disposition'] = f'inline; filename="{document.original_name}"'
        response['Access-Control-Allow-Origin'] = '*'
        return response

    # 2. Try storage backend .open()
    try:
        response = FileResponse(document.file.open('rb'), content_type=content_type)
        response['Content-Disposition'] = f'inline; filename="{document.original_name}"'
        response['Access-Control-Allow-Origin'] = '*'
        return response
    except Exception as e:
        logger.warning(f"document.file.open streaming failed for doc {document.id}: {e}")

    # 3. Stream from database blob if disk file is missing (cross-container Railway resilience)
    if getattr(document, 'file_content', None):
        import io
        response = FileResponse(io.BytesIO(document.file_content), content_type=content_type)
        response['Content-Disposition'] = f'inline; filename="{document.original_name}"'
        response['Access-Control-Allow-Origin'] = '*'
        return response

    # 4. Fallback redirect
    return redirect(document.file.url if document.file else '/')


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
    chunks_count = document.chunks.count()
    stage = "chunking"
    if chunks_count > 0 and not is_processed:
        stage = "indexing"
    elif is_processed:
        stage = "ready"

    insights_ready = False
    try:
        insights = DocumentInsights.objects.get(document=document)
        if insights.status in [DocumentInsights.STATUS_COMPLETED, DocumentInsights.STATUS_FAILED]:
            insights_ready = True
    except DocumentInsights.DoesNotExist:
        pass

    return JsonResponse({
        "is_processed": is_processed,
        "is_ready": is_processed,
        "stage": stage,
        "chunks_count": chunks_count,
        "insights_ready": insights_ready
    })


@login_required(login_url='login')
def search_chats_view(request):
    """Search across all chat history for the current user."""
    query = request.GET.get('q', '').strip()
    results = []

    if query and len(query) >= 2:
        messages_qs = (
            ChatMessage.objects.filter(
                session__user=request.user,
                content__icontains=query,
            )
            .select_related('session', 'session__document', 'session__workspace')
            .order_by('-created_at')[:50]
        )

        for msg in messages_qs:
            doc = msg.session.document
            workspace = msg.session.workspace
            target_title = doc.title if doc else (workspace.name if workspace else "General Chat")
            target_type = "document" if doc else ("workspace" if workspace else "general")
            target_url = ""
            if doc:
                from django.urls import reverse
                target_url = reverse("chat", kwargs={"document_id": doc.id})
            elif workspace:
                from django.urls import reverse
                target_url = reverse("workspace_chat", kwargs={"workspace_id": workspace.id})

            results.append({
                'id': msg.id,
                'role': msg.role,
                'content_preview': msg.content[:240],
                'target_title': target_title,
                'target_type': target_type,
                'target_url': target_url,
                'created_at': msg.created_at.strftime('%b %d, %Y %I:%M %p'),
            })

    if request.headers.get('Accept') == 'application/json' or request.GET.get('format') == 'json':
        return JsonResponse({'results': results, 'query': query})

    return render(request, 'search_results.html', {
        'results': results,
        'query': query,
    })


@login_required(login_url='login')
def export_chat_view(request, document_id):
    """Export document chat history as a standalone downloadable HTML report."""
    document = get_object_or_404(Document, id=document_id, owner=request.user)
    session = ChatSession.objects.filter(document=document, user=request.user).first()

    if not session:
        messages.error(request, 'No chat history found to export.')
        return redirect('chat', document_id=document_id)

    chat_messages = ChatMessage.objects.filter(session=session).order_by('created_at')

    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Chat Export - {html.escape(document.title)}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; max-width: 860px; margin: 0 auto; padding: 40px 20px; background: #020617; color: #e2e8f0; line-height: 1.6; }}
        h1 {{ color: #a78bfa; font-size: 24px; margin-bottom: 4px; }}
        .meta {{ color: #64748b; font-size: 13px; margin-bottom: 28px; border-bottom: 1px solid rgba(255,255,255,0.08); padding-bottom: 12px; }}
        .message {{ margin-bottom: 20px; padding: 18px 20px; border-radius: 12px; }}
        .user {{ background: #0f172a; border-left: 4px solid #a78bfa; }}
        .assistant {{ background: #1e293b; border-left: 4px solid #38bdf8; }}
        .role {{ font-weight: 700; font-size: 12px; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px; display: flex; justify-content: space-between; }}
        .role-user {{ color: #a78bfa; }}
        .role-assistant {{ color: #38bdf8; }}
        .content {{ white-space: pre-wrap; font-size: 14px; }}
        .time {{ color: #475569; font-size: 11px; font-weight: 400; }}
        .footer {{ margin-top: 40px; padding-top: 16px; border-top: 1px solid rgba(255,255,255,0.08); color: #64748b; font-size: 12px; text-align: center; }}
    </style>
</head>
<body>
    <h1>📄 {html.escape(document.title)}</h1>
    <div class="meta">
        Exported on {timezone.now().strftime('%B %d, %Y at %I:%M %p')} · InsightDocs AI
    </div>
"""

    for msg in chat_messages:
        role_class = "user" if msg.role == "user" else "assistant"
        role_label = "👤 You" if msg.role == "user" else "🤖 InsightDocs AI"
        time_str = msg.created_at.strftime('%I:%M %p')
        safe_content = html.escape(msg.content)
        html_content += f"""
    <div class="message {role_class}">
        <div class="role role-{role_class}">
            <span>{role_label}</span>
            <span class="time">{time_str}</span>
        </div>
        <div class="content">{safe_content}</div>
    </div>
"""

    html_content += """
    <div class="footer">InsightDocs AI · Chat with PDF Export</div>
</body>
</html>
"""

    response = HttpResponse(html_content, content_type='text/html')
    safe_title = document.title.replace(' ', '_')[:40]
    response['Content-Disposition'] = f'attachment; filename="chat_{safe_title}.html"'
    return response