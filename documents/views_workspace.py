import html
import json
import logging

from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.http import HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone
from django.views.decorators.http import require_POST

from .models import Document, Workspace, ChatSession, ChatMessage

logger = logging.getLogger(__name__)


@login_required(login_url='login')
def workspace_list_view(request):
    """List all workspaces for the current user."""
    workspaces = Workspace.objects.filter(owner=request.user).prefetch_related('documents')
    return render(request, 'workspaces.html', {
        'workspaces': workspaces,
    })


@login_required(login_url='login')
@require_POST
def workspace_create_view(request):
    """Create a new workspace."""
    name = request.POST.get('name', '').strip()
    description = request.POST.get('description', '').strip()

    if not name:
        messages.error(request, 'Workspace name is required.')
        return redirect('workspace_list')

    # Workspace limit: 200 for Industry, 50 for Pro, 3 for Free
    is_premium = getattr(request.user, 'is_premium', False)
    max_workspaces = 50 if is_premium else 3
    current_count = Workspace.objects.filter(owner=request.user).count()
    if current_count >= max_workspaces:
        if not is_premium:
            messages.error(request, f'Strict limit reached: Free plan allows up to {max_workspaces} workspaces. Upgrade to Pro (₹199 - 50 workspaces) or Industry (₹1,999 - 200 workspaces) to create more.')
        else:
            messages.error(request, f'You have reached your workspace limit of {max_workspaces}.')
        return redirect('workspace_list')

    workspace, created = Workspace.objects.get_or_create(
        name=name,
        owner=request.user,
        defaults={'description': description},
    )
    if not created:
        messages.info(request, f'Workspace "{name}" already exists.')
    else:
        messages.success(request, f'Workspace "{name}" created successfully.')

    return redirect('workspace_detail', workspace_id=workspace.id)


@login_required(login_url='login')
def workspace_detail_view(request, workspace_id):
    """View workspace details, manage documents, and see recent chat."""
    workspace = get_object_or_404(Workspace, id=workspace_id, owner=request.user)

    workspace_docs = workspace.documents.all().order_by('-uploaded_at')
    workspace_doc_ids = workspace_docs.values_list('id', flat=True)

    # Documents available to be added
    available_docs = Document.objects.filter(owner=request.user).exclude(
        id__in=workspace_doc_ids
    ).order_by('-uploaded_at')

    session = ChatSession.objects.filter(workspace=workspace, user=request.user).first()
    chat_history = []
    if session:
        chat_history = ChatMessage.objects.filter(session=session).order_by('created_at')[:10]

    return render(request, 'workspace_detail.html', {
        'workspace': workspace,
        'workspace_docs': workspace_docs,
        'available_docs': available_docs,
        'chat_history': chat_history,
        'session': session,
    })


@login_required(login_url='login')
@require_POST
def workspace_add_document(request, workspace_id):
    """Add a document to a workspace."""
    workspace = get_object_or_404(Workspace, id=workspace_id, owner=request.user)
    doc_id = request.POST.get('document_id')

    if doc_id:
        doc = get_object_or_404(Document, id=doc_id, owner=request.user)
        workspace.documents.add(doc)
        workspace.save(update_fields=['updated_at'])
        messages.success(request, f'"{doc.title}" added to {workspace.name}.')

    return redirect('workspace_detail', workspace_id=workspace.id)


@login_required(login_url='login')
@require_POST
def workspace_remove_document(request, workspace_id):
    """Remove a document from a workspace."""
    workspace = get_object_or_404(Workspace, id=workspace_id, owner=request.user)
    doc_id = request.POST.get('document_id')

    if doc_id:
        doc = get_object_or_404(Document, id=doc_id, owner=request.user)
        workspace.documents.remove(doc)
        workspace.save(update_fields=['updated_at'])
        messages.success(request, f'"{doc.title}" removed from {workspace.name}.')

    return redirect('workspace_detail', workspace_id=workspace.id)


@login_required(login_url='login')
@require_POST
def workspace_delete_view(request, workspace_id):
    """Delete a workspace (does not delete underlying documents)."""
    workspace = get_object_or_404(Workspace, id=workspace_id, owner=request.user)
    name = workspace.name
    workspace.delete()
    messages.success(request, f'Workspace "{name}" deleted successfully.')
    return redirect('workspace_list')


@login_required(login_url='login')
def workspace_chat_view(request, workspace_id):
    """Dedicated cross-document chat view for a workspace."""
    workspace = get_object_or_404(Workspace, id=workspace_id, owner=request.user)
    workspace_docs = workspace.documents.all()

    if not workspace_docs.exists():
        messages.warning(
            request,
            'No documents in this workspace yet. Please add documents to start chatting.'
        )
        return redirect('workspace_detail', workspace_id=workspace.id)

    session, _ = ChatSession.objects.get_or_create(
        workspace=workspace,
        user=request.user,
        defaults={'title': f'Chat: {workspace.name}'}
    )

    chat_history = ChatMessage.objects.filter(session=session).order_by('created_at')

    # Suggested multi-document questions
    suggested_questions = [
        f"Compare the key findings across all documents in {workspace.name}",
        "What are the main common themes and differences?",
        "Provide an executive summary synthesizing these documents",
    ]

    # Primary document for default PDF preview
    primary_doc = workspace_docs.first()

    # Serialized documents list for instant client-side PDF switching & citations
    import json
    from django.urls import reverse
    docs_metadata = []
    for d in workspace_docs:
        docs_metadata.append({
            'id': d.id,
            'title': d.title,
            'extension': d.extension,
            'url': reverse('serve_document', kwargs={'document_id': d.id}),
        })

    recent_workspaces = Workspace.objects.filter(owner=request.user).order_by('-updated_at')[:20]
    recent_documents = Document.objects.filter(owner=request.user).order_by('-uploaded_at')[:20]

    total_docs = workspace_docs.count()
    processed_count = workspace_docs.filter(is_processed=True).count()
    is_all_processed = (total_docs > 0 and processed_count == total_docs)

    return render(request, 'chat.html', {
        'workspace': workspace,
        'workspace_docs': workspace_docs,
        'document': primary_doc,
        'primary_doc': primary_doc,
        'docs_metadata_json': json.dumps(docs_metadata),
        'recent_workspaces': recent_workspaces,
        'recent_documents': recent_documents,
        'chat_history': chat_history,
        'session': session,
        'suggested_questions': suggested_questions,
        'follow_up_questions': suggested_questions,
        'is_workspace_processed': is_all_processed,
        'total_docs': total_docs,
        'processed_docs': processed_count,
    })


@login_required(login_url='login')
def export_workspace_chat_view(request, workspace_id):
    """Export workspace chat history as a clean, styled HTML report."""
    workspace = get_object_or_404(Workspace, id=workspace_id, owner=request.user)
    session = ChatSession.objects.filter(workspace=workspace, user=request.user).first()

    if not session:
        messages.error(request, 'No chat history to export.')
        return redirect('workspace_detail', workspace_id=workspace.id)

    chat_messages = ChatMessage.objects.filter(session=session).order_by('created_at')
    doc_titles = ", ".join([d.title for d in workspace.documents.all()[:5]])

    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Workspace Chat Export - {html.escape(workspace.name)}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; max-width: 860px; margin: 0 auto; padding: 40px 20px; background: #020617; color: #e2e8f0; line-height: 1.6; }}
        h1 {{ color: #a78bfa; font-size: 24px; margin-bottom: 4px; }}
        .meta {{ color: #64748b; font-size: 13px; margin-bottom: 28px; border-bottom: 1px solid rgba(255,255,255,0.08); padding-bottom: 12px; }}
        .badge {{ display: inline-block; background: rgba(167, 139, 250, 0.15); color: #c4b5fd; padding: 2px 8px; border-radius: 6px; font-size: 11px; margin-right: 6px; }}
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
    <h1>📁 {html.escape(workspace.name)}</h1>
    <div class="meta">
        <div><strong>Workspace Documents:</strong> {html.escape(doc_titles or "None")}</div>
        <div style="margin-top: 4px;">Exported on {timezone.now().strftime('%B %d, %Y at %I:%M %p')} · InsightDocs AI</div>
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
    <div class="footer">InsightDocs AI · Multi-Document Workspace Export</div>
</body>
</html>
"""

    response = HttpResponse(html_content, content_type='text/html')
    safe_name = workspace.name.replace(' ', '_')[:40]
    response['Content-Disposition'] = f'attachment; filename="workspace_{safe_name}_chat.html"'
    return response


@login_required(login_url='login')
def workspace_status_view(request, workspace_id):
    """Returns whether all documents in the workspace are indexed and processed."""
    workspace = get_object_or_404(Workspace, id=workspace_id, owner=request.user)
    docs = workspace.documents.all()
    total = docs.count()
    if total == 0:
        return JsonResponse({"is_processed": False, "total_docs": 0, "processed_docs": 0})

    processed_count = docs.filter(is_processed=True).count()
    return JsonResponse({
        "is_processed": processed_count == total,
        "total_docs": total,
        "processed_docs": processed_count,
    })


