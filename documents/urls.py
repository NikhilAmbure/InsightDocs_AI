from django.urls import path

from . import views
from . import views_workspace
from . import views_compare

urlpatterns = [
    # Public landing page
    path("", views.landing_page_view, name="landing_page"),
    path('coming_soon', views.coming_soon, name='coming_soon'),
    path("upload/", views.upload_view, name="upload"),
    path("subscription/", views.subscription_view, name="subscription"),

    # Single Document Chat & Actions
    path("chat/<int:document_id>/", views.chat_view, name="chat"),
    path("chat/<int:document_id>/export/", views.export_chat_view, name="export_chat"),
    path("document/<int:document_id>/serve/", views.serve_document_view, name="serve_document"),
    path("document/<int:document_id>/delete/", views.delete_document_view, name="delete_document"),
    path("document/<int:document_id>/status/", views.document_status_view, name="document_status"),

    # Multi-Document Workspaces
    path("workspaces/", views_workspace.workspace_list_view, name="workspace_list"),
    path("workspaces/create/", views_workspace.workspace_create_view, name="workspace_create"),
    path("workspaces/<int:workspace_id>/", views_workspace.workspace_detail_view, name="workspace_detail"),
    path("workspaces/<int:workspace_id>/chat/", views_workspace.workspace_chat_view, name="workspace_chat"),
    path("workspaces/<int:workspace_id>/status/", views_workspace.workspace_status_view, name="workspace_status"),
    path("workspaces/<int:workspace_id>/export/", views_workspace.export_workspace_chat_view, name="export_workspace_chat"),
    path("workspaces/<int:workspace_id>/add-doc/", views_workspace.workspace_add_document, name="workspace_add_document"),
    path("workspaces/<int:workspace_id>/remove-doc/", views_workspace.workspace_remove_document, name="workspace_remove_document"),
    path("workspaces/<int:workspace_id>/delete/", views_workspace.workspace_delete_view, name="workspace_delete"),

    # Document Comparison
    path("compare/", views_compare.compare_select_view, name="compare_select"),
    path("compare/result/", views_compare.compare_view, name="compare_view"),
    path("compare/analyze/", views_compare.compare_analyze_api, name="compare_analyze"),

    # Cross-document chat search
    path("search/", views.search_chats_view, name="search_chats"),
]