from django.urls import path

from . import views

urlpatterns = [
    # Public landing page
    path("", views.landing_page_view, name="landing_page"),
    path('coming_soon', views.coming_soon, name='coming_soon'),
    path("upload/", views.upload_view, name="upload"),
    path("subscription/", views.subscription_view, name="subscription"),

    path("chat/<int:document_id>/", views.chat_view, name="chat"),
    path("document/<int:document_id>/serve/", views.serve_document_view, name="serve_document"),
    path("document/<int:document_id>/delete/", views.delete_document_view, name="delete_document"),
    path("document/<int:document_id>/status/", views.document_status_view, name="document_status"),
    # path("document/<int:document_id>/editable/", views.document_editable_api_view, name="document_editable_api"),
]