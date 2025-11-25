from django.urls import path

from . import views

urlpatterns = [
    # Public landing page
    path("", views.landing_page_view, name="landing_page"),
    path('coming_soon', views.coming_soon, name='coming_soon'),
    path("upload/", views.upload_view, name="upload"),
    path("subscription/", views.subscription_view, name="subscription"),

    path("chat/<int:document_id>/", views.chat_view, name="chat"),
    
]