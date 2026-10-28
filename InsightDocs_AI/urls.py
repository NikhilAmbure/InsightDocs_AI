"""
URL configuration for InsightDocs_AI project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from django.views.static import serve
from django.views.generic.base import RedirectView
from django.urls import re_path
from payments import views as payment_views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('auth/', include('accounts.urls')),
    # Legacy / convenience shortcuts redirect to canonical /auth/ routes
    path('login/', RedirectView.as_view(url='/auth/login/', permanent=False)),
    path('signup/', RedirectView.as_view(url='/auth/signup/', permanent=False)),
    path('logout/', RedirectView.as_view(url='/auth/logout/', permanent=False)),
    # Standard Razorpay API routes
    path('api/create-order', payment_views.create_order, name='api_create_order_noslash'),
    path('api/create-order/', payment_views.create_order, name='api_create_order'),
    path('api/verify-payment', payment_views.verify_payment, name='api_verify_payment_noslash'),
    path('api/verify-payment/', payment_views.verify_payment, name='api_verify_payment'),
    # Document-related pages (upload workspace, subscription, etc.)
    path('', include('documents.urls')),
    # Razorpay Payment Gateway
    path('payments/', include('payments.urls')),
    re_path(r'^media/(?P<path>.*)$', serve, {'document_root': settings.MEDIA_ROOT}),
]

if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
    # urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)