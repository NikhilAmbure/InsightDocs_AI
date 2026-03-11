from django.urls import path
from . import views

app_name = "payments"

urlpatterns = [
    path("create-order/", views.create_order, name="create_order"),
    path("verify/", views.verify_payment, name="verify_payment"),
    path("webhook/", views.razorpay_webhook, name="razorpay_webhook"),
    path("success/", views.payment_success, name="payment_success"),
    path("failed/", views.payment_failed, name="payment_failed"),
]
