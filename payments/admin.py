from django.contrib import admin
from .models import SubscriptionPlan, Subscription, Payment


@admin.register(SubscriptionPlan)
class SubscriptionPlanAdmin(admin.ModelAdmin):
    list_display = ("name", "slug", "price_inr", "duration_days", "doc_limit", "chat_limit", "is_active")
    list_filter = ("is_active",)
    prepopulated_fields = {"slug": ("name",)}


@admin.register(Subscription)
class SubscriptionAdmin(admin.ModelAdmin):
    list_display = ("user", "plan", "status", "start_date", "end_date")
    list_filter = ("status", "plan")
    search_fields = ("user__username", "user__email")
    raw_id_fields = ("user",)


@admin.register(Payment)
class PaymentAdmin(admin.ModelAdmin):
    list_display = ("order_id", "user", "plan", "amount", "currency", "status", "created_at")
    list_filter = ("status", "currency", "plan")
    search_fields = ("order_id", "razorpay_order_id", "razorpay_payment_id", "user__username")
    readonly_fields = ("razorpay_order_id", "razorpay_payment_id", "razorpay_signature")
    raw_id_fields = ("user",)
