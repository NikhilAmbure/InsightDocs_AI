import uuid
from django.conf import settings
from django.db import models
from django.utils import timezone


class SubscriptionPlan(models.Model):
    """
    Stores the different subscription tiers.
    """
    name = models.CharField(max_length=50, unique=True)              # e.g. "Free", "Pro", "Enterprise"
    slug = models.SlugField(unique=True)                              # e.g. "free", "pro", "enterprise"
    price_inr = models.DecimalField(max_digits=10, decimal_places=2)  # Price in INR (₹)
    price_display = models.CharField(max_length=20, blank=True)       # e.g. "₹799", "$9.99"
    duration_days = models.IntegerField(default=30)     
              # Billing cycle length
    doc_limit = models.IntegerField(default=10)
    chat_limit = models.IntegerField(default=500)
    max_file_size_mb = models.IntegerField(default=10)
    is_active = models.BooleanField(default=True)

    # Token-metered usage (pay-as-you-go against the master Gemini key).
    # 0 = "not token metered" — falls back to doc_limit/chat_limit instead
    # (e.g. the Free plan). Pro-style plans set this to the purchased budget,
    # e.g. 5_000_000 for a $10 top-up.
    token_quota = models.BigIntegerField(
        default=0,
        help_text="LLM tokens granted per billing cycle. 0 = not token-metered.",
    )

    class Meta:
        ordering = ("price_inr",)

    def __str__(self):
        return f"{self.name} – ₹{self.price_inr}"

    @property
    def price_paise(self):
        """Razorpay expects amount in paise (smallest currency unit)."""
        return int(self.price_inr * 100)


class Subscription(models.Model):
    """
    Tracks a user's active subscription.
    """
    STATUS_CHOICES = (
        ("active", "Active"),
        ("expired", "Expired"),
        ("cancelled", "Cancelled"),
    )

    user = models.OneToOneField(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="subscription",
    )
    plan = models.ForeignKey(
        SubscriptionPlan,
        on_delete=models.SET_NULL,
        null=True,
        related_name="subscriptions",
    )
    plan_type = models.CharField(max_length=50, default="free", help_text="Plan type (e.g., free, pro).")
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="active")
    start_date = models.DateTimeField(default=timezone.now)
    end_date = models.DateTimeField(null=True, blank=True)

    razorpay_subscription_id = models.CharField(max_length=255, blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    tokens_allocated = models.IntegerField(default=0, help_text="Tracks remaining token balance.")
    tokens_used = models.BigIntegerField(default=0)
    tokens_granted_at = models.DateTimeField(
        null=True, blank=True, help_text="When the current token bucket was last topped up."
    )

    @property
    def current_period_end(self):
        return self.end_date  # you already have end_date

    @property
    def tokens_remaining(self):
        return max(self.tokens_allocated, 0)

    @property
    def is_token_metered(self):
        return bool(self.plan and self.plan.token_quota > 0) or self.plan_type == 'pro'

    @property
    def has_tokens_available(self):
        if not self.is_active:
            return False
        if not self.is_token_metered:
            return True  # not gated here — Free plan uses chat_limit instead
        return self.tokens_allocated > 0

    def __str__(self):
        return f"{self.user.username} – {self.plan.name if self.plan else self.plan_type} ({self.status})"

    @property
    def is_active(self):
        if self.status != "active":
            return False
        if self.end_date and timezone.now() > self.end_date:
            return False
        return True


class Payment(models.Model):
    """
    Stores each individual payment transaction.
    """
    STATUS_CHOICES = (
        ("created", "Created"),
        ("authorized", "Authorized"),
        ("captured", "Captured"),
        ("failed", "Failed"),
        ("refunded", "Refunded"),
    )

    order_id = models.CharField(max_length=100, unique=True, db_index=True)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="payments",
    )
    plan = models.ForeignKey(
        SubscriptionPlan,
        on_delete=models.SET_NULL,
        null=True,
        related_name="payments",
    )
    razorpay_order_id = models.CharField(max_length=255, unique=True, db_index=True)
    razorpay_payment_id = models.CharField(max_length=255, blank=True, null=True)
    razorpay_signature = models.CharField(max_length=512, blank=True, null=True)
    amount = models.DecimalField(max_digits=10, decimal_places=2)
    currency = models.CharField(max_length=10, default="INR")
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="created")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ("-created_at",)

    def __str__(self):
        return f"Payment {self.order_id} – {self.user.username} – {self.status}"
