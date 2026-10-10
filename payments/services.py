import logging
# pyrefly: ignore [missing-import]
from django.db.models import F
from .models import Subscription

logger = logging.getLogger(__name__)


def get_subscription(user):
    if not user or not user.is_authenticated:
        return None
    try:
        return getattr(user, "subscription", None)
    except Exception:
        return None


def has_available_tokens(user) -> bool:
    subscription = get_subscription(user)
    if subscription is None:
        # User has no subscription record yet -> default free tier (not token gated)
        return True
    return subscription.has_tokens_available


def ensure_default_plans():
    """
    Ensure standard subscription plans (Free, Pro @ ₹200, Industry @ ₹2,000)
    exist in the database and have current pricing and quotas.
    """
    from .models import SubscriptionPlan
    plans_data = [
        {
            "slug": "free",
            "name": "Free",
            "price_inr": 0,
            "price_display": "₹0",
            "duration_days": 30,
            "doc_limit": 10,
            "chat_limit": 500,
            "max_file_size_mb": 10,
            "token_quota": 0,
            "is_active": True,
        },
        {
            "slug": "pro",
            "name": "Pro",
            "price_inr": 199,
            "price_display": "₹199",
            "duration_days": 30,
            "doc_limit": 50,
            "chat_limit": 5000,
            "max_file_size_mb": 50,
            "token_quota": 2000000,
            "is_active": True,
        },
        {
            "slug": "industry",
            "name": "Industry",
            "price_inr": 1999,
            "price_display": "₹1,999",
            "duration_days": 30,
            "doc_limit": 200,
            "chat_limit": 25000,
            "max_file_size_mb": 100,
            "token_quota": 20000000,
            "is_active": True,
        },
    ]

    result = {}
    for data in plans_data:
        try:
            plan, created = SubscriptionPlan.objects.get_or_create(
                slug=data["slug"],
                defaults=data
            )
            if not created:
                updated_fields = []
                for field in ["price_inr", "price_display", "token_quota", "doc_limit", "chat_limit", "max_file_size_mb", "is_active", "duration_days"]:
                    if getattr(plan, field) != data[field]:
                        setattr(plan, field, data[field])
                        updated_fields.append(field)
                if updated_fields:
                    plan.save(update_fields=updated_fields)
            result[data["slug"]] = plan
        except Exception as e:
            logger.warning(f"ensure_default_plans error for {data['slug']}: {e}")
    return result


def tokens_remaining(user) -> int:
    subscription = get_subscription(user)
    return subscription.tokens_remaining if subscription else 0


def deduct_tokens(user, consumed_tokens: int) -> None:
    """Atomic F() update — deduct from tokens_allocated and track in tokens_used."""
    try:
        consumed_tokens = int(consumed_tokens or 0)
    except (TypeError, ValueError):
        consumed_tokens = 0
    if consumed_tokens <= 0:
        return

    updated = Subscription.objects.filter(user=user).update(
        tokens_allocated=F("tokens_allocated") - consumed_tokens,
        tokens_used=F("tokens_used") + consumed_tokens
    )
    if not updated:
        logger.warning(
            "deduct_tokens: no Subscription row for user_id=%s — %s tokens spent were not recorded.",
            getattr(user, "id", None), consumed_tokens,
        )


def extract_token_usage(response) -> int:
    usage = getattr(response, "usage_metadata", None)
    return getattr(usage, "total_token_count", 0) or 0 if usage else 0

def get_usage_snapshot(user) -> dict:
    subscription = get_subscription(user)
    if subscription is None or not subscription.is_token_metered:
        plan_slug = subscription.plan.slug if (subscription and subscription.plan) else (subscription.plan_type if subscription else "free")
        plan_name = subscription.plan.name if (subscription and subscription.plan) else plan_slug.capitalize()
        return {
            "is_token_metered": False,
            "plan_type": plan_slug,
            "plan_name": plan_name,
            "is_premium": bool(getattr(user, "is_premium", False)),
            "tokens_used": 0,
            "tokens_allocated": 0,
            "tokens_remaining": 0,
            "total_tokens": 0,
            "percent_used": 0,
            "current_period_end": None,
        }

    allocated = max(subscription.tokens_allocated, 0)
    used = max(subscription.tokens_used, 0)
    total_tokens = allocated + used
    percent_used = round((used / total_tokens) * 100, 1) if total_tokens > 0 else 0

    plan_slug = subscription.plan.slug if subscription.plan else subscription.plan_type
    plan_name = subscription.plan.name if subscription.plan else plan_slug.capitalize()

    return {
        "is_token_metered": True,
        "plan_type": plan_slug,
        "plan_name": plan_name,
        "is_premium": subscription.is_active,
        "tokens_used": used,
        "tokens_allocated": allocated,
        "tokens_remaining": subscription.tokens_remaining,
        "total_tokens": total_tokens,
        "percent_used": min(percent_used, 100),
        "current_period_end": subscription.current_period_end.isoformat() if subscription.current_period_end else None,
    }