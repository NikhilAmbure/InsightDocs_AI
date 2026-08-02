import logging
from django.db.models import F
from .models import Subscription

logger = logging.getLogger(__name__)


def get_subscription(user):
    if not user or not user.is_authenticated:
        return None
    return getattr(user, "subscription", None)


def has_available_tokens(user) -> bool:
    subscription = get_subscription(user)
    if subscription is None:
        return False
    return subscription.has_tokens_available


def tokens_remaining(user) -> int:
    subscription = get_subscription(user)
    return subscription.tokens_remaining if subscription else 0


def deduct_tokens(user, consumed_tokens: int) -> None:
    """Atomic F() update — same pattern as your Aforro inventory deduction."""
    try:
        consumed_tokens = int(consumed_tokens or 0)
    except (TypeError, ValueError):
        consumed_tokens = 0
    if consumed_tokens <= 0:
        return

    updated = Subscription.objects.filter(user=user).update(
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
        return {
            "is_token_metered": False,
            "plan_type": subscription.plan.slug if subscription and subscription.plan else "free",
            "is_premium": bool(getattr(user, "is_premium", False)),
            "tokens_used": 0, "tokens_allocated": 0, "tokens_remaining": 0,
            "percent_used": 0, "current_period_end": None,
        }

    allocated = subscription.tokens_allocated
    used = subscription.tokens_used
    percent_used = round((used / allocated) * 100, 1) if allocated > 0 else 0

    return {
        "is_token_metered": True,
        "plan_type": subscription.plan.slug,
        "is_premium": subscription.is_active,
        "tokens_used": used,
        "tokens_allocated": allocated,
        "tokens_remaining": subscription.tokens_remaining,
        "percent_used": min(percent_used, 100),
        "current_period_end": subscription.current_period_end.isoformat() if subscription.current_period_end else None,
    }