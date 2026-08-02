import logging
import json
import uuid
from datetime import timedelta

import razorpay
from django.conf import settings
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse, HttpResponseBadRequest
from django.shortcuts import redirect, render, get_object_or_404
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST

from .models import SubscriptionPlan, Subscription, Payment
from .services import get_usage_snapshot

logger = logging.getLogger(__name__)


def _get_razorpay_client():
    """Return a configured Razorpay client."""
    return razorpay.Client(
        auth=(settings.RAZORPAY_KEY_ID, settings.RAZORPAY_KEY_SECRET)
    )


# ─────────────────────── Create Razorpay Order ───────────────────────
@login_required(login_url="login")
@require_POST
def create_order(request):
    """
    AJAX endpoint — creates a Razorpay order and returns the order details
    for the frontend checkout.
    Expects POST JSON: { "plan_id": <int> }
    """
    # Validate Razorpay keys are configured
    if not settings.RAZORPAY_KEY_ID or not settings.RAZORPAY_KEY_SECRET:
        logger.error("Razorpay API keys are not configured!")
        return JsonResponse(
            {"error": "Payment gateway is not configured. Please contact the administrator."},
            status=500,
        )

    try:
        body = json.loads(request.body)
        plan_id = body.get("plan_id")
    except (json.JSONDecodeError, AttributeError):
        return JsonResponse({"error": "Invalid request body."}, status=400)

    plan = get_object_or_404(SubscriptionPlan, id=plan_id, is_active=True)

    if plan.price_inr <= 0:
        return JsonResponse({"error": "Cannot purchase a free plan."}, status=400)

    client = _get_razorpay_client()

    internal_order_id = f"INS-{uuid.uuid4().hex[:12].upper()}"

    try:
        razorpay_order = client.order.create(
            {
                "amount": plan.price_paise,
                "currency": "INR",
                "receipt": internal_order_id,
                "notes": {
                    "user_id": str(request.user.id),
                    "plan_id": str(plan.id),
                    "plan_name": plan.name,
                },
            }
        )
    except razorpay.errors.BadRequestError as e:
        logger.error(f"Razorpay order creation failed: {e}")
        return JsonResponse(
            {"error": "Payment gateway authentication failed. Please check your Razorpay API keys."},
            status=502,
        )
    except Exception as e:
        logger.error(f"Unexpected error creating Razorpay order: {e}")
        return JsonResponse(
            {"error": "Failed to create payment order. Please try again."},
            status=500,
        )

    # Persist a Payment record with status=created
    Payment.objects.create(
        order_id=internal_order_id,
        user=request.user,
        plan=plan,
        razorpay_order_id=razorpay_order["id"],
        amount=plan.price_inr,
        currency="INR",
        status="created",
    )

    return JsonResponse(
        {
            "order_id": razorpay_order["id"],
            "amount": plan.price_paise,
            "currency": "INR",
            "key_id": settings.RAZORPAY_KEY_ID,
            "plan_name": plan.name,
            "user_email": request.user.email,
            "user_name": request.user.get_full_name() or request.user.username,
        }
    )


# ─────────── Verify Payment (client‑side callback) ──────────────────
@login_required(login_url="login")
@require_POST
def verify_payment(request):
    """
    Called from the frontend after Razorpay checkout success.
    Expects POST JSON with razorpay_order_id, razorpay_payment_id, razorpay_signature.
    """
    try:
        body = json.loads(request.body)
    except (json.JSONDecodeError, AttributeError):
        return JsonResponse({"error": "Invalid request body."}, status=400)

    razorpay_order_id = body.get("razorpay_order_id")
    razorpay_payment_id = body.get("razorpay_payment_id")
    razorpay_signature = body.get("razorpay_signature")

    if not all([razorpay_order_id, razorpay_payment_id, razorpay_signature]):
        return JsonResponse({"error": "Missing payment parameters."}, status=400)

    # Look up the local Payment record
    try:
        payment = Payment.objects.get(razorpay_order_id=razorpay_order_id, user=request.user)
    except Payment.DoesNotExist:
        return JsonResponse({"error": "Payment record not found."}, status=404)

    # Verify signature
    client = _get_razorpay_client()
    try:
        client.utility.verify_payment_signature(
            {
                "razorpay_order_id": razorpay_order_id,
                "razorpay_payment_id": razorpay_payment_id,
                "razorpay_signature": razorpay_signature,
            }
        )
    except razorpay.errors.SignatureVerificationError:
        payment.status = "failed"
        payment.save(update_fields=["status", "updated_at"])
        logger.warning(f"Signature verification failed for order {razorpay_order_id}")
        return JsonResponse({"error": "Payment verification failed."}, status=400)

    # ── Payment is valid ──
    payment.razorpay_payment_id = razorpay_payment_id
    payment.razorpay_signature = razorpay_signature
    payment.status = "captured"
    payment.save(update_fields=["razorpay_payment_id", "razorpay_signature", "status", "updated_at"])

    # Activate / update the user's subscription
    _activate_subscription(request.user, payment.plan)

    logger.info(f"Payment captured: {payment.order_id} for user {request.user.username}")

    return JsonResponse({"status": "success", "message": "Payment verified and subscription activated!"})


# ─────────── Razorpay Webhook (server‑side verification) ────────────
@csrf_exempt
@require_POST
def razorpay_webhook(request):
    """
    Handles Razorpay webhook events (payment.captured, payment.failed, etc.).
    Verify the webhook signature using X-Razorpay-Signature header.
    """
    webhook_secret = getattr(settings, "RAZORPAY_WEBHOOK_SECRET", "")

    if not webhook_secret:
        logger.error("RAZORPAY_WEBHOOK_SECRET is not configured!")
        return JsonResponse({"error": "Webhook not configured."}, status=500)

    signature = request.headers.get("X-Razorpay-Signature", "")
    payload = request.body.decode("utf-8")

    client = _get_razorpay_client()
    try:
        client.utility.verify_webhook_signature(payload, signature, webhook_secret)
    except razorpay.errors.SignatureVerificationError:
        logger.warning("Webhook signature verification failed.")
        return JsonResponse({"error": "Invalid signature."}, status=400)

    # Parse event
    try:
        event = json.loads(payload)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON payload."}, status=400)

    event_type = event.get("event", "")
    payment_entity = event.get("payload", {}).get("payment", {}).get("entity", {})

    razorpay_order_id = payment_entity.get("order_id")
    razorpay_payment_id = payment_entity.get("id")

    if not razorpay_order_id:
        return JsonResponse({"status": "ignored"})

    try:
        payment = Payment.objects.get(razorpay_order_id=razorpay_order_id)
    except Payment.DoesNotExist:
        logger.warning(f"Webhook: Payment not found for order {razorpay_order_id}")
        return JsonResponse({"status": "not_found"}, status=404)

    if event_type == "payment.captured":
        if payment.status == "captured":
            logger.info(f"Webhook: order {razorpay_order_id} already captured, skipping.")
            return JsonResponse({"status": "already_processed"})

        payment.razorpay_payment_id = razorpay_payment_id
        payment.status = "captured"
        payment.save(update_fields=["razorpay_payment_id", "status", "updated_at"])
        _activate_subscription(payment.user, payment.plan)

    elif event_type == "payment.failed":
        payment.razorpay_payment_id = razorpay_payment_id
        payment.status = "failed"
        payment.save(update_fields=["razorpay_payment_id", "status", "updated_at"])
        logger.info(f"Webhook: payment.failed for order {razorpay_order_id}")

    return JsonResponse({"status": "ok"})


# ────────── Payment Success & Failure Redirect pages ─────────────────
@login_required(login_url="login")
def payment_success(request):
    """Display a success page after payment."""
    return render(request, "payments/success.html")


@login_required(login_url="login")
def payment_failed(request):
    """Display a failure page after payment."""
    return render(request, "payments/failed.html")


# ────────────────────── Helper ───────────────────────────────────────
def _activate_subscription(user, plan):
    """Create or update the user's subscription and set is_premium."""
    now = timezone.now()
    end_date = now + timedelta(days=plan.duration_days)

    subscription, created = Subscription.objects.update_or_create(
        user=user,
        defaults={
            "plan": plan,
            "status": "active",
            "start_date": now,
            "end_date": end_date,
            "tokens_allocated": plan.token_quota,
            "tokens_used": 0,
            "tokens_granted_at": now,
        },
    )

    # Update the user's premium flag
    user.is_premium = True
    user.save(update_fields=["is_premium"])
    return subscription

@login_required(login_url="login")
def usage_status(request):
    return JsonResponse(get_usage_snapshot(request.user))