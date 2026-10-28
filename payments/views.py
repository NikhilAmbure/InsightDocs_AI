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
@csrf_exempt
@require_POST
def create_order(request):
    """
    Creates a Razorpay order and returns the order details for the frontend checkout.
    Supports:
      1. Direct amount: { "amount": <paise (min 100)>, "currency": "INR", "receipt": "..." }
      2. Plan-based:   { "plan_id": <int> }
    """
    # Validate Razorpay keys are configured
    if not settings.RAZORPAY_KEY_ID or not settings.RAZORPAY_KEY_SECRET:
        logger.error("Razorpay API keys are not configured!")
        return JsonResponse(
            {"error": "Payment gateway is not configured. Please contact the administrator."},
            status=500,
        )

    try:
        body = json.loads(request.body) if request.body else {}
    except (json.JSONDecodeError, AttributeError):
        return JsonResponse({"error": "Invalid request body."}, status=400)

    amount = body.get("amount")
    currency = body.get("currency", "INR")
    receipt = body.get("receipt")
    plan_id = body.get("plan_id")

    plan = None
    amount_paise = None

    if amount is not None:
        try:
            amount_paise = int(amount)
        except (ValueError, TypeError):
            return JsonResponse({"error": "Amount must be an integer in paise."}, status=400)

        # Minimum amount validation: at least 100 paise (₹1)
        if amount_paise < 100:
            return JsonResponse({"error": "Minimum amount must be at least 100 paise."}, status=400)

    elif plan_id is not None:
        from .services import ensure_default_plans
        try:
            plan = SubscriptionPlan.objects.get(id=plan_id, is_active=True)
        except SubscriptionPlan.DoesNotExist:
            ensure_default_plans()
            try:
                plan = SubscriptionPlan.objects.get(id=plan_id, is_active=True)
            except SubscriptionPlan.DoesNotExist:
                return JsonResponse({"error": "Selected plan is not available."}, status=404)

        if plan.price_paise < 100:
            return JsonResponse({"error": "Cannot purchase a plan below 100 paise."}, status=400)
        amount_paise = plan.price_paise
    else:
        return JsonResponse({"error": "Either amount (in paise >= 100) or plan_id is required."}, status=400)

    client = _get_razorpay_client()
    internal_order_id = receipt or f"INS-{uuid.uuid4().hex[:12].upper()}"

    notes = {}
    if request.user.is_authenticated:
        notes["user_id"] = str(request.user.id)
    if plan:
        notes["plan_id"] = str(plan.id)
        notes["plan_name"] = plan.name

    try:
        razorpay_order = client.order.create(
            {
                "amount": amount_paise,
                "currency": currency,
                "receipt": internal_order_id,
                "notes": notes,
            }
        )
    except (razorpay.errors.BadRequestError, razorpay.errors.GatewayError) as e:
        err_msg = str(e).lower()
        logger.error(f"Razorpay order creation failed: {e}")
        if "auth" in err_msg or "unauthorized" in err_msg or "invalid key" in err_msg:
            return JsonResponse({"error": "Razorpay authentication failed."}, status=401)
        return JsonResponse({"error": f"Razorpay API error: {str(e)}"}, status=500)
    except Exception as e:
        err_msg = str(e).lower()
        logger.error(f"Unexpected error creating Razorpay order: {e}")
        if "auth" in err_msg or "unauthorized" in err_msg:
            return JsonResponse({"error": "Razorpay authentication failed."}, status=401)
        return JsonResponse({"error": "Failed to create payment order. Please try again."}, status=500)

    # If associated with a user and plan, persist local Payment record
    if request.user.is_authenticated and plan:
        Payment.objects.create(
            order_id=internal_order_id,
            user=request.user,
            plan=plan,
            razorpay_order_id=razorpay_order["id"],
            amount=plan.price_inr,
            currency=currency,
            status="created",
        )

    response_payload = {
        "order_id": razorpay_order["id"],
        "amount": razorpay_order["amount"],
        "currency": razorpay_order["currency"],
        "key_id": settings.RAZORPAY_KEY_ID,
    }

    if plan:
        response_payload["plan_name"] = plan.name
    if request.user.is_authenticated:
        response_payload["user_email"] = request.user.email
        response_payload["user_name"] = request.user.get_full_name() or request.user.username

    return JsonResponse(response_payload)


# ─────────── Verify Payment (client‑side callback) ──────────────────
@csrf_exempt
@require_POST
def verify_payment(request):
    """
    Called from frontend or API after Razorpay checkout success.
    Expects POST JSON with razorpay_order_id, razorpay_payment_id, razorpay_signature.
    Verifies cryptographic signature: HMAC-SHA256(order_id + "|" + payment_id, KEY_SECRET).
    """
    import hmac
    import hashlib

    try:
        body = json.loads(request.body) if request.body else {}
    except (json.JSONDecodeError, AttributeError):
        return JsonResponse({"error": "Invalid request body."}, status=400)

    razorpay_order_id = body.get("razorpay_order_id")
    razorpay_payment_id = body.get("razorpay_payment_id")
    razorpay_signature = body.get("razorpay_signature")

    if not all([razorpay_order_id, razorpay_payment_id, razorpay_signature]):
        return JsonResponse(
            {"error": "Missing payment parameters. razorpay_order_id, razorpay_payment_id, and razorpay_signature are required."},
            status=400,
        )

    # 1. Cryptographic HMAC-SHA256 signature verification
    msg = f"{razorpay_order_id}|{razorpay_payment_id}".encode("utf-8")
    secret = settings.RAZORPAY_KEY_SECRET.encode("utf-8")
    generated_signature = hmac.new(secret, msg, hashlib.sha256).hexdigest()

    if not hmac.compare_digest(generated_signature, razorpay_signature):
        logger.warning(f"Signature mismatch for order {razorpay_order_id}")
        if request.user.is_authenticated:
            Payment.objects.filter(razorpay_order_id=razorpay_order_id, user=request.user).update(
                status="failed", updated_at=timezone.now()
            )
        return JsonResponse({"error": "Payment verification failed. Signature mismatch.", "status": "failed"}, status=400)

    # 2. Signature is valid — update local database if Payment record exists
    if request.user.is_authenticated:
        payment = Payment.objects.filter(razorpay_order_id=razorpay_order_id, user=request.user).first()
        if payment:
            # Idempotency check: if already captured, do not re-credit
            if payment.status != "captured":
                payment.razorpay_payment_id = razorpay_payment_id
                payment.razorpay_signature = razorpay_signature
                payment.status = "captured"
                payment.save(update_fields=["razorpay_payment_id", "razorpay_signature", "status", "updated_at"])

                # Activate subscription if plan attached
                if payment.plan:
                    _activate_subscription(request.user, payment.plan)
                logger.info(f"Payment captured: {payment.order_id} for user {request.user.username}")

    return JsonResponse({"status": "success", "message": "Payment verified successfully!"})


# ─────────── Razorpay Webhook (server‑side verification) ────────────
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status as drf_status
from django.utils.decorators import method_decorator

@method_decorator(csrf_exempt, name='dispatch')
class RazorpayWebhookView(APIView):
    authentication_classes = []
    permission_classes = []

    def post(self, request, *args, **kwargs):
        webhook_secret = getattr(settings, "RAZORPAY_WEBHOOK_SECRET", "")

        if not webhook_secret:
            logger.error("RAZORPAY_WEBHOOK_SECRET is not configured!")
            return Response({"error": "Webhook secret not configured."}, status=drf_status.HTTP_500_INTERNAL_SERVER_ERROR)

        signature = request.headers.get("X-Razorpay-Signature", "")
        payload = request.body.decode("utf-8")

        client = _get_razorpay_client()
        try:
            client.utility.verify_webhook_signature(payload, signature, webhook_secret)
        except razorpay.errors.SignatureVerificationError:
            logger.warning("Webhook signature verification failed.")
            return Response({"error": "Invalid signature."}, status=drf_status.HTTP_400_BAD_REQUEST)

        try:
            event = json.loads(payload)
        except json.JSONDecodeError:
            return Response({"error": "Invalid JSON payload."}, status=drf_status.HTTP_400_BAD_REQUEST)

        event_type = event.get("event", "")
        payload_data = event.get("payload", {})
        payment_entity = payload_data.get("payment", {}).get("entity", {})
        subscription_entity = payload_data.get("subscription", {}).get("entity", {})

        user = None
        plan = None
        payment = None

        if event_type in ["payment.captured", "subscription.charged"]:
            notes = payment_entity.get("notes", {}) or subscription_entity.get("notes", {}) or {}
            user_id = notes.get("user_id")
            plan_id = notes.get("plan_id")
            razorpay_order_id = payment_entity.get("order_id")

            # Check if this payment was already processed
            if razorpay_order_id:
                try:
                    payment = Payment.objects.get(razorpay_order_id=razorpay_order_id)
                    user = payment.user
                    plan = payment.plan
                except Payment.DoesNotExist:
                    pass

            if payment and payment.status == "captured":
                logger.info(f"Webhook: Payment {razorpay_order_id} already captured and credited. Deduplicating.")
                return Response({"status": "success", "message": "Payment already processed."}, status=drf_status.HTTP_200_OK)

            razorpay_sub_id = subscription_entity.get("id") or payment_entity.get("subscription_id")
            if not user and razorpay_sub_id:
                try:
                    sub_record = Subscription.objects.get(razorpay_subscription_id=razorpay_sub_id)
                    user = sub_record.user
                    plan = sub_record.plan
                except Subscription.DoesNotExist:
                    pass

            from accounts.models import User
            if not user and user_id:
                try:
                    user = User.objects.get(id=user_id)
                except User.DoesNotExist:
                    pass

            if not plan and plan_id:
                try:
                    plan = SubscriptionPlan.objects.get(id=plan_id)
                except SubscriptionPlan.DoesNotExist:
                    pass

            if user:
                # Standardize activation through helper to prevent logic drift
                sub = _activate_subscription(user, plan)
                if razorpay_sub_id and not sub.razorpay_subscription_id:
                    sub.razorpay_subscription_id = razorpay_sub_id
                    sub.save(update_fields=["razorpay_subscription_id"])

                if razorpay_order_id:
                    Payment.objects.filter(razorpay_order_id=razorpay_order_id).update(
                        status="captured",
                        razorpay_payment_id=payment_entity.get("id") or "",
                        updated_at=timezone.now()
                    )

                logger.info(f"Webhook success: Activated subscription for user {user.username}")
                return Response({"status": "success", "message": "Subscription activated successfully."}, status=drf_status.HTTP_200_OK)
            else:
                logger.warning("Webhook user identification failed.")
                return Response({"error": "User identification failed."}, status=drf_status.HTTP_400_BAD_REQUEST)

        elif event_type in ["subscription.cancelled", "subscription.halted", "subscription.paused"]:
            razorpay_sub_id = subscription_entity.get("id") or payment_entity.get("subscription_id")
            if razorpay_sub_id:
                sub = Subscription.objects.filter(razorpay_subscription_id=razorpay_sub_id).first()
                if sub:
                    sub.status = "cancelled"
                    sub.save(update_fields=["status", "updated_at"])
                    if sub.user:
                        sub.user.is_premium = False
                        sub.user.save(update_fields=["is_premium"])
                    logger.info(f"Webhook: Subscription {razorpay_sub_id} marked as cancelled.")
            return Response({"status": "ok", "message": "Subscription status updated."}, status=drf_status.HTTP_200_OK)

        elif event_type == "payment.failed":
            razorpay_order_id = payment_entity.get("order_id")
            if razorpay_order_id:
                Payment.objects.filter(razorpay_order_id=razorpay_order_id).update(
                    status="failed",
                    razorpay_payment_id=payment_entity.get("id") or "",
                    updated_at=timezone.now()
                )
            return Response({"status": "ok"}, status=drf_status.HTTP_200_OK)

        return Response({"status": "ignored"}, status=drf_status.HTTP_200_OK)


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
    duration_days = plan.duration_days if plan else 30
    end_date = now + timedelta(days=duration_days)

    tokens_to_allocate = plan.token_quota if (plan and plan.token_quota > 0) else 2000000

    try:
        subscription = Subscription.objects.get(user=user)
        subscription.plan = plan
        subscription.plan_type = plan.slug if plan else "pro"
        subscription.status = "active"
        if subscription.end_date and subscription.end_date > now:
            subscription.end_date = subscription.end_date + timedelta(days=duration_days)
        else:
            subscription.start_date = now
            subscription.end_date = end_date
        subscription.tokens_allocated = F("tokens_allocated") + tokens_to_allocate
        subscription.tokens_granted_at = now
        subscription.save()
    except Subscription.DoesNotExist:
        subscription = Subscription.objects.create(
            user=user,
            plan=plan,
            plan_type=plan.slug if plan else "pro",
            status="active",
            start_date=now,
            end_date=end_date,
            tokens_allocated=tokens_to_allocate,
            tokens_used=0,
            tokens_granted_at=now,
        )

    # Update the user's premium flag
    user.is_premium = True
    user.save(update_fields=["is_premium"])
    return subscription

@login_required(login_url="login")
def usage_status(request):
    return JsonResponse(get_usage_snapshot(request.user))