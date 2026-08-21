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

        if event_type in ["payment.captured", "subscription.charged"]:
            notes = payment_entity.get("notes", {}) or subscription_entity.get("notes", {}) or {}
            user_id = notes.get("user_id")
            plan_id = notes.get("plan_id")

            razorpay_order_id = payment_entity.get("order_id")
            if not user_id and razorpay_order_id:
                try:
                    payment = Payment.objects.get(razorpay_order_id=razorpay_order_id)
                    user = payment.user
                    plan = payment.plan
                except Payment.DoesNotExist:
                    pass

            razorpay_sub_id = subscription_entity.get("id") or payment_entity.get("subscription_id")
            if not user and razorpay_sub_id:
                try:
                    subscription = Subscription.objects.get(razorpay_subscription_id=razorpay_sub_id)
                    user = subscription.user
                    plan = subscription.plan
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
                user.is_premium = True
                user.save(update_fields=["is_premium"])

                now = timezone.now()
                from django.db.models import F
                duration_days = plan.duration_days if plan else 60
                end_date = now + timedelta(days=duration_days)

                sub, created = Subscription.objects.get_or_create(
                    user=user,
                    defaults={
                        "plan": plan,
                        "plan_type": plan.slug if plan else "pro",
                        "status": "active",
                        "start_date": now,
                        "end_date": end_date,
                        "tokens_allocated": 2000000,
                        "tokens_used": 0,
                        "tokens_granted_at": now,
                        "razorpay_subscription_id": razorpay_sub_id or "",
                    }
                )
                if not created:
                    sub.status = "active"
                    if plan:
                        sub.plan = plan
                        sub.plan_type = plan.slug
                    sub.end_date = end_date
                    sub.tokens_allocated = F('tokens_allocated') + 2000000
                    sub.tokens_granted_at = now
                    if razorpay_sub_id:
                        sub.razorpay_subscription_id = razorpay_sub_id
                    sub.save()

                if razorpay_order_id:
                    Payment.objects.filter(razorpay_order_id=razorpay_order_id).update(
                        status="captured",
                        razorpay_payment_id=payment_entity.get("id"),
                        updated_at=now
                    )

                logger.info(f"Webhook success: Added 2,000,000 tokens for user {user.username}")
                return Response({"status": "success", "message": "Tokens added successfully."}, status=drf_status.HTTP_200_OK)
            else:
                logger.warning("Webhook user identification failed.")
                return Response({"error": "User identification failed."}, status=drf_status.HTTP_400_BAD_REQUEST)

        elif event_type == "payment.failed":
            razorpay_order_id = payment_entity.get("order_id")
            if razorpay_order_id:
                Payment.objects.filter(razorpay_order_id=razorpay_order_id).update(
                    status="failed",
                    razorpay_payment_id=payment_entity.get("id"),
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
    end_date = now + timedelta(days=plan.duration_days)

    tokens_to_allocate = 2000000 if plan.slug == 'pro' else plan.token_quota

    try:
        subscription = Subscription.objects.get(user=user)
        subscription.plan = plan
        subscription.plan_type = plan.slug
        subscription.status = "active"
        subscription.end_date = end_date
        subscription.tokens_allocated = F("tokens_allocated") + tokens_to_allocate
        subscription.tokens_granted_at = now
        subscription.save()
    except Subscription.DoesNotExist:
        subscription = Subscription.objects.create(
            user=user,
            plan=plan,
            plan_type=plan.slug,
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