from django.contrib.auth import login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.hashers import make_password
from django.contrib import messages
from django.core.exceptions import ValidationError
from django.core.validators import validate_email
from django.http import JsonResponse
from django.shortcuts import redirect, render

import random
import logging

from .emailer import sendOTPToEmail
from .forms import ProfileUpdateForm
from .models import User

from documents.models import Document
from .tasks import send_otp_email_task

logger = logging.getLogger(__name__)


def _dispatch_otp_email(email, subject, otp):
    """
    Safely dispatch OTP email via Celery background task if available,
    falling back to synchronous delivery via Resend API immediately
    if Celery or Redis is unreachable.
    """
    try:
        send_otp_email_task.delay(email, subject, otp)
    except Exception as e:
        logger.warning(f"Celery dispatch failed for {email}, falling back to direct delivery: {e}")
        sendOTPToEmail(email, subject, otp)


# ----------------- Register -----------------
def register_view(request):
    """
    Handles user registration and sending OTP to email.
    Securely stores only password hash in session during verification.
    """
    if request.method == 'POST':
        username = (request.POST.get('username') or '').strip()
        email = (request.POST.get('email') or '').strip().lower()
        password = request.POST.get('password')

        # Basic validation
        if not username or not email or not password:
            messages.error(request, "All fields are required.")
            return render(request, 'signup.html')

        try:
            validate_email(email)
        except ValidationError:
            messages.error(request, "Please enter a valid email address.")
            return render(request, 'signup.html')

        if len(password) < 8:
            messages.error(request, "Password must be at least 8 characters long.")
            return render(request, 'signup.html')

        if User.objects.filter(username=username).exists():
            messages.error(request, "Username already taken.")
            return render(request, 'signup.html')

        if User.objects.filter(email=email).exists():
            messages.error(request, "Email already in use.")
            return render(request, 'signup.html')

        # Store cryptographic password hash in session instead of plaintext credentials
        request.session['registration_data'] = {
            'username': username,
            'email': email,
            'password_hash': make_password(password),
        }

        # Generate and send OTP
        otp = random.randint(100000, 999999)
        request.session['registration_otp'] = otp
        subject = "Verify your InsightDocs AI Account"

        try:
            _dispatch_otp_email(email, subject, otp)
            messages.success(request, "We have sent a 6-digit OTP to your email.")
            return redirect('verify_otp')
        except Exception as e:
            logger.error(f"Failed to send OTP to {email}: {str(e)}")
            messages.error(request, "Failed to send OTP. Please try again.")
            return render(request, 'signup.html')

    return render(request, 'signup.html')

# ----------------- Login & 2FA Logic -----------------
def login_view(request):
    if request.method == 'POST':
        email = request.POST.get('email')
        password = request.POST.get('password')

        try:
            user = User.objects.get(email=email)
            if user.check_password(password):
                # CHECK 2FA STATUS
                if user.is_2fa_enabled:
                    otp = random.randint(100000, 999999)

                    request.session['pre_2fa_login_user_id'] = user.id
                    request.session['login_2fa_otp'] = otp

                    subject = "Login Verification Code"
                    
                    # Safe dispatch (Celery with synchronous Resend fallback)
                    _dispatch_otp_email(user.email, subject, otp)

                    messages.success(request, "Enter the code sent to your email to log in.")
                    return redirect('verify_login_2fa')
                else:
                    # Standard Login
                    login(request, user)
                    return redirect('upload')
            else:
                messages.error(request, "Invalid password")
        except User.DoesNotExist:
            messages.error(request, "User does not exist")

        return render(request, 'login.html')

    return render(request, 'login.html')

def verify_login_2fa_view(request):
    """
    Step 2 of Login: User enters OTP to finalize login.
    """
    user_id = request.session.get('pre_2fa_login_user_id')
    stored_otp = request.session.get('login_2fa_otp')

    if not user_id or not stored_otp:
        messages.error(request, "Session expired. Please log in again.")
        return redirect('login')

    if request.method == 'POST':
        entered_otp = request.POST.get('otp')
        
        try:
            if int(entered_otp) == int(stored_otp):
                # Verify Success
                user = User.objects.get(id=user_id)
                login(request, user)
                
                # Cleanup session
                request.session.pop('pre_2fa_login_user_id', None)
                request.session.pop('login_2fa_otp', None)
                
                return redirect('upload')
            else:
                messages.error(request, "Invalid code. Please try again.")
        except (ValueError, User.DoesNotExist):
            messages.error(request, "Invalid input or user not found.")

    return render(request, 'otp_reset_template.html', {'mode': 'otp_verification', 'action_url': 'verify_login_2fa'})


# ----------------- 2FA Management (Enable/Disable) -----------------
@login_required(login_url='login')
def enable_2fa_init_view(request):
    """
    Step 1 Enable: Ask user to confirm/enter email for 2FA.
    """
    if request.method == 'POST':
        email = request.POST.get('email')
        
        if not email:
            messages.error(request, "Email is required.")
            return redirect('enable_2fa_init')
            
        if email != request.user.email:
             messages.warning(request, "For security, we are using your account email.")
             email = request.user.email

        # Generate OTP
        otp = random.randint(100000, 999999)
        request.session['enable_2fa_data'] = {'email': email, 'otp': otp}
        
        subject = "Verify 2FA Setup"
        
        # Safe dispatch
        _dispatch_otp_email(email, subject, otp)
        
        messages.success(request, "We sent a code to your email to enable 2FA.")
        return redirect('verify_enable_2fa_otp')

    return render(request, 'two_factor/enable_2fa_email.html')

@login_required(login_url='login')
def verify_enable_2fa_otp_view(request):
    """
    Step 2 Enable: Verify OTP and set is_2fa_enabled = True.
    """
    data = request.session.get('enable_2fa_data')
    if not data:
        messages.error(request, "Setup session expired. Please start again.")
        return redirect('profile')

    if request.method == 'POST':
        entered_otp = request.POST.get('otp')
        try:
            if int(entered_otp) == int(data['otp']):
                request.user.is_2fa_enabled = True
                request.user.save()
                
                request.session.pop('enable_2fa_data', None)
                messages.success(request, "Two-Factor Authentication has been ENABLED.")
                return redirect('profile')
            else:
                messages.error(request, "Invalid code.")
        except ValueError:
            messages.error(request, "Invalid code format.")

    return render(request, 'otp_reset_template.html', {
        'mode': 'otp_verification', 
        'action_url': 'verify_enable_2fa_otp',
        'subtitle': 'Verify to Enable 2FA' 
    })

@login_required(login_url='login')
def disable_2fa_init_view(request):
    """
    Step 1 Disable: Send OTP to verify user before disabling.
    """
    otp = random.randint(100000, 999999)
    request.session['disable_2fa_otp'] = otp
    
    subject = "Verify 2FA Deactivation"

    # Safe dispatch
    _dispatch_otp_email(request.user.email, subject, otp)   

    messages.success(request, "We sent a code to your email to confirm deactivation.")
    return redirect('verify_disable_2fa_otp')

@login_required(login_url='login')
def verify_disable_2fa_otp_view(request):
    """
    Step 2 Disable: Verify OTP and set is_2fa_enabled = False.
    """
    stored_otp = request.session.get('disable_2fa_otp')
    if not stored_otp:
        messages.error(request, "Session expired.")
        return redirect('profile')

    if request.method == 'POST':
        entered_otp = request.POST.get('otp')
        try:
            if int(entered_otp) == int(stored_otp):
                request.user.is_2fa_enabled = False
                request.user.save()
                
                request.session.pop('disable_2fa_otp', None)
                messages.success(request, "Two-Factor Authentication has been DISABLED.")
                return redirect('profile')
            else:
                messages.error(request, "Invalid code.")
        except ValueError:
            messages.error(request, "Invalid code format.")

    return render(request, 'otp_reset_template.html', {
        'mode': 'otp_verification', 
        'action_url': 'verify_disable_2fa_otp',
        'subtitle': 'Verify to Disable 2FA'
    })

# ----------------- Logout ----------------- 
def logout_view(request):
    logout(request)
    return redirect('login')

# ----------------- Profile -----------------
@login_required(login_url='login')
def profile_view(request):
    """
    Allow the authenticated user to view and update their profile.
    """
    if request.method == 'POST':
        form = ProfileUpdateForm(request.POST, instance=request.user)
        if form.is_valid():
            form.save()
            messages.success(request, "Profile updated successfully.")
            return redirect('profile')
        else:
            messages.error(request, "Please fix the errors below.")
    else:
        form = ProfileUpdateForm(instance=request.user)
        
    documents = Document.objects.filter(owner=request.user).order_by('-uploaded_at')

    # Fetch subscription info and usage for the profile page
    from payments.models import Subscription, Payment
    from payments.services import get_usage_snapshot, ensure_default_plans

    plans = ensure_default_plans()
    subscription = Subscription.objects.filter(user=request.user).select_related('plan').first()
    
    # If user is marked premium but has no subscription record, auto-link to Pro plan
    if not subscription and request.user.is_premium:
        pro_plan = plans.get('pro')
        if pro_plan:
            subscription, _ = Subscription.objects.get_or_create(
                user=request.user,
                defaults={
                    'plan': pro_plan,
                    'plan_type': 'pro',
                    'status': 'active',
                    'tokens_allocated': pro_plan.token_quota,
                    'tokens_used': 0,
                }
            )

    usage = get_usage_snapshot(request.user)
    recent_payments = Payment.objects.filter(user=request.user).select_related('plan').order_by('-created_at')[:5]

    doc_count = documents.count()
    if request.user.is_premium:
        doc_limit = subscription.plan.doc_limit if (subscription and subscription.plan) else 50
        chat_limit = subscription.plan.chat_limit if (subscription and subscription.plan) else 5000
    else:
        doc_limit = 10
        chat_limit = 500

    return render(request, 'profile.html', {
        'form': form,
        'recent_documents': documents,
        'subscription': subscription,
        'usage': usage,
        'recent_payments': recent_payments,
        'doc_count': doc_count,
        'doc_limit': doc_limit,
        'chat_limit': chat_limit,
    })


# ----------------- OTP Verification (registration) -----------------
def verify_otp_view(request):
    if request.method == 'POST':
        entered_otp = request.POST.get('otp')
        stored_otp = request.session.get('registration_otp')
        registration_data = request.session.get('registration_data')

        # Ensure there is a pending registration
        if not stored_otp or not registration_data:
            messages.error(request, "Registration session expired. Please sign up again.")
            return redirect('signup')

        if not entered_otp:
            messages.error(request, "Please enter the OTP.")
            # Using render instead of redirect -> to show error immediately on the form
            return render(request, 'verify_otp.html')

        try:
            if int(entered_otp) == int(stored_otp):
                password_hash = registration_data.get('password_hash')
                raw_password = registration_data.get('password')

                user = User(
                    username=registration_data['username'],
                    email=registration_data['email'],
                )
                if password_hash:
                    user.password = password_hash
                elif raw_password:
                    user.set_password(raw_password)

                user.email_verified = True
                user.is_2fa_enabled = False
                user.save()

                # --- Auto Login the user immediately ---
                login(request, user)

                # Clear session data
                request.session.pop('registration_data', None)
                request.session.pop('registration_otp', None)

                return redirect('upload')
            else:
                messages.error(request, "Invalid OTP. Please try again.")
        except ValueError:
            messages.error(request, "OTP must be a 6-digit number.")

        return redirect('verify_otp')

    # GET request → show OTP page only if registration is in progress
    if not request.session.get('registration_otp') or not request.session.get('registration_data'):
        messages.error(request, "No registration in progress. Please sign up first.")
        return redirect('signup')

    return render(request, 'verify_otp.html')


# ----------------- Password reset (email + OTP + new password) -----------------
def password_reset_request(request):
    """
    Step 1: User submits email to receive a password reset OTP.
    Uses template: forgot-password.html
    """
    if request.method == 'POST':
        email = request.POST.get('email')

        if not email:
            messages.error(request, "Please enter your email address.")
            return render(request, 'forgot-password.html')

        try:
            user = User.objects.get(email=email)
        except User.DoesNotExist:
            # Do not reveal whether the email exists
            messages.success(
                request,
                "If an account with that email exists, we've sent a verification code."
            )
            return redirect('password_reset')

        # Generate and store OTP in session
        otp = random.randint(100000, 999999)
        request.session['reset_password_data'] = {
            'email': email,
            'otp': otp,
        }
        subject = "Reset your Password"

        # Safe dispatch (Celery with synchronous Resend fallback)
        _dispatch_otp_email(email, subject, otp)

        messages.success(request, "We sent a 6-digit verification code to your email.")

        return redirect('verify_reset_otp')

    # GET request → show email input form
    return render(request, 'forgot-password.html')


def verify_reset_otp(request):
    """
    Step 2: User enters the OTP they received to verify password reset.
    Uses template: otp_reset_template.html with mode='otp_verification'
    """
    reset_data = request.session.get('reset_password_data')
    if not reset_data:
        messages.error(request, "No password reset request found. Please start again.")
        return redirect('password_reset')

    if request.method == 'POST':
        entered_otp = request.POST.get('otp')

        if not entered_otp:
            messages.error(request, "Please enter the verification code.")
            return redirect('verify_reset_otp')

        try:
            if int(entered_otp) == int(reset_data.get('otp')):
                # Mark OTP as verified
                request.session['reset_password_verified'] = True
                messages.success(request, "Verification successful. Please set a new password.")
                return redirect('reset_password')
            else:
                messages.error(request, "Invalid verification code. Please try again.")
        except ValueError:
            messages.error(request, "Verification code must be a 6-digit number.")

        return render(request, 'otp_reset_template.html', {'mode': 'otp_verification'})

    return render(request, 'otp_reset_template.html', {'mode': 'otp_verification'})


def resend_reset_otp(request):
    """
    AJAX endpoint used by otp_reset_template.html to resend the reset OTP.
    Returns JSON.
    """
    reset_data = request.session.get('reset_password_data')
    if not reset_data:
        return JsonResponse({'error': 'No password reset request in progress.'}, status=400)

    email = reset_data.get('email')
    if not email:
        return JsonResponse({'error': 'Invalid reset session data.'}, status=400)

    # Generate a new OTP and update session
    otp = random.randint(100000, 999999)
    reset_data['otp'] = otp
    request.session['reset_password_data'] = reset_data
    subject = "New Password Reset Code"
    
    try:
        _dispatch_otp_email(email, subject, otp)
        return JsonResponse({'message': 'A new verification code has been sent to your email.'})
    except Exception as e:
        logger.error(f"Failed to resend OTP to {email}: {str(e)}")
        return JsonResponse({'error': 'Failed to send verification code. Please try again.'}, status=500)


def reset_password(request):
    """
    Step 3: After OTP verification, user sets a new password.
    Uses template: otp_reset_template.html with mode='password_reset'
    """
    reset_data = request.session.get('reset_password_data')
    verified = request.session.get('reset_password_verified')

    if not reset_data or not verified:
        messages.error(request, "Password reset session expired. Please start again.")
        return redirect('password_reset')

    if request.method == 'POST':
        new_password = request.POST.get('new_password')
        confirm_password = request.POST.get('confirm_password')

        if not new_password or not confirm_password:
            messages.error(request, "Please fill in all password fields.")
            return redirect('reset_password')

        if new_password != confirm_password:
            messages.error(request, "Passwords do not match.")
            return redirect('reset_password')

        email = reset_data.get('email')
        try:
            user = User.objects.get(email=email)
        except User.DoesNotExist:
            messages.error(request, "User not found. Please try again.")
            return redirect('password_reset')

        user.set_password(new_password)
        user.save()

        # Clean up reset session data
        request.session.pop('reset_password_data', None)
        request.session.pop('reset_password_verified', None)

        messages.success(request, "Your password has been reset successfully. Please login.")
        return redirect('login')

    return render(request, 'otp_reset_template.html', {'mode': 'password_reset'})


def _get_google_redirect_uri(request):
    """
    Returns the exact redirect URI configured for Google OAuth.
    If GOOGLE_OAUTH_REDIRECT_URI is explicitly set in settings/env AND matches the current host, use it.
    Otherwise, builds it dynamically from the current request host and scheme so that
    it always matches the exact domain and session cookie the user is currently browsing.
    """
    from django.conf import settings
    from django.urls import reverse
    from urllib.parse import urlparse, urlunparse

    curr_host = request.get_host().split(':')[0].lower()
    custom_uri = getattr(settings, 'GOOGLE_OAUTH_REDIRECT_URI', '')
    if custom_uri:
        custom_host = urlparse(custom_uri).netloc.split(':')[0].lower()
        if custom_host == curr_host:
            uri = custom_uri.strip()
            if not uri.endswith('/'):
                uri += '/'
            return uri

    redirect_uri = request.build_absolute_uri(reverse('google_callback'))
    parsed = urlparse(redirect_uri)
    scheme = parsed.scheme
    netloc = parsed.netloc

    # Strip default ports (:80, :443) from netloc if added by reverse proxies
    if ':' in netloc:
        host_part, port_part = netloc.split(':', 1)
        if (scheme == 'https' and port_part == '443') or (scheme == 'http' and port_part == '80'):
            netloc = host_part

    # Ensure https on production / non-localhost
    if curr_host not in ('localhost', '127.0.0.1'):
        scheme = 'https'

    path = parsed.path
    if not path.endswith('/'):
        path += '/'

    redirect_uri = urlunparse((scheme, netloc, path, parsed.params, parsed.query, parsed.fragment))
    return redirect_uri


def google_login(request):
    """
    Redirects the user to Google's OAuth2 authorization page.
    """
    from django.conf import settings
    from django.utils.crypto import get_random_string
    from urllib.parse import urlencode

    client_id = getattr(settings, 'GOOGLE_OAUTH_CLIENT_ID', '')
    if not client_id:
        messages.error(request, "Google OAuth is not configured. Please contact support.")
        return redirect('login')

    redirect_uri = _get_google_redirect_uri(request)
    logger.info(f"Google OAuth initiated. redirect_uri={redirect_uri}")

    state = get_random_string(32)
    request.session['google_oauth_state'] = state
    try:
        request.session.modified = True
        request.session.save()
    except (AttributeError, Exception):
        pass

    params = {
        'response_type': 'code',
        'client_id': client_id,
        'redirect_uri': redirect_uri,
        'scope': 'openid email profile',
        'state': state,
    }
    auth_url = f"https://accounts.google.com/o/oauth2/v2/auth?{urlencode(params)}"

    curr_host = request.get_host().split(':')[0].lower()
    is_secure = request.is_secure() or (curr_host not in ('localhost', '127.0.0.1'))

    response = redirect(auth_url)
    response.set_cookie(
        'google_oauth_state',
        state,
        max_age=300,
        httponly=True,
        samesite='Lax',
        secure=is_secure,
    )
    return response


def google_callback(request):
    """
    Handles the callback from Google, exchanges the auth code for access token,
    retrieves user details, and logs the user in (creating account if needed).
    """
    import requests
    from django.conf import settings
    from django.utils.crypto import get_random_string

    state = request.GET.get('state')
    code = request.GET.get('code')
    error = request.GET.get('error')

    if error or not code:
        messages.error(request, f"Google login failed: {error or 'No code provided'}")
        response = redirect('login')
        response.delete_cookie('google_oauth_state')
        return response

    saved_state = request.session.pop('google_oauth_state', None)
    if not saved_state:
        saved_state = request.COOKIES.get('google_oauth_state')

    if not state or state != saved_state:
        logger.warning(f"OAuth state mismatch. Received state={state}, Saved state={saved_state}")
        messages.error(request, "Security verification failed (Invalid state). Please try again.")
        response = redirect('login')
        response.delete_cookie('google_oauth_state')
        return response

    client_id = getattr(settings, 'GOOGLE_OAUTH_CLIENT_ID', '')
    client_secret = getattr(settings, 'GOOGLE_OAUTH_CLIENT_SECRET', '')
    redirect_uri = _get_google_redirect_uri(request)

    token_url = "https://oauth2.googleapis.com/token"
    data = {
        'code': code,
        'client_id': client_id,
        'client_secret': client_secret,
        'redirect_uri': redirect_uri,
        'grant_type': 'authorization_code'
    }

    try:
        token_response = requests.post(token_url, data=data, timeout=10)
        token_response_data = token_response.json()

        if 'error' in token_response_data:
            logger.error(f"Google token exchange error: {token_response_data}")
            messages.error(request, f"Token exchange failed: {token_response_data.get('error_description', '')}")
            return redirect('login')

        access_token = token_response_data.get('access_token')
        if not access_token:
            messages.error(request, "Failed to get access token from Google.")
            return redirect('login')

        user_info_url = "https://www.googleapis.com/oauth2/v3/userinfo"
        headers = {'Authorization': f'Bearer {access_token}'}
        user_info_response = requests.get(user_info_url, headers=headers, timeout=10)
        user_info = user_info_response.json()

        email = user_info.get('email')
        if not email:
            messages.error(request, "Failed to retrieve email address from Google.")
            return redirect('login')

        from django.core.validators import validate_email
        from django.core.exceptions import ValidationError
        try:
            validate_email(email)
        except ValidationError:
            messages.error(request, "Invalid email address returned from Google.")
            return redirect('login')

        try:
            user = User.objects.get(email=email)
        except User.DoesNotExist:
            username_prefix = email.split('@')[0]
            username = username_prefix
            counter = 1
            while User.objects.filter(username=username).exists():
                username = f"{username_prefix}{counter}"
                counter += 1

            user = User.objects.create(
                username=username,
                email=email,
                email_verified=True,
                is_2fa_enabled=False
            )
            user.set_password(get_random_string(32))
            user.save()

        login(request, user)
        messages.success(request, f"Welcome, {user.username}! Successfully logged in via Google.")
        response = redirect('upload')
        response.delete_cookie('google_oauth_state')
        return response

    except Exception as e:
        logger.error(f"Exception during Google callback: {str(e)}", exc_info=True)
        messages.error(request, f"Google login failed due to a server error. Please try again.")
        response = redirect('login')
        response.delete_cookie('google_oauth_state')
        return response