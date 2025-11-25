from django.contrib.auth import login, logout
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import JsonResponse
from django.shortcuts import redirect, render

import random

from .emailer import sendOTPToEmail
from .forms import ProfileUpdateForm
from .models import User

from documents.models import Document


def register_view(request):
    """
    Handles user registration and sending OTP to email.
    Actual OTP verification and user creation are handled in verify_otp_view.
    """
    if request.method == 'POST':
        # Initial registration form submission
        username = request.POST.get('username')
        email = request.POST.get('email')
        password = request.POST.get('password')

        # Basic validation
        if not username or not email or not password:
            messages.error(request, "All fields are required.")
            return render(request, 'signup.html')

        if User.objects.filter(username=username).exists():
            messages.error(request, "Username already taken.")
            return render(request, 'signup.html')

        if User.objects.filter(email=email).exists():
            messages.error(request, "Email already in use.")
            return render(request, 'signup.html')

        # Store registration data in session
        request.session['registration_data'] = {
            'username': username,
            'email': email,
            'password': password,
        }

        # Generate and send OTP
        otp = random.randint(100000, 999999)
        request.session['registration_otp'] = otp
        subject = "Verify your InsightDocs AI Account"
        sendOTPToEmail(email, subject, otp) 

        messages.success(request, "We have sent a 6-digit OTP to your email.")
        # Redirect to the dedicated OTP verification page
        return redirect('verify_otp')

    # GET request → show signup form
    return render(request, 'signup.html')

# ----------------- Login & Dashboard -----------------
def login_view(request):
    if request.method == 'POST':
        email = request.POST.get('email')
        password = request.POST.get('password')

        try:
            user = User.objects.get(email=email)
            if user.check_password(password):
                login(request, user)
                return redirect('upload')
            else:
                messages.error(request, "Invalid password")
        except User.DoesNotExist:
            messages.error(request, "User does not exist")

        return redirect('login')

    return render(request, 'login.html')


def logout_view(request):
    logout(request)
    return redirect('login')


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

    return render(request, 'profile.html', {'form': form, 'recent_documents': documents})


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
            return redirect('verify_otp')

        try:
            if int(entered_otp) == int(stored_otp):
                # Create the user
                user = User(
                    username=registration_data['username'],
                    email=registration_data['email'],
                )
                user.set_password(registration_data['password'])
                user.save()

                # Clear session data
                request.session.pop('registration_data', None)
                request.session.pop('registration_otp', None)

                messages.success(request, "Account created successfully. Please login.")
                return redirect('login')
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
        sendOTPToEmail(email, subject, otp)
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

        return redirect('verify_reset_otp')

    # GET request → show OTP input form
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
    sendOTPToEmail(email, subject, otp)

    return JsonResponse({'message': 'A new verification code has been sent to your email.'})


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

    # GET request → show new password form
    return render(request, 'otp_reset_template.html', {'mode': 'password_reset'})
