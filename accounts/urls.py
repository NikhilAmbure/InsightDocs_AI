from django.urls import path
from . import views

urlpatterns = [
    # Authentication
    path('signup/', views.register_view, name='signup'),
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    path('profile/', views.profile_view, name='profile'),
    path('google/login/', views.google_login, name='google_login'),
    path('google/callback/', views.google_callback, name='google_callback'),

    # Registration OTP verification
    path('verify-otp/', views.verify_otp_view, name='verify_otp'),

    # 2FA Login Verification
    path('verify-login-2fa/', views.verify_login_2fa_view, name='verify_login_2fa'),

    # 2FA Management (Enable/Disable)
    path('2fa/enable/init/', views.enable_2fa_init_view, name='enable_2fa_init'),
    path('2fa/enable/verify/', views.verify_enable_2fa_otp_view, name='verify_enable_2fa_otp'),
    path('2fa/disable/init/', views.disable_2fa_init_view, name='disable_2fa_init'),
    path('2fa/disable/verify/', views.verify_disable_2fa_otp_view, name='verify_disable_2fa_otp'),

    # Password reset (email + OTP + new password)
    path('password-reset/', views.password_reset_request, name='password_reset'),
    path('verify-reset-otp/', views.verify_reset_otp, name='verify_reset_otp'),
    path('resend-reset-otp/', views.resend_reset_otp, name='resend_reset_otp'),
    path('reset-password/', views.reset_password, name='reset_password'),
]
