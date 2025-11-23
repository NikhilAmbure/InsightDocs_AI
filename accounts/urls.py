from django.urls import path
from . import views

urlpatterns = [
    # Authentication
    path('signup/', views.register_view, name='signup'),
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    path('profile/', views.profile_view, name='profile'),

    # Registration OTP verification
    path('verify-otp/', views.verify_otp_view, name='verify_otp'),

    # Password reset (email + OTP + new password)
    path('password-reset/', views.password_reset_request, name='password_reset'),
    path('verify-reset-otp/', views.verify_reset_otp, name='verify_reset_otp'),
    path('resend-reset-otp/', views.resend_reset_otp, name='resend_reset_otp'),
    path('reset-password/', views.reset_password, name='reset_password'),
]
