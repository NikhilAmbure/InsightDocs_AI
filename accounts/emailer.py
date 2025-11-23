from django.core.mail import send_mail, EmailMultiAlternatives
from django.conf import settings
from django.template.loader import render_to_string
from django.utils.html import strip_tags
from decimal import Decimal

def sendOTPToEmail(email, subject, message):
    send_mail(subject, message, settings.EMAIL_HOST_USER, [email])