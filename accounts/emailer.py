from django.core.mail import EmailMultiAlternatives
from django.utils.html import strip_tags
from django.template.loader import render_to_string
from django.conf import settings

def sendOTPToEmail(email, subject, otp):
    html_content = render_to_string('emails/otp_email.html', {'otp': otp})
    text_content = strip_tags(html_content)

    email_message = EmailMultiAlternatives(
        subject,
        text_content,
        settings.DEFAULT_FROM_EMAIL,
        [email]
    )

    email_message.attach_alternative(html_content, "text/html")
    email_message.send()
