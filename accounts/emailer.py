from django.core.mail import EmailMultiAlternatives
from django.utils.html import strip_tags
from django.template.loader import render_to_string
from django.conf import settings
import logging

logger = logging.getLogger(__name__)

def sendOTPToEmail(email, subject, otp):
    try:
        logger.info(f"Starting to send OTP to {email}")
        logger.info(f"EMAIL_HOST_USER: {settings.EMAIL_HOST_USER}")
        logger.info(f"EMAIL_HOST: {settings.EMAIL_HOST}")
        logger.info(f"DEFAULT_FROM_EMAIL: {settings.DEFAULT_FROM_EMAIL}")
        
        html_content = render_to_string('emails/otp_email.html', {'otp': otp})
        text_content = strip_tags(html_content)

        email_message = EmailMultiAlternatives(
            subject,
            text_content,
            settings.DEFAULT_FROM_EMAIL,
            [email]
        )

        email_message.attach_alternative(html_content, "text/html")
        result = email_message.send()
        
        logger.info(f"Email sent successfully to {email}. Result: {result}")
        
    except Exception as e:
        logger.error(f"Failed to send OTP to {email}: {str(e)}", exc_info=True)
        raise