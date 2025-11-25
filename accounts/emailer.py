from django.core.mail import EmailMultiAlternatives
from django.conf import settings
from django.template.loader import render_to_string
from django.utils.html import strip_tags

def sendOTPToEmail(email, subject, otp):
    """
    Sends an HTML email with the OTP code.
    """
    # Context data for the template
    context = {
        'otp': otp,
        'subject': subject
    }

    # Render HTML content
    html_content = render_to_string('emails/otp_email.html', context)
    
    # Create plain text version (strips HTML tags) for email clients that disable HTML
    text_content = strip_tags(html_content)

    # Create the email object
    email_message = EmailMultiAlternatives(
        subject,
        text_content, # Content for plain text clients
        settings.EMAIL_HOST_USER,
        [email]
    )
    
    # Attach HTML version
    email_message.attach_alternative(html_content, "text/html")
    
    # Send
    email_message.send()