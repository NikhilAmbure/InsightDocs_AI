import resend
from django.template.loader import render_to_string
from django.utils.html import strip_tags

resend.api_key = os.getenv("RESEND_API_KEY")

def sendOTPToEmail(email, subject, otp):
    """
    Sends OTP using Resend API (works on Railway free tier).
    """
    # Prepare HTML content
    html_content = render_to_string('emails/otp_email.html', {'otp': otp})
    text_content = strip_tags(html_content)

    resend.Emails.send({
        "from": "InsightDocs <onboarding@resend.dev>",
        "to": email,
        "subject": subject,
        "html": html_content,
        "text": text_content,
    })
