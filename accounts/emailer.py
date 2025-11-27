from django.template.loader import render_to_string
from django.utils.html import strip_tags
import resend

def sendOTPToEmail(email, subject, otp):
    try:
        html_content = render_to_string('emails/otp_email.html', {'otp': otp})
        text_content = strip_tags(html_content)

        resend.Emails.send({
            "from": "InsightDocs <onboarding@resend.dev>",
            "to": email,
            "subject": subject,
            "html": html_content,
            "text": text_content,
        })
    except Exception as e:
        print("Resend email failed:", str(e))
