from celery import shared_task
from celery.utils.log import get_task_logger
from .emailer import sendOTPToEmail

logger = get_task_logger(__name__)

@shared_task
def send_otp_email_task(email, subject, otp):
    try:
        logger.info(f"Sending OTP to {email}")
        sendOTPToEmail(email, subject, otp)
        logger.info(f"OTP sent successfully to {email}")
        return True
    except Exception as e:
        logger.error(f"Failed to send OTP email to {email}: {str(e)}")
        return False