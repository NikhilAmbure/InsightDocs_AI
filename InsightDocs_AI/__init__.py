# Loading the Celery application instance
from .celery import app as celery_app

__all__ = ('celery_app',)