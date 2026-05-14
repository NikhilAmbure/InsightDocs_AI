from django.db import models
from django.contrib.auth.models import AbstractUser

class User(AbstractUser):
    email_verified = models.BooleanField(default=False)
    is_2fa_enabled = models.BooleanField(default=False)

    is_premium = models.BooleanField(default=False, help_text="Designates whether the user has a premium subscription.")