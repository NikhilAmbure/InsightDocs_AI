from django.db import models
from django.contrib.auth.models import AbstractUser
import random
from django.utils import timezone
import datetime

class User(AbstractUser):
    email_verified = models.BooleanField(default=False)

