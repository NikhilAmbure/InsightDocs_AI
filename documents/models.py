import os

from django.conf import settings
from django.db import models
from django.utils.text import slugify
import uuid
from cloudinary_storage.storage import RawMediaCloudinaryStorage


class Document(models.Model):
    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="documents",
    )
    title = models.CharField(max_length=255)
    file = models.FileField(upload_to="documents/", storage=RawMediaCloudinaryStorage())
    original_name = models.CharField(max_length=255)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    

    class Meta:
        ordering = ("-uploaded_at",)

    def __str__(self) -> str:
        return f"{self.title} ({self.owner})"

    @property
    def extension(self) -> str:
        _, ext = os.path.splitext(self.file.name)
        return ext.lower().lstrip(".")

    @property
    def is_pdf(self) -> bool:
        return self.extension == "pdf"


class ChatSession(models.Model):
    document = models.ForeignKey(
        Document,
        on_delete=models.CASCADE,
        related_name="sessions",
    )
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="chat_sessions",
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = ("document", "user")
        ordering = ("-updated_at",)

    def __str__(self) -> str:
        return f"ChatSession<{self.user} · {self.document}>"


class ChatMessage(models.Model):
    ROLE_CHOICES = (
        ("user", "User"),
        ("assistant", "Assistant"),
    )

    session = models.ForeignKey(
        ChatSession,
        on_delete=models.CASCADE,
        related_name="messages",
    )
    role = models.CharField(max_length=16, choices=ROLE_CHOICES)
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ("created_at",)

    def __str__(self) -> str:
        return f"{self.role}: {self.content[:40]}..."
