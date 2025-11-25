import os

from django import forms
from django.conf import settings
from django.core.exceptions import ValidationError

from .models import Document
from .models import ChatMessage

class DocumentUploadForm(forms.ModelForm):
    title = forms.CharField(required=False, widget=forms.HiddenInput())
    file = forms.FileField(required=True)

    class Meta:
        model = Document
        fields = ("file", "title")

    def clean_file(self):
        uploaded_file = self.cleaned_data.get("file")
        if not uploaded_file:
            return uploaded_file

        max_size = getattr(settings, "MAX_UPLOAD_SIZE", 15 * 1024 * 1024)
        if uploaded_file.size > max_size:
            raise ValidationError(
                f"File is too large. Maximum allowed size is {max_size // (1024 * 1024)} MB."
            )

        allowed_extensions = {
            ext.lower().lstrip(".")
            for ext in getattr(settings, "ALLOWED_UPLOAD_EXTENSIONS", [])
        }
        _, ext = os.path.splitext(uploaded_file.name)
        normalized_ext = ext.lower().lstrip(".")

        if allowed_extensions and normalized_ext not in allowed_extensions:
            raise ValidationError(
                f"Unsupported file type '.{normalized_ext}'. Allowed types: {', '.join(sorted(allowed_extensions))}."
            )

        return uploaded_file

    def clean(self):
        cleaned_data = super().clean()
        uploaded_file = cleaned_data.get("file")
        title = cleaned_data.get("title")

        if uploaded_file and not title:
            cleaned_data["title"] = uploaded_file.name

        return cleaned_data

class ChatMessageForm(forms.ModelForm):
    class Meta:
        model = ChatMessage
        fields = ['content']
        widgets = {
            'content': forms.TextInput(attrs={
                'placeholder': 'Ask a question about this document...',
                'class': 'flex-1 bg-transparent py-3 text-sm text-white placeholder:text-zinc-600 focus:outline-none'
            })
        }