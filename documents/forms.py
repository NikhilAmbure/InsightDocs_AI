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

    def __init__(self, *args, **kwargs):
        self.user = kwargs.pop("user", None)
        super().__init__(*args, **kwargs)

    def clean_file(self):
        uploaded_file = self.cleaned_data.get("file")
        if not uploaded_file:
            return uploaded_file

        if self.user and self.user.is_premium:
            max_size = 50 * 1024 * 1024  # 50 MB for premium users
            limit_lable = "50 MB (Premium)"
        else:
            max_size = 10 * 1024 * 1024  # 10 MB for free
            limit_lable = "10 MB (Free Tier)"

        if uploaded_file.size > max_size:
            raise ValidationError(
                f"File is too large. Maximum allowed size is {limit_lable} MB."
                f"Current file size is {uploaded_file.size / (1024 * 1024):.2f} MB."
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
    
    def save(self, commit=True):
        instance = super().save(commit=False)

        uploaded_file = self.cleaned_data.get("file")

        if self.cleaned_data.get("file"):
            f = self.cleaned_data["file"]
            # Read file content into the BinaryField for DB storage
            if hasattr(f, 'seek'):
                f.seek(0)
            instance.file_content = f.read()

            # Reset cursor for standard save
            if hasattr(f, 'seek'):
                f.seek(0)

            if commit: 
                instance.save()
            
        return instance

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