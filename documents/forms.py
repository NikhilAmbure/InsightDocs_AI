from django import forms

from .models import Document
from .models import ChatMessage

class DocumentUploadForm(forms.ModelForm):
    title = forms.CharField(required=False, widget=forms.HiddenInput())
    file = forms.FileField(required=True)

    class Meta:
        model = Document
        fields = ("file", "title")

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