from django.contrib import admin

from .models import ChatMessage, ChatSession, Document


@admin.register(Document)
class DocumentAdmin(admin.ModelAdmin):
    list_display = ("title", "owner", "uploaded_at")
    search_fields = ("title", "owner__username", "original_name")
    list_filter = ("uploaded_at",)


class ChatMessageInline(admin.TabularInline):
    model = ChatMessage
    extra = 0
    readonly_fields = ("role", "content", "created_at")


@admin.register(ChatSession)
class ChatSessionAdmin(admin.ModelAdmin):
    list_display = ("document", "user", "updated_at")
    search_fields = ("document__title", "user__username")
    inlines = (ChatMessageInline,)
