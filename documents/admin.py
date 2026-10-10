from django.contrib import admin

from .models import ChatMessage, ChatSession, Document, Workspace


@admin.register(Workspace)
class WorkspaceAdmin(admin.ModelAdmin):
    list_display = ("name", "owner", "document_count", "created_at", "updated_at")
    search_fields = ("name", "owner__username")
    list_filter = ("created_at",)

    def document_count(self, obj):
        return obj.documents.count()
    document_count.short_description = "Docs"


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
    list_display = ("id", "user", "document", "workspace", "updated_at")
    search_fields = ("document__title", "workspace__name", "user__username")
    inlines = (ChatMessageInline,)

