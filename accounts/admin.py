from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from .models import User


@admin.register(User)
class CustomUserAdmin(UserAdmin):
    list_display = (
        "username",
        "email",
        "is_premium",
        "email_verified",
        "is_2fa_enabled",
        "is_staff",
        "is_active",
    )
    list_filter = (
        "is_premium",
        "email_verified",
        "is_2fa_enabled",
        "is_staff",
        "is_superuser",
        "is_active",
    )
    search_fields = ("username", "email", "first_name", "last_name")
    ordering = ("username",)
    fieldsets = UserAdmin.fieldsets + (
        (
            "InsightDocs AI Profile",
            {
                "fields": (
                    "is_premium",
                    "email_verified",
                    "is_2fa_enabled",
                ),
            },
        ),
    )